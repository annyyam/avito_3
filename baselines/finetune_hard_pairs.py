import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from baselines.pairwise import (
    PairwiseDataset,
    make_pairwise_examples,
    ndcg_at_5,
    predict,
    read_jsonl,
    read_labels,
    read_pools,
    read_split,
    write_submission,
)


BASELINE_NDCG = 0.915292
BASELINE_HCVR = 0.003774


def read_hard_pairs(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def select_hard_pairs(
    rows,
    train_need_ids,
    max_pairs_per_need,
):
    """
    Берём только реальные ошибки:

    правильный кандидат имеет больший grade,
    но модель поставила его ниже неправильного.
    """
    train_need_ids = set(train_need_ids)
    by_need = defaultdict(list)

    for row in rows:
        if row["need_id"] not in train_need_ids:
            continue

        if not row.get("is_inversion", False):
            continue

        by_need[row["need_id"]].append(row)

    selected = []

    for need_id, need_rows in by_need.items():
        need_rows.sort(
            key=lambda row: (
                -float(row["hardness"]),
                -float(row["weight"]),
                row["pos_candidate_id"],
                row["neg_candidate_id"],
            )
        )

        for row in need_rows[:max_pairs_per_need]:
            selected.append({
                "need_id": row["need_id"],
                "pos_candidate_id":
                    row["pos_candidate_id"],
                "neg_candidate_id":
                    row["neg_candidate_id"],
                "weight": float(row["weight"]),
            })

    return selected


def merge_examples(regular_examples, hard_examples):
    """
    Удаляем дубли пар.

    Если одна пара есть и среди обычных, и среди сложных,
    сохраняем её один раз с максимальным весом.
    """
    merged = {}

    for source, examples in (
        ("regular", regular_examples),
        ("hard", hard_examples),
    ):
        for example in examples:
            key = (
                example["need_id"],
                example["pos_candidate_id"],
                example["neg_candidate_id"],
            )

            new_row = {
                "need_id": example["need_id"],
                "pos_candidate_id":
                    example["pos_candidate_id"],
                "neg_candidate_id":
                    example["neg_candidate_id"],
                "weight": float(example["weight"]),
                "source": source,
            }

            old_row = merged.get(key)

            if old_row is None:
                merged[key] = new_row
            else:
                old_row["weight"] = max(
                    old_row["weight"],
                    new_row["weight"],
                )

                if source == "hard":
                    old_row["source"] = "hard"

    return list(merged.values())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default=(
            "/home/User25/avito_airi_3_case/"
            "student_package/data"
        ),
    )
    parser.add_argument(
        "--split",
        default=(
            "/home/User25/avito_3_recovered/"
            "splits/need_split.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/home/User25/outputs/"
            "bge_m3_pairwise/best"
        ),
    )
    parser.add_argument(
        "--hard-pairs",
        default=(
            "/home/User25/outputs/"
            "bge_m3_hard_negatives/"
            "hard_pairs.train.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/User25/outputs/"
            "bge_m3_hard_finetune_v1"
        ),
    )

    parser.add_argument(
        "--hard-pairs-per-need",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--regular-pairs-per-need",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    need_rows = read_jsonl(
        data_dir / "needs.train.jsonl"
    )
    provider_rows = read_jsonl(
        data_dir / "providers.jsonl"
    )

    needs = {
        row["need_id"]: row["need_text"]
        for row in need_rows
    }

    providers = {
        row["candidate_id"]: row["profile_summary"]
        for row in provider_rows
    }

    pools = read_pools(
        data_dir / "candidate_pools.train.csv"
    )
    labels = read_labels(
        data_dir / "labels.train.csv"
    )
    split_by_need = read_split(args.split)

    train_need_ids = [
        need_id
        for need_id in needs
        if split_by_need.get(need_id) == "train"
        and need_id in pools
    ]

    dev_need_ids = [
        need_id
        for need_id in needs
        if split_by_need.get(need_id) == "dev"
        and need_id in pools
    ]

    regular_examples = make_pairwise_examples(
        need_ids=train_need_ids,
        needs=needs,
        providers=providers,
        pools=pools,
        labels=labels,
        hard_penalty=0.0,
        max_pairs_per_need=(
            args.regular_pairs_per_need
        ),
        seed=args.seed,
    )

    all_hard_rows = read_hard_pairs(
        args.hard_pairs
    )

    hard_examples = select_hard_pairs(
        rows=all_hard_rows,
        train_need_ids=train_need_ids,
        max_pairs_per_need=(
            args.hard_pairs_per_need
        ),
    )

    train_examples = merge_examples(
        regular_examples=regular_examples,
        hard_examples=hard_examples,
    )

    random.Random(args.seed).shuffle(
        train_examples
    )

    hard_count = sum(
        row["source"] == "hard"
        for row in train_examples
    )
    regular_count = len(train_examples) - hard_count

    print("=" * 72)
    print("ДАННЫЕ ДЛЯ ДООБУЧЕНИЯ")
    print("=" * 72)
    print("Train needs:", len(train_need_ids))
    print("Dev needs:", len(dev_need_ids))
    print(
        "Обычных пар до объединения:",
        len(regular_examples),
    )
    print(
        "Hard-пар до объединения:",
        len(hard_examples),
    )
    print(
        "Итоговых уникальных пар:",
        len(train_examples),
    )
    print("Из них hard:", hard_count)
    print("Из них regular:", regular_count)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.checkpoint,
            dtype=torch.float32,
        )
        .to(device)
    )

    dataset = PairwiseDataset(
        examples=train_examples,
        needs=needs,
        providers=providers,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = (
        len(dataloader) * args.epochs
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps * args.warmup_ratio
        ),
        num_training_steps=total_steps,
    )

    use_amp = (
        args.use_fp16
        and device == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        losses = []
        accuracies = []

        progress = tqdm(
            dataloader,
            desc=f"Hard finetune epoch {epoch}",
        )

        for (
            pos_inputs,
            neg_inputs,
            weights,
        ) in progress:
            pos_inputs = {
                name: tensor.to(device)
                for name, tensor
                in pos_inputs.items()
            }

            neg_inputs = {
                name: tensor.to(device)
                for name, tensor
                in neg_inputs.items()
            }

            weights = weights.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                score_pos = model(
                    **pos_inputs
                ).logits.view(-1)

                score_neg = model(
                    **neg_inputs
                ).logits.view(-1)

                # Устойчивая версия RankNet loss:
                # штрафуем, когда score_pos
                # не больше score_neg.
                pair_losses = F.softplus(
                    -(score_pos - score_neg)
                )

                loss = (
                    pair_losses * weights
                ).mean()

            if use_amp:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            scheduler.step()

            accuracy = (
                score_pos > score_neg
            ).float().mean().item()

            losses.append(
                float(loss.detach().cpu())
            )
            accuracies.append(accuracy)

            progress.set_postfix(
                loss=(
                    f"{np.mean(losses[-50:]):.4f}"
                ),
                acc=(
                    f"{np.mean(accuracies[-50:]):.3f}"
                ),
            )

        print("\nОцениваем неизменённый dev...")

        dev_predictions = predict(
            model=model,
            tokenizer=tokenizer,
            need_ids=dev_need_ids,
            needs=needs,
            providers=providers,
            pools=pools,
            device=device,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
        )

        dev_pools = {
            need_id: pools[need_id]
            for need_id in dev_need_ids
        }

        ndcg, hcvr = ndcg_at_5(
            dev_predictions,
            labels,
            dev_pools,
        )

        prediction_path = (
            output_dir
            / f"dev_epoch_{epoch}.jsonl"
        )

        write_submission(
            prediction_path,
            dev_predictions,
        )

        checkpoint_dir = (
            output_dir
            / f"checkpoint_epoch_{epoch}"
        )

        model.save_pretrained(
            checkpoint_dir
        )
        tokenizer.save_pretrained(
            checkpoint_dir
        )

        print("\n" + "=" * 72)
        print("РЕЗУЛЬТАТ ДООБУЧЕНИЯ")
        print("=" * 72)
        print(
            f"Dev NDCG@5: {ndcg:.6f}"
        )
        print(
            f"Dev HCVR@5: {hcvr:.6f}"
        )
        print(
            "Δ NDCG к текущей лучшей:",
            f"{ndcg - BASELINE_NDCG:+.6f}",
        )
        print(
            "Δ HCVR к текущей лучшей:",
            f"{hcvr - BASELINE_HCVR:+.6f}",
        )
        print(
            "Checkpoint:",
            checkpoint_dir,
        )
        print(
            "Предсказания:",
            prediction_path,
        )


if __name__ == "__main__":
    main()
