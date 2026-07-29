import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from baselines.pairwise import (
    ndcg_at_5,
    predict,
    read_jsonl,
    read_labels,
    read_pools,
    read_split,
    write_submission,
)


CURRENT_BEST_NDCG = 0.918173
CURRENT_BEST_HCVR = 0.003774


def effective_grade(label):
    """
    Формально недопустимого кандидата считаем нулевым
    для обучения, как в предыдущей pairwise-модели.
    """
    if not label["hard_eligible"]:
        return 0.0

    return float(label["grade"])


def make_lambda_loss(
    scores,
    grades,
    candidate_ids,
    k=5,
    ranknet_mix=0.15,
):
    """
    LambdaRank-подобный pairwise loss.

    Каждая пара получает вес, равный изменению NDCG@k,
    которое возникло бы при перестановке кандидатов.
    """
    count = scores.shape[0]

    ideal_grades = torch.sort(
        grades,
        descending=True,
    ).values

    ideal_gains = torch.pow(
        2.0,
        ideal_grades,
    ) - 1.0

    ideal_positions = torch.arange(
        count,
        device=scores.device,
        dtype=torch.float32,
    )

    ideal_discounts = torch.where(
        ideal_positions < k,
        1.0 / torch.log2(ideal_positions + 2.0),
        torch.zeros_like(ideal_positions),
    )

    idcg = (
        ideal_gains * ideal_discounts
    ).sum()

    if float(idcg.detach().cpu()) <= 1e-12:
        return None, {}

    # Повторяем tie-break правила метрики:
    # сначала score по убыванию, затем candidate_id.
    detached_scores = (
        scores.detach().float().cpu().tolist()
    )

    predicted_order = sorted(
        range(count),
        key=lambda index: (
            -detached_scores[index],
            candidate_ids[index],
        ),
    )

    ranks = torch.empty(
        count,
        dtype=torch.long,
        device=scores.device,
    )

    for rank, index in enumerate(predicted_order):
        ranks[index] = rank

    rank_positions = ranks.float()

    discounts = torch.where(
        rank_positions < k,
        1.0 / torch.log2(rank_positions + 2.0),
        torch.zeros_like(rank_positions),
    )

    gains = torch.pow(2.0, grades) - 1.0

    # [i, j]: кандидат i должен быть выше кандидата j.
    grade_difference = (
        grades[:, None] - grades[None, :]
    )

    valid_pairs = grade_difference > 0

    score_difference = (
        scores[:, None] - scores[None, :]
    )

    pair_losses = F.softplus(
        -score_difference
    )

    gain_difference = (
        gains[:, None] - gains[None, :]
    )

    discount_difference = (
        discounts[:, None] - discounts[None, :]
    )

    delta_ndcg = (
        gain_difference * discount_difference
    ).abs() / idcg

    lambda_weights = (
        delta_ndcg * valid_pairs.float()
    )

    lambda_weight_sum = lambda_weights.sum()

    if float(
        lambda_weight_sum.detach().cpu()
    ) > 1e-12:
        lambda_loss = (
            pair_losses * lambda_weights
        ).sum() / lambda_weight_sum
    else:
        lambda_loss = scores.sum() * 0.0

    # Небольшая примесь обычного RankNet помогает
    # не забыть порядок кандидатов ниже top-5.
    ranknet_weights = (
        grade_difference.clamp(min=0.0)
        * valid_pairs.float()
    )

    ranknet_weight_sum = ranknet_weights.sum()

    if float(
        ranknet_weight_sum.detach().cpu()
    ) > 1e-12:
        ranknet_loss = (
            pair_losses * ranknet_weights
        ).sum() / ranknet_weight_sum
    else:
        ranknet_loss = scores.sum() * 0.0

    loss = (
        (1.0 - ranknet_mix) * lambda_loss
        + ranknet_mix * ranknet_loss
    )

    predicted_grade_order = grades[
        torch.tensor(
            predicted_order,
            device=grades.device,
        )
    ]

    top_grades = predicted_grade_order[:k]

    current_dcg = (
        (torch.pow(2.0, top_grades) - 1.0)
        / torch.log2(
            torch.arange(
                len(top_grades),
                device=grades.device,
                dtype=torch.float32,
            )
            + 2.0
        )
    ).sum()

    statistics = {
        "lambda_loss": float(
            lambda_loss.detach().cpu()
        ),
        "ranknet_loss": float(
            ranknet_loss.detach().cpu()
        ),
        "train_ndcg": float(
            (current_dcg / idcg).detach().cpu()
        ),
        "active_pairs": int(
            (lambda_weights > 0).sum()
            .detach().cpu()
        ),
    }

    return loss, statistics


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
            "bge_m3_hard_finetune_v1/"
            "checkpoint_epoch_1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/User25/outputs/"
            "bge_m3_lambda_ndcg_v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
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
        default=2e-6,
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
        "--ranknet-mix",
        type=float,
        default=0.15,
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print("=" * 72)
    print("LAMBDA NDCG@5 FINE-TUNING")
    print("=" * 72)
    print("Train needs:", len(train_need_ids))
    print("Dev needs:", len(dev_need_ids))
    print("Checkpoint:", args.checkpoint)
    print("Learning rate:", args.lr)
    print("RankNet mix:", args.ranknet_mix)

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

    use_amp = (
        args.use_fp16
        and device == "cuda"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_need_ids)
        / args.gradient_accumulation
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * args.epochs
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_optimizer_steps
            * args.warmup_ratio
        ),
        num_training_steps=total_optimizer_steps,
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

        shuffled_need_ids = list(
            train_need_ids
        )

        random.Random(
            args.seed + epoch
        ).shuffle(shuffled_need_ids)

        optimizer.zero_grad(
            set_to_none=True
        )

        pending_gradients = 0
        losses = []
        lambda_losses = []
        ranknet_losses = []
        train_ndcgs = []
        active_pairs = []

        progress = tqdm(
            shuffled_need_ids,
            desc=f"Lambda epoch {epoch}",
        )

        for need_id in progress:
            candidate_ids = pools[need_id]

            query = needs[need_id]

            score_chunks = []

            for start in range(
                0,
                len(candidate_ids),
                args.micro_batch_size,
            ):
                batch_ids = candidate_ids[
                    start:
                    start + args.micro_batch_size
                ]

                inputs = tokenizer(
                    [query] * len(batch_ids),
                    [
                        providers[candidate_id]
                        for candidate_id in batch_ids
                    ],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )

                inputs = {
                    name: tensor.to(device)
                    for name, tensor
                    in inputs.items()
                }

                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    chunk_scores = model(
                        **inputs
                    ).logits.view(-1)

                score_chunks.append(
                    chunk_scores.float()
                )

            scores = torch.cat(
                score_chunks,
                dim=0,
            )

            grades = torch.tensor(
                [
                    effective_grade(
                        labels[
                            (need_id, candidate_id)
                        ]
                    )
                    for candidate_id in candidate_ids
                ],
                dtype=torch.float32,
                device=device,
            )

            loss, statistics = make_lambda_loss(
                scores=scores,
                grades=grades,
                candidate_ids=candidate_ids,
                k=5,
                ranknet_mix=args.ranknet_mix,
            )

            if loss is None:
                continue

            scaled_loss = (
                loss
                / args.gradient_accumulation
            )

            if use_amp:
                scaler.scale(
                    scaled_loss
                ).backward()
            else:
                scaled_loss.backward()

            pending_gradients += 1

            losses.append(
                float(loss.detach().cpu())
            )
            lambda_losses.append(
                statistics["lambda_loss"]
            )
            ranknet_losses.append(
                statistics["ranknet_loss"]
            )
            train_ndcgs.append(
                statistics["train_ndcg"]
            )
            active_pairs.append(
                statistics["active_pairs"]
            )

            if (
                pending_gradients
                >= args.gradient_accumulation
            ):
                if use_amp:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                pending_gradients = 0

            progress.set_postfix(
                loss=(
                    f"{np.mean(losses[-30:]):.4f}"
                ),
                ndcg=(
                    f"{np.mean(train_ndcgs[-30:]):.3f}"
                ),
                pairs=(
                    f"{np.mean(active_pairs[-30:]):.1f}"
                ),
            )

        if pending_gradients > 0:
            if use_amp:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        print("\nСредние train-показатели:")
        print(
            "Loss:",
            f"{np.mean(losses):.6f}",
        )
        print(
            "Lambda loss:",
            f"{np.mean(lambda_losses):.6f}",
        )
        print(
            "RankNet loss:",
            f"{np.mean(ranknet_losses):.6f}",
        )
        print(
            "Train pool NDCG@5:",
            f"{np.mean(train_ndcgs):.6f}",
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
        print("РЕЗУЛЬТАТ LAMBDA NDCG@5")
        print("=" * 72)
        print(
            "Dev NDCG@5:",
            f"{ndcg:.6f}",
        )
        print(
            "Dev HCVR@5:",
            f"{hcvr:.6f}",
        )
        print(
            "Δ NDCG к hard-finetune:",
            f"{ndcg - CURRENT_BEST_NDCG:+.6f}",
        )
        print(
            "Δ HCVR к hard-finetune:",
            f"{hcvr - CURRENT_BEST_HCVR:+.6f}",
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
