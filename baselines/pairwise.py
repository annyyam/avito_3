import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_pools(path):
    pools = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            pools[row["need_id"]].append(row["candidate_id"])
    return dict(pools)


def read_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            labels[(row["need_id"], row["candidate_id"])] = {
                "grade": int(row["relevance_grade"]),
                "hard_eligible": row["hard_eligible"].lower() == "true",
            }
    return labels


def read_split(path):
    split_by_need = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            split_by_need[row["need_id"]] = row["split"]
    return split_by_need


def dcg(grades, k=5):
    return sum((2 ** grade - 1) / math.log2(position + 2) for position, grade in enumerate(grades[:k]))


def ndcg_at_5(predictions, labels, pools):
    by_need_grades = defaultdict(list)
    for (need_id, candidate_id), label in labels.items():
        if need_id in pools:
            by_need_grades[need_id].append(label["grade"])

    scores = []
    violations = 0
    total = 0

    for need_id, ranked in predictions.items():
        ranked = sorted(ranked, key=lambda item: (-item["score"], item["candidate_id"]))[:5]
        grades = []

        for item in ranked:
            label = labels[(need_id, item["candidate_id"])]
            grades.append(label["grade"])
            violations += not label["hard_eligible"]
            total += 1

        ideal = sorted(by_need_grades[need_id], reverse=True)[:5]
        denom = dcg(ideal)
        scores.append(dcg(grades) / denom if denom else 0.0)

    return float(np.mean(scores)), violations / max(total, 1)


class PairwiseDataset(Dataset):
    def __init__(self, examples, needs, providers, tokenizer, max_length):
        self.examples = examples
        self.needs = needs
        self.providers = providers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = {}

        print("Pre-tokenizing unique pairs...")
        for ex in tqdm(self.examples):
            need_text = self.needs[ex["need_id"]]
            pos_text = self.providers[ex["pos_candidate_id"]]
            neg_text = self.providers[ex["neg_candidate_id"]]

            pair1 = (need_text, pos_text)
            pair2 = (need_text, neg_text)

            if pair1 not in self.cache:
                self.cache[pair1] = tokenizer(
                    pair1[0], pair1[1],
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )

            if pair2 not in self.cache:
                self.cache[pair2] = tokenizer(
                    pair2[0], pair2[1],
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )

        print(f"Cached {len(self.cache)} unique pairs.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        need_text = self.needs[ex["need_id"]]
        pos_text = self.providers[ex["pos_candidate_id"]]
        neg_text = self.providers[ex["neg_candidate_id"]]

        return {
            "pos": self.cache[(need_text, pos_text)],
            "neg": self.cache[(need_text, neg_text)],
            "weight": ex["weight"],
        }

    def collate_fn(self, batch):
        pos = self.tokenizer.pad([x["pos"] for x in batch], return_tensors="pt")
        neg = self.tokenizer.pad([x["neg"] for x in batch], return_tensors="pt")
        weights = torch.tensor([x["weight"] for x in batch], dtype=torch.float32)
        return pos, neg, weights


def make_pairwise_examples(
        need_ids,
        needs,
        providers,
        pools,
        labels,
        hard_penalty,
        max_pairs_per_need,
        seed,
):

    rng = random.Random(seed)
    examples = []

    for need_id in need_ids:
        candidates = []

        for candidate_id in pools.get(need_id, []):
            label = labels.get((need_id, candidate_id))
            if label is None:
                continue

            grade = float(label["grade"])
            if not label["hard_eligible"]:
                grade *= hard_penalty

            candidates.append({
                "candidate_id": candidate_id,
                "grade": grade,
                "original_grade": label["grade"],
            })

        # Сортируем по убыванию
        candidates.sort(key=lambda x: x["grade"], reverse=True)

        need_pairs = []

        # Берем только пары с разницей > 0
        for i in range(len(candidates)):
            positive = candidates[i]
            if positive["grade"] <= 0:
                continue

            added = 0
            for j in range(i + 1, len(candidates)):
                negative = candidates[j]

                if positive["grade"] <= negative["grade"]:
                    continue

                need_pairs.append({
                    "need_id": need_id,
                    "pos_candidate_id": positive["candidate_id"],
                    "neg_candidate_id": negative["candidate_id"],
                    "weight": positive["grade"] - negative["grade"],
                })

                added += 1
                if max_pairs_per_need > 0 and added >= max_pairs_per_need:
                    break

        examples.extend(need_pairs)

    rng.shuffle(examples)
    return examples


def score_need(model, tokenizer, query, candidate_ids, providers, device, batch_size, max_length):
    scores = []

    for start in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[start:start + batch_size]
        inputs = tokenizer(
            [query] * len(batch_ids),
            [providers[candidate_id] for candidate_id in batch_ids],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits.view(-1)

        scores.extend(logits.float().cpu().numpy().tolist())

    return scores


def predict(model, tokenizer, need_ids, needs, providers, pools, device, batch_size, max_length):
    model.eval()
    predictions = {}

    for need_id in tqdm(need_ids, desc="Predict"):
        candidate_ids = pools[need_id]
        scores = score_need(
            model=model,
            tokenizer=tokenizer,
            query=needs[need_id],
            candidate_ids=candidate_ids,
            providers=providers,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        predictions[need_id] = [
            {"candidate_id": candidate_id, "score": float(score)}
            for candidate_id, score in zip(candidate_ids, scores)
        ]

    return predictions


def write_submission(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for need_id, rows in predictions.items():
            top = sorted(rows, key=lambda item: (-item["score"], item["candidate_id"]))[:5]
            file.write(json.dumps({
                "need_id": need_id,
                "ranked_candidates": top,
            }, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BGE-M3 with pairwise ranking loss")
    parser.add_argument("--needs-train", default="avito_airi_3_case/student_package/data/needs.train.jsonl")
    parser.add_argument("--needs-validation", default="avito_airi_3_case/student_package/data/needs.validation.jsonl")
    parser.add_argument("--providers", default="avito_airi_3_case/student_package/data/providers.jsonl")
    parser.add_argument("--pools-train", default="avito_airi_3_case/student_package/data/candidate_pools.train.csv")
    parser.add_argument("--pools-validation",
                        default="avito_airi_3_case/student_package/data/candidate_pools.validation.csv")
    parser.add_argument("--labels", default="avito_airi_3_case/student_package/data/labels.train.csv")
    parser.add_argument("--split", default="splits/need_split.csv")
    parser.add_argument("--model", default="BAAI/bge-m3")  # Меняем на BGE-M3!
    parser.add_argument("--output-dir", default="outputs/bge_m3_pairwise")
    parser.add_argument("--validation-output", default="outputs/bge_m3_pairwise/dev_submission.jsonl")
    parser.add_argument("--final-output", default="outputs/bge_m3_pairwise/submission.validation.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hard-penalty", type=float, default=0.0)
    parser.add_argument("--max-pairs-per-need", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-validation", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-weighted-loss", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    need_rows = read_jsonl(args.needs_train)
    provider_rows = read_jsonl(args.providers)
    pools = read_pools(args.pools_train)
    labels = read_labels(args.labels)
    split_by_need = read_split(args.split)

    needs = {row["need_id"]: row["need_text"] for row in need_rows}
    providers = {row["candidate_id"]: row["profile_summary"] for row in provider_rows}

    train_need_ids = [need_id for need_id in needs if split_by_need.get(need_id) == "train" and need_id in pools]
    dev_need_ids = [need_id for need_id in needs if split_by_need.get(need_id) == "dev" and need_id in pools]

    train_examples = make_pairwise_examples(
        need_ids=train_need_ids,
        needs=needs,
        providers=providers,
        pools=pools,
        labels=labels,
        hard_penalty=args.hard_penalty,
        max_pairs_per_need=args.max_pairs_per_need,
        seed=args.seed,
    )

    print(f"Train needs: {len(train_need_ids)}")
    print(f"Dev needs: {len(dev_need_ids)}")
    print(f"Train pairs: {len(train_examples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        dtype=torch.float32,
    ).to(device)

    use_amp = args.use_fp16 and device == "cuda"

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
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    def ranknet_loss_weighted(score_pos, score_neg, weights):
        prob = torch.sigmoid(score_pos - score_neg)
        loss = -weights * torch.log(prob + 1e-8)
        return loss.mean()

    def ranknet_loss_unweighted(score_pos, score_neg):
        prob = torch.sigmoid(score_pos - score_neg)
        return -torch.log(prob + 1e-8).mean()

    if args.use_weighted_loss:
        loss_fn = ranknet_loss_weighted
        print("Using weighted RankNet loss")
    else:
        loss_fn = ranknet_loss_unweighted
        print("Using unweighted RankNet loss")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_ndcg = -1.0
    best_dir = output_dir / "best"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        accuracies = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch}")
        for pos_inputs, neg_inputs, weights in progress:
            pos_inputs = {name: tensor.to(device) for name, tensor in pos_inputs.items()}
            neg_inputs = {name: tensor.to(device) for name, tensor in neg_inputs.items()}
            weights = weights.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    score_pos = model(**pos_inputs).logits.view(-1)
                    score_neg = model(**neg_inputs).logits.view(-1)
                    loss = loss_fn(score_pos, score_neg, weights) if args.use_weighted_loss else loss_fn(score_pos,
                                                                                                         score_neg)
            else:
                score_pos = model(**pos_inputs).logits.view(-1)
                score_neg = model(**neg_inputs).logits.view(-1)
                loss = loss_fn(score_pos, score_neg, weights) if args.use_weighted_loss else loss_fn(score_pos,
                                                                                                     score_neg)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            losses.append(float(loss.detach().cpu()))
            acc = (score_pos > score_neg).float().mean().item()
            accuracies.append(acc)

            progress.set_postfix(
                loss=f"{np.mean(losses[-50:]):.4f}",
                acc=f"{np.mean(accuracies[-50:]):.3f}"
            )

        # Валидация после эпохи
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
        write_submission(args.validation_output, dev_predictions)
        ndcg, violation_rate = ndcg_at_5(dev_predictions, labels, {need_id: pools[need_id] for need_id in dev_need_ids})

        print(
            f"Epoch {epoch}: loss={np.mean(losses):.5f} dev_NDCG@5={ndcg:.6f} dev_HardViolation@5={violation_rate:.6f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved best model to {best_dir}")

    if args.predict_validation:
        print("Loading best model and scoring organizer validation pools...")
        tokenizer = AutoTokenizer.from_pretrained(best_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            best_dir,
            dtype=torch.float32,
        ).to(device)

        validation_rows = read_jsonl(args.needs_validation)
        validation_needs = {row["need_id"]: row["need_text"] for row in validation_rows}
        validation_pools = read_pools(args.pools_validation)
        validation_need_ids = [row["need_id"] for row in validation_rows if row["need_id"] in validation_pools]

        validation_predictions = predict(
            model=model,
            tokenizer=tokenizer,
            need_ids=validation_need_ids,
            needs=validation_needs,
            providers=providers,
            pools=validation_pools,
            device=device,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
        )
        write_submission(args.final_output, validation_predictions)
        print(f"Final validation submission: {args.final_output}")


if __name__ == "__main__":
    main()