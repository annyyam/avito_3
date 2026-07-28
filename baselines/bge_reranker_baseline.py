import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: str) -> list[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Ошибка JSON в файле {path}, строка {line_number}"
                ) from error

    return rows


def read_pools(path: str) -> dict[str, list[str]]:
    pools = defaultdict(list)

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required = {"need_id", "candidate_id"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"В {path} должны быть столбцы {sorted(required)}"
            )

        for row in reader:
            pools[row["need_id"]].append(row["candidate_id"])

    return dict(pools)


def score_pairs(
    model,
    tokenizer,
    query: str,
    documents: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    all_scores = []

    for start in range(0, len(documents), batch_size):
        batch_documents = documents[start:start + batch_size]
        batch_queries = [query] * len(batch_documents)

        inputs = tokenizer(
            batch_queries,
            batch_documents,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {
            name: tensor.to(device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits.view(-1)

        all_scores.extend(
            logits.float().cpu().numpy().tolist()
        )

    return np.asarray(all_scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ранжирование полных пулов через BGE reranker"
    )

    parser.add_argument("--needs", required=True)
    parser.add_argument("--providers", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--model",
        default="BAAI/bge-reranker-v2-m3",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    print("Читаем данные...")

    need_rows = read_jsonl(args.needs)
    provider_rows = read_jsonl(args.providers)
    pools = read_pools(args.pools)

    needs = {
        row["need_id"]: row["need_text"]
        for row in need_rows
    }

    providers = {
        row["candidate_id"]: row["profile_summary"]
        for row in provider_rows
    }

    need_ids = [
        row["need_id"]
        for row in need_rows
        if row["need_id"] in pools
    ]

    missing_candidates = sorted({
        candidate_id
        for need_id in need_ids
        for candidate_id in pools[need_id]
        if candidate_id not in providers
    })

    if missing_candidates:
        raise ValueError(
            f"Нет текстов для {len(missing_candidates)} кандидатов"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Потребностей: {len(need_ids)}")
    print(f"Устройство: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Загружаем модель: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Ранжируем полные пулы кандидатов...")

    with output_path.open("w", encoding="utf-8") as output_file:
        for need_id in tqdm(need_ids, desc="Needs"):
            pool_ids = pools[need_id]
            pool_texts = [
                providers[candidate_id]
                for candidate_id in pool_ids
            ]

            scores = score_pairs(
                model=model,
                tokenizer=tokenizer,
                query=needs[need_id],
                documents=pool_texts,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            top_positions = np.argsort(
                -scores,
                kind="stable",
            )[:5]

            result = {
                "need_id": need_id,
                "ranked_candidates": [
                    {
                        "candidate_id": pool_ids[position],
                        "score": float(scores[position]),
                    }
                    for position in top_positions
                ],
            }

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print(f"Готово: {output_path}")
    print(f"Строк записано: {len(need_ids)}")


if __name__ == "__main__":
    main()
