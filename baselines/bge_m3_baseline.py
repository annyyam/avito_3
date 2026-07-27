import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_pools(path):
    pools = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pools[row["need_id"]].append(row["candidate_id"])
    return dict(pools)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--needs", required=True)
    parser.add_argument("--providers", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    need_rows = read_jsonl(args.needs)
    provider_rows = read_jsonl(args.providers)
    pools = read_pools(args.pools)

    needs = {row["need_id"]: row["need_text"] for row in need_rows}
    providers = {
        row["candidate_id"]: row["profile_summary"]
        for row in provider_rows
    }

    need_ids = [row["need_id"] for row in need_rows if row["need_id"] in pools]
    candidate_ids = list(dict.fromkeys(
        candidate_id
        for need_id in need_ids
        for candidate_id in pools[need_id]
    ))

    missing = [cid for cid in candidate_ids if cid not in providers]
    if missing:
        raise ValueError(f"Нет текстов для {len(missing)} кандидатов")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Потребностей: {len(need_ids)}")
    print(f"Уникальных кандидатов: {len(candidate_ids)}")
    print(f"Устройство: {device}")

    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length

    if device == "cuda":
        model.half()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Строим embeddings потребностей...")
    need_embeddings = model.encode(
        [needs[nid] for nid in need_ids],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("Строим embeddings кандидатов...")
    candidate_embeddings = model.encode(
        [providers[cid] for cid in candidate_ids],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    candidate_index = {cid: i for i, cid in enumerate(candidate_ids)}
    output_path = Path(args.output)

    print("Ранжируем...")
    with output_path.open("w", encoding="utf-8") as out:
        for need_pos, need_id in enumerate(need_ids):
            pool_ids = pools[need_id]
            indices = np.array([candidate_index[cid] for cid in pool_ids])
            scores = candidate_embeddings[indices] @ need_embeddings[need_pos]
            top_positions = np.argsort(-scores, kind="stable")[:5]

            result = {
                "need_id": need_id,
                "ranked_candidates": [
                    {
                        "candidate_id": pool_ids[pos],
                        "score": float(scores[pos]),
                    }
                    for pos in top_positions
                ],
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Готово: {output_path}")
    print(f"Строк записано: {len(need_ids)}")


if __name__ == "__main__":
    main()
