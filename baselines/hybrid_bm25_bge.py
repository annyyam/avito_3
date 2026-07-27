import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from nltk.stem.snowball import RussianStemmer
from sentence_transformers import SentenceTransformer

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
STEMMER = RussianStemmer()


def tokenize(text):
    words = [w.lower() for w in TOKEN_RE.findall(text or "") if len(w) > 1]
    return [STEMMER.stem(w) if re.fullmatch(r"[а-яё]+", w) else w for w in words]


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


def normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def alpha_name(alpha):
    return f"{alpha:.1f}".replace(".", "_")


def main():
    parser = argparse.ArgumentParser(description="Hybrid BGE-M3 + BM25 stemming")
    parser.add_argument("--needs", required=True)
    parser.add_argument("--providers", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    args = parser.parse_args()

    if any(not 0.0 <= a <= 1.0 for a in args.alphas):
        raise ValueError("alpha must be between 0 and 1")

    need_rows = read_jsonl(args.needs)
    provider_rows = read_jsonl(args.providers)
    pools = read_pools(args.pools)

    needs = {row["need_id"]: row["need_text"] for row in need_rows}
    providers = {row["candidate_id"]: row["profile_summary"] for row in provider_rows}
    need_ids = [row["need_id"] for row in need_rows if row["need_id"] in pools]
    candidate_ids = list(dict.fromkeys(
        cid for nid in need_ids for cid in pools[nid]
    ))

    missing = [cid for cid in candidate_ids if cid not in providers]
    if missing:
        raise ValueError(f"Missing provider texts: {len(missing)}")

    print(f"Потребностей: {len(need_ids)}")
    print(f"Уникальных кандидатов: {len(candidate_ids)}")
    print("Готовим BM25...")

    candidate_tokens = {cid: tokenize(providers[cid]) for cid in candidate_ids}
    term_frequencies = {cid: Counter(tokens) for cid, tokens in candidate_tokens.items()}
    document_frequency = Counter()
    for tokens in candidate_tokens.values():
        document_frequency.update(set(tokens))

    n_docs = len(candidate_ids)
    avg_len = sum(len(tokens) for tokens in candidate_tokens.values()) / max(n_docs, 1)

    def bm25(query_tokens, candidate_id):
        tf_counter = term_frequencies[candidate_id]
        doc_len = len(candidate_tokens[candidate_id])
        score = 0.0
        for token in query_tokens:
            tf = tf_counter.get(token, 0)
            if tf == 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + args.k1 * (1.0 - args.b + args.b * doc_len / max(avg_len, 1e-12))
            score += idf * tf * (args.k1 + 1.0) / denom
        return score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство BGE-M3: {device}")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        a: (output_dir / f"submission.hybrid.alpha_{alpha_name(a)}.jsonl").open("w", encoding="utf-8")
        for a in args.alphas
    }

    print("Считаем гибридные рейтинги...")
    try:
        for need_pos, need_id in enumerate(need_ids):
            pool_ids = pools[need_id]
            indices = np.array([candidate_index[cid] for cid in pool_ids], dtype=np.int64)
            semantic_scores = candidate_embeddings[indices] @ need_embeddings[need_pos]
            query_tokens = tokenize(needs[need_id])
            bm25_scores = np.array([bm25(query_tokens, cid) for cid in pool_ids], dtype=np.float32)

            semantic_norm = normalize(semantic_scores)
            bm25_norm = normalize(bm25_scores)

            for alpha, out in files.items():
                hybrid = alpha * semantic_norm + (1.0 - alpha) * bm25_norm
                top = np.argsort(-hybrid, kind="stable")[:5]
                result = {
                    "need_id": need_id,
                    "ranked_candidates": [
                        {"candidate_id": pool_ids[pos], "score": float(hybrid[pos])}
                        for pos in top
                    ],
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        for out in files.values():
            out.close()

    print("Готово. Созданы варианты alpha от 0.0 до 1.0")


if __name__ == "__main__":
    main()
