import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from nltk.stem.snowball import RussianStemmer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
STEMMER = RussianStemmer()


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
                    f"Некорректный JSON: {path}, строка {line_number}"
                ) from error
    return rows


def read_pools(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"need_id", "candidate_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"В {path} должны быть столбцы need_id и candidate_id"
            )
        return list(reader)


def tokenize_and_stem(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(str(text).lower())
    return [STEMMER.stem(token) for token in tokens]


class BM25:
    def __init__(
        self,
        documents: dict[str, str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = {
            doc_id: tokenize_and_stem(text)
            for doc_id, text in documents.items()
        }
        self.doc_lengths = {
            doc_id: len(tokens)
            for doc_id, tokens in self.doc_tokens.items()
        }
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)
        )
        self.term_frequencies = {
            doc_id: Counter(tokens)
            for doc_id, tokens in self.doc_tokens.items()
        }

        document_frequency = Counter()
        for tokens in self.doc_tokens.values():
            document_frequency.update(set(tokens))

        document_count = len(self.doc_tokens)
        self.idf = {
            term: math.log(
                1.0
                + (document_count - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str, doc_id: str) -> float:
        query_terms = tokenize_and_stem(query)
        frequencies = self.term_frequencies[doc_id]
        document_length = self.doc_lengths[doc_id]

        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue

            denominator = frequency + self.k1 * (
                1.0
                - self.b
                + self.b * document_length / max(self.avg_doc_length, 1e-12)
            )
            score += self.idf.get(term, 0.0) * (
                frequency * (self.k1 + 1.0)
            ) / denominator

        return float(score)


def score_reranker_pairs(
    model,
    tokenizer,
    queries: list[str],
    documents: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    scores = []

    for start in tqdm(
        range(0, len(queries), batch_size),
        desc="Reranker batches",
    ):
        batch_queries = queries[start:start + batch_size]
        batch_documents = documents[start:start + batch_size]

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

        scores.extend(logits.float().cpu().numpy().tolist())

    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Сохраняет BM25, BGE-M3 и BGE-reranker оценки "
            "для каждой пары потребность-кандидат"
        )
    )
    parser.add_argument("--needs", required=True)
    parser.add_argument("--providers", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bge-model", default="BAAI/bge-m3")
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--bge-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)

    args = parser.parse_args()

    need_rows = read_jsonl(args.needs)
    provider_rows = read_jsonl(args.providers)
    pool_rows = read_pools(args.pools)

    needs = {
        row["need_id"]: row["need_text"]
        for row in need_rows
    }
    providers = {
        row["candidate_id"]: row["profile_summary"]
        for row in provider_rows
    }

    missing_needs = sorted({
        row["need_id"]
        for row in pool_rows
        if row["need_id"] not in needs
    })
    missing_candidates = sorted({
        row["candidate_id"]
        for row in pool_rows
        if row["candidate_id"] not in providers
    })

    if missing_needs:
        raise ValueError(
            f"Не найдены тексты для {len(missing_needs)} потребностей"
        )
    if missing_candidates:
        raise ValueError(
            f"Не найдены анкеты для {len(missing_candidates)} кандидатов"
        )

    unique_need_ids = list(dict.fromkeys(
        row["need_id"] for row in pool_rows
    ))
    unique_candidate_ids = list(dict.fromkeys(
        row["candidate_id"] for row in pool_rows
    ))

    print(f"Потребностей: {len(unique_need_ids)}")
    print(f"Уникальных кандидатов: {len(unique_candidate_ids)}")
    print(f"Пар потребность-кандидат: {len(pool_rows)}")

    print("\nСчитаем BM25...")
    bm25 = BM25(
        {
            candidate_id: providers[candidate_id]
            for candidate_id in unique_candidate_ids
        },
        k1=args.k1,
        b=args.b,
    )
    bm25_scores = np.asarray([
        bm25.score(
            needs[row["need_id"]],
            row["candidate_id"],
        )
        for row in tqdm(pool_rows, desc="BM25 pairs")
    ], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nУстройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nЗагружаем BGE-M3: {args.bge_model}")
    bge_model = SentenceTransformer(
        args.bge_model,
        device=device,
    )
    bge_model.max_seq_length = args.max_seq_length

    need_texts = [needs[need_id] for need_id in unique_need_ids]
    candidate_texts = [
        providers[candidate_id]
        for candidate_id in unique_candidate_ids
    ]

    print("Строим embeddings потребностей...")
    need_embeddings = bge_model.encode(
        need_texts,
        batch_size=args.bge_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Строим embeddings кандидатов...")
    candidate_embeddings = bge_model.encode(
        candidate_texts,
        batch_size=args.bge_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    need_index = {
        need_id: index
        for index, need_id in enumerate(unique_need_ids)
    }
    candidate_index = {
        candidate_id: index
        for index, candidate_id in enumerate(unique_candidate_ids)
    }

    bge_scores = np.asarray([
        float(
            np.dot(
                need_embeddings[need_index[row["need_id"]]],
                candidate_embeddings[
                    candidate_index[row["candidate_id"]]
                ],
            )
        )
        for row in pool_rows
    ], dtype=np.float32)

    del bge_model
    del need_embeddings
    del candidate_embeddings
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nЗагружаем реранкер: {args.reranker_model}")
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        args.reranker_model
    )
    reranker_model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.reranker_model,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        .to(device)
    )
    reranker_model.eval()

    pair_queries = [needs[row["need_id"]] for row in pool_rows]
    pair_documents = [
        providers[row["candidate_id"]] for row in pool_rows
    ]

    reranker_scores = score_reranker_pairs(
        model=reranker_model,
        tokenizer=reranker_tokenizer,
        queries=pair_queries,
        documents=pair_documents,
        device=device,
        batch_size=args.reranker_batch_size,
        max_length=args.max_seq_length,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nСохраняем: {output_path}")
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "need_id",
            "candidate_id",
            "bm25_score",
            "bge_score",
            "reranker_score",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(pool_rows):
            writer.writerow({
                "need_id": row["need_id"],
                "candidate_id": row["candidate_id"],
                "bm25_score": float(bm25_scores[index]),
                "bge_score": float(bge_scores[index]),
                "reranker_score": float(reranker_scores[index]),
            })

    print(f"Готово. Строк записано: {len(pool_rows)}")


if __name__ == "__main__":
    main()
