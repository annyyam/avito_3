#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = ["bm25_score", "bge_score", "reranker_score"]


@dataclass
class NeedData:
    candidate_ids: np.ndarray
    features: np.ndarray
    grades: np.ndarray
    eligible: np.ndarray
    ideal_dcg: float


def dcg(grades: np.ndarray, k: int = 5) -> float:
    grades = grades[:k]
    return float(
        sum(
            (2 ** int(grade) - 1) / math.log2(index + 2)
            for index, grade in enumerate(grades)
        )
    )


def minmax_normalize_per_need(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        group_min = result.groupby("need_id")[column].transform("min")
        group_max = result.groupby("need_id")[column].transform("max")
        denominator = group_max - group_min

        result[f"{column}_norm"] = np.where(
            denominator > 0,
            (result[column] - group_min) / denominator,
            0.0,
        )

    return result


def build_need_data(frame: pd.DataFrame) -> dict[str, NeedData]:
    need_data: dict[str, NeedData] = {}
    feature_columns = [f"{column}_norm" for column in SCORE_COLUMNS]

    for need_id, group in frame.groupby("need_id", sort=False):
        candidate_ids = group["candidate_id"].astype(str).to_numpy()
        features = group[feature_columns].to_numpy(dtype=np.float64)
        grades = group["relevance_grade"].to_numpy(dtype=np.int64)
        eligible = group["hard_eligible"].to_numpy(dtype=bool)

        ideal_grades = np.sort(grades)[::-1][:5]
        ideal_dcg = dcg(ideal_grades)

        need_data[str(need_id)] = NeedData(
            candidate_ids=candidate_ids,
            features=features,
            grades=grades,
            eligible=eligible,
            ideal_dcg=ideal_dcg,
        )

    return need_data


def rank_top5(data: NeedData, weights: np.ndarray) -> np.ndarray:
    scores = data.features @ weights

    # Полностью повторяем evaluate.py:
    # score по убыванию, candidate_id по возрастанию при равенстве score.
    order = np.lexsort((data.candidate_ids, -scores))
    return order[:5]


def evaluate_weights(
    need_ids: list[str],
    need_data: dict[str, NeedData],
    weights: np.ndarray,
) -> tuple[float, float]:
    ndcg_values = []
    violations = 0
    total = 0

    for need_id in need_ids:
        data = need_data[need_id]
        top_indices = rank_top5(data, weights)

        ranked_grades = data.grades[top_indices]
        ranked_eligible = data.eligible[top_indices]

        current_dcg = dcg(ranked_grades)
        ndcg_values.append(
            current_dcg / data.ideal_dcg
            if data.ideal_dcg > 0
            else 0.0
        )

        violations += int((~ranked_eligible).sum())
        total += len(top_indices)

    return float(np.mean(ndcg_values)), violations / total


def make_weight_grid(step: float) -> list[np.ndarray]:
    parts = round(1.0 / step)

    if not math.isclose(parts * step, 1.0, abs_tol=1e-9):
        raise ValueError(
            "--step должен делить 1 без остатка, например 0.1 или 0.05"
        )

    grid = []

    for bm25_part in range(parts + 1):
        for bge_part in range(parts - bm25_part + 1):
            reranker_part = parts - bm25_part - bge_part

            grid.append(
                np.asarray(
                    [
                        bm25_part / parts,
                        bge_part / parts,
                        reranker_part / parts,
                    ],
                    dtype=np.float64,
                )
            )

    return grid


def choose_best_weights(
    train_need_ids: list[str],
    need_data: dict[str, NeedData],
    weight_grid: list[np.ndarray],
) -> tuple[np.ndarray, float, float, list[dict]]:
    rows = []

    for weights in weight_grid:
        ndcg_value, hcvr_value = evaluate_weights(
            train_need_ids,
            need_data,
            weights,
        )

        rows.append({
            "bm25_weight": float(weights[0]),
            "bge_weight": float(weights[1]),
            "reranker_weight": float(weights[2]),
            "ndcg_at_5": ndcg_value,
            "hcvr_at_5": hcvr_value,
        })

    # Главный критерий — NDCG@5. При полном равенстве выбираем меньший HCVR.
    rows.sort(
        key=lambda row: (
            -row["ndcg_at_5"],
            row["hcvr_at_5"],
            row["reranker_weight"],
            row["bm25_weight"],
        )
    )

    best = rows[0]
    best_weights = np.asarray(
        [
            best["bm25_weight"],
            best["bge_weight"],
            best["reranker_weight"],
        ],
        dtype=np.float64,
    )

    return (
        best_weights,
        float(best["ndcg_at_5"]),
        float(best["hcvr_at_5"]),
        rows,
    )


def submission_rows(
    need_ids: list[str],
    need_data: dict[str, NeedData],
    weights: np.ndarray,
) -> list[dict]:
    rows = []

    for need_id in need_ids:
        data = need_data[need_id]
        combined_scores = data.features @ weights
        top_indices = rank_top5(data, weights)

        rows.append({
            "need_id": need_id,
            "ranked_candidates": [
                {
                    "candidate_id": str(data.candidate_ids[index]),
                    "score": float(combined_scores[index]),
                }
                for index in top_indices
            ],
        })

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Nested 5-fold подбор весов ансамбля "
            "BM25 + BGE-M3 + BGE-reranker"
        )
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Шаг весов. 0.05 даёт 231 комбинацию.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Читаем оценки моделей...")
    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.labels)
    folds = pd.read_csv(args.folds)

    required_scores = {
        "need_id",
        "candidate_id",
        *SCORE_COLUMNS,
    }
    required_labels = {
        "need_id",
        "candidate_id",
        "relevance_grade",
        "hard_eligible",
    }
    required_folds = {"need_id", "fold"}

    if not required_scores.issubset(scores.columns):
        raise ValueError(
            f"В scores не хватает столбцов: "
            f"{sorted(required_scores - set(scores.columns))}"
        )
    if not required_labels.issubset(labels.columns):
        raise ValueError(
            f"В labels не хватает столбцов: "
            f"{sorted(required_labels - set(labels.columns))}"
        )
    if not required_folds.issubset(folds.columns):
        raise ValueError(
            f"В folds не хватает столбцов: "
            f"{sorted(required_folds - set(folds.columns))}"
        )

    labels = labels.copy()
    labels["hard_eligible"] = (
        labels["hard_eligible"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    merged = scores.merge(
        labels,
        on=["need_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )

    if merged["relevance_grade"].isna().any():
        missing = int(merged["relevance_grade"].isna().sum())
        raise ValueError(f"Не найдены метки для {missing} пар")

    fold_map = dict(
        zip(
            folds["need_id"].astype(str),
            folds["fold"].astype(int),
        )
    )

    score_need_ids = set(merged["need_id"].astype(str))
    fold_need_ids = set(fold_map)

    if score_need_ids != fold_need_ids:
        raise ValueError(
            "Наборы need_id в scores и folds не совпадают: "
            f"без fold={len(score_need_ids - fold_need_ids)}, "
            f"лишних fold={len(fold_need_ids - score_need_ids)}"
        )

    print("Нормализуем три score внутри каждого пула...")
    normalized = minmax_normalize_per_need(
        merged,
        SCORE_COLUMNS,
    )

    need_data = build_need_data(normalized)
    weight_grid = make_weight_grid(args.step)

    print(f"Потребностей: {len(need_data)}")
    print(f"Комбинаций весов: {len(weight_grid)}")

    unique_folds = sorted(set(fold_map.values()))
    fold_results = []
    oof_rows = []

    print("\nЗапускаем 5-fold cross-validation...")

    for held_fold in unique_folds:
        train_need_ids = [
            need_id
            for need_id, fold in fold_map.items()
            if fold != held_fold
        ]
        held_need_ids = [
            need_id
            for need_id, fold in fold_map.items()
            if fold == held_fold
        ]

        (
            best_weights,
            train_ndcg,
            train_hcvr,
            _,
        ) = choose_best_weights(
            train_need_ids,
            need_data,
            weight_grid,
        )

        held_ndcg, held_hcvr = evaluate_weights(
            held_need_ids,
            need_data,
            best_weights,
        )

        fold_result = {
            "held_fold": int(held_fold),
            "train_needs": len(train_need_ids),
            "held_needs": len(held_need_ids),
            "bm25_weight": float(best_weights[0]),
            "bge_weight": float(best_weights[1]),
            "reranker_weight": float(best_weights[2]),
            "train_ndcg_at_5": train_ndcg,
            "train_hcvr_at_5": train_hcvr,
            "held_ndcg_at_5": held_ndcg,
            "held_hcvr_at_5": held_hcvr,
        }
        fold_results.append(fold_result)

        oof_rows.extend(
            submission_rows(
                held_need_ids,
                need_data,
                best_weights,
            )
        )

        print(
            f"Fold {held_fold}: "
            f"weights=({best_weights[0]:.2f}, "
            f"{best_weights[1]:.2f}, "
            f"{best_weights[2]:.2f}) | "
            f"held NDCG={held_ndcg:.6f}, "
            f"HCVR={held_hcvr:.6f}"
        )

    # Точная общая OOF-метрика на всех 847 потребностях:
    # для каждой потребности использованы веса, выбранные без её фолда.
    oof_metrics_by_need = {}

    for row in oof_rows:
        need_id = row["need_id"]
        ranked_ids = [
            item["candidate_id"]
            for item in row["ranked_candidates"]
        ]

        data = need_data[need_id]
        candidate_to_index = {
            candidate_id: index
            for index, candidate_id in enumerate(data.candidate_ids)
        }
        top_indices = np.asarray(
            [candidate_to_index[candidate_id] for candidate_id in ranked_ids]
        )

        value = (
            dcg(data.grades[top_indices]) / data.ideal_dcg
            if data.ideal_dcg > 0
            else 0.0
        )
        violations = int((~data.eligible[top_indices]).sum())

        oof_metrics_by_need[need_id] = (value, violations)

    oof_ndcg = float(
        np.mean([value[0] for value in oof_metrics_by_need.values()])
    )
    oof_hcvr = (
        sum(value[1] for value in oof_metrics_by_need.values())
        / (5 * len(oof_metrics_by_need))
    )

    print("\nПодбираем финальные веса на всех internal_train...")
    (
        final_weights,
        full_train_ndcg,
        full_train_hcvr,
        full_grid_rows,
    ) = choose_best_weights(
        list(need_data),
        need_data,
        weight_grid,
    )

    print("\n===== ИТОГ CROSS-VALIDATION =====")
    print(f"OOF NDCG@5: {oof_ndcg:.6f}")
    print(f"OOF HCVR@5: {oof_hcvr:.6f}")

    print("\n===== ФИНАЛЬНЫЕ ВЕСА НА 847 NEEDS =====")
    print(f"BM25:     {final_weights[0]:.2f}")
    print(f"BGE-M3:   {final_weights[1]:.2f}")
    print(f"Reranker: {final_weights[2]:.2f}")
    print(f"Train NDCG@5: {full_train_ndcg:.6f}")
    print(f"Train HCVR@5: {full_train_hcvr:.6f}")

    pd.DataFrame(fold_results).to_csv(
        output_dir / "fold_results.csv",
        index=False,
    )

    pd.DataFrame(full_grid_rows).to_csv(
        output_dir / "full_train_grid.csv",
        index=False,
    )

    write_jsonl(
        output_dir / "oof_submission.jsonl",
        oof_rows,
    )

    final_submission = submission_rows(
        list(need_data),
        need_data,
        final_weights,
    )
    write_jsonl(
        output_dir / "final_internal_train_submission.jsonl",
        final_submission,
    )

    summary = {
        "step": args.step,
        "combinations": len(weight_grid),
        "oof_ndcg_at_5": oof_ndcg,
        "oof_hcvr_at_5": oof_hcvr,
        "final_weights": {
            "bm25": float(final_weights[0]),
            "bge": float(final_weights[1]),
            "reranker": float(final_weights[2]),
        },
        "full_train_ndcg_at_5": full_train_ndcg,
        "full_train_hcvr_at_5": full_train_hcvr,
        "fold_results": fold_results,
    }

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
