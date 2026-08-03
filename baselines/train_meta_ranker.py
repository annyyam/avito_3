from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
try:
    from catboost import CatBoostRanker, Pool
except ModuleNotFoundError:
    CatBoostRanker = None
    Pool = None


DATA_DIR = Path("avito_airi_3_case/student_package/data")
BASE_FEATURE_COLUMNS = [
    "bge_score_raw",
    "bge_score_norm",
    "bge_score_rank_pct",
    "history_mean",
    "history_smoothed",
    "history_count",
    "history_log_count",
    "history_seen",
]


def feature_columns(bge_features: str) -> list[str]:
    history_columns = BASE_FEATURE_COLUMNS[3:]
    if bge_features == "all":
        return BASE_FEATURE_COLUMNS
    if bge_features == "normalized_and_rank":
        return ["bge_score_norm", "bge_score_rank_pct", *history_columns]
    if bge_features == "rank_only":
        return ["bge_score_rank_pct", *history_columns]
    raise ValueError(f"Unknown BGE feature set: {bge_features}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_labels(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    labels = {}
    for row in read_csv_rows(path):
        labels[row["need_id"], row["candidate_id"]] = {
            "grade": int(row["relevance_grade"]),
            "hard_eligible": row["hard_eligible"].lower() == "true",
        }
    return labels


def load_score_files(paths: list[Path], score_column: str) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for path in paths:
        for row in read_csv_rows(path):
            required = {"need_id", "candidate_id", score_column}
            if not required.issubset(row):
                raise ValueError(f"{path} must contain {sorted(required)}")
            key = row["need_id"], row["candidate_id"]
            if key in scores:
                raise ValueError(f"Duplicate score for {key} across score files")
            scores[key] = float(row[score_column])
    return scores


def load_mapping(path: Path, column: str) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows or "need_id" not in rows[0] or column not in rows[0]:
        raise ValueError(f"{path} must contain need_id,{column}")
    return {row["need_id"]: row[column] for row in rows}


def minmax_and_rank(rows: list[dict]) -> None:
    values = [row["bge_score_raw"] for row in rows]
    low, high = min(values), max(values)
    for row in rows:
        row["bge_score_norm"] = (row["bge_score_raw"] - low) / (high - low) if high > low else 0.0
    for rank, row in enumerate(sorted(rows, key=lambda item: (-item["bge_score_raw"], item["candidate_id"]))):
        row["bge_score_rank_pct"] = 1.0 - rank / max(len(rows) - 1, 1)


def make_history(labels: dict, fitting_need_ids: set[str], smoothing: float) -> tuple[dict[str, tuple[float, int]], float]:
    grades: dict[str, list[int]] = defaultdict(list)
    all_grades = []
    for (need_id, candidate_id), label in labels.items():
        if need_id in fitting_need_ids:
            grade = int(label["grade"])
            grades[candidate_id].append(grade)
            all_grades.append(grade)
    global_mean = mean(all_grades)
    result = {
        candidate_id: (sum(values) / len(values), len(values))
        for candidate_id, values in grades.items()
    }
    return result, global_mean


def make_rows(
    need_ids: set[str],
    pools: dict[str, list[str]],
    labels: dict,
    scores: dict[tuple[str, str], float],
    history: dict[str, tuple[float, int]],
    global_mean: float,
    smoothing: float,
) -> list[dict]:
    output = []
    for need_id in sorted(need_ids):
        per_need = []
        for candidate_id in pools[need_id]:
            key = need_id, candidate_id
            if key not in scores:
                raise ValueError(f"Missing base score for {key}")
            if key not in labels:
                raise ValueError(f"Missing label for {key}")
            raw_mean, count = history.get(candidate_id, (global_mean, 0))
            smoothed = (raw_mean * count + smoothing * global_mean) / (count + smoothing)
            per_need.append({
                "need_id": need_id,
                "candidate_id": candidate_id,
                "relevance_grade": int(labels[key]["grade"]),
                "hard_eligible": int(bool(labels[key]["hard_eligible"])),
                "bge_score_raw": scores[key],
                "history_mean": raw_mean,
                "history_smoothed": smoothed,
                "history_count": float(count),
                "history_log_count": math.log1p(count),
                "history_seen": float(count > 0),
            })
        minmax_and_rank(per_need)
        output.extend(per_need)
    return output


def make_pool(rows: list[dict], columns: list[str], target_transform: str) -> Pool:
    rows = sorted(rows, key=lambda row: (row["need_id"], row["candidate_id"]))
    group_map = {need_id: index for index, need_id in enumerate(sorted({row["need_id"] for row in rows}))}
    matrix = np.asarray([[row[column] for column in columns] for row in rows], dtype=np.float32)
    grades = np.asarray([row["relevance_grade"] for row in rows], dtype=np.float32)
    labels = grades if target_transform == "raw" else np.power(2.0, grades) - 1.0
    group_id = np.asarray([group_map[row["need_id"]] for row in rows], dtype=np.int64)
    return Pool(matrix, label=labels, group_id=group_id, feature_names=columns)


def evaluate(rows: list[dict], predictions: np.ndarray) -> tuple[float, float, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row, score in zip(sorted(rows, key=lambda item: (item["need_id"], item["candidate_id"])), predictions):
        grouped[row["need_id"]].append({**row, "meta_score": float(score)})
    ndcgs, violations, total = [], 0, 0
    submissions = []
    for need_id, values in grouped.items():
        ranked = sorted(values, key=lambda row: (-row["meta_score"], row["candidate_id"]))[:5]
        ideal = sorted((row["relevance_grade"] for row in values), reverse=True)[:5]
        def dcg(grades: list[int]) -> float:
            return sum((2 ** grade - 1) / math.log2(position + 2) for position, grade in enumerate(grades))
        denominator = dcg(ideal)
        ndcgs.append(dcg([row["relevance_grade"] for row in ranked]) / denominator if denominator else 0.0)
        violations += sum(not row["hard_eligible"] for row in ranked)
        total += len(ranked)
        submissions.append({
            "need_id": need_id,
            "ranked_candidates": [
                {"candidate_id": row["candidate_id"], "score": row["meta_score"]}
                for row in ranked
            ],
        })
    return mean(ndcgs), violations / total, submissions


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate a CatBoost meta-ranker.")
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.train.csv")
    parser.add_argument("--pools", type=Path, default=DATA_DIR / "candidate_pools.train.csv")
    parser.add_argument("--need-split", type=Path, default=Path("splits/need_split.csv"))
    parser.add_argument("--folds", type=Path, default=Path("splits/internal_train_folds.csv"))
    parser.add_argument("--oof-scores", type=Path, nargs="+", required=True, help="One or more held-fold BGE score CSVs")
    parser.add_argument("--dev-scores", type=Path, required=True, help="BGE scores for internal-dev from a checkpoint trained on internal-train")
    parser.add_argument("--score-column", default="bge_score")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/meta_ranker"))
    parser.add_argument("--history-smoothing", type=float, default=5.0)
    parser.add_argument("--target-transform", choices=["raw", "ndcg_gain"], default="raw")
    parser.add_argument("--bge-features", choices=["all", "normalized_and_rank", "rank_only"], default="all")
    parser.add_argument("--loss-function", choices=["YetiRank", "YetiRankPairwise"], default="YetiRank")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if CatBoostRanker is None or Pool is None:
        raise RuntimeError(
            "CatBoost is required. Install it in the GPU environment with "
            "`python -m pip install catboost`, then rerun this script."
        )

    labels = load_labels(args.labels)
    pool_rows = read_csv_rows(args.pools)
    pools: dict[str, list[str]] = defaultdict(list)
    for row in pool_rows:
        pools[row["need_id"]].append(row["candidate_id"])
    split = load_mapping(args.need_split, "split")
    folds = {need_id: int(value) for need_id, value in load_mapping(args.folds, "fold").items()}
    train_need_ids = {need_id for need_id, part in split.items() if part == "train"}
    dev_need_ids = {need_id for need_id, part in split.items() if part == "dev"}
    if train_need_ids != set(folds):
        raise ValueError("internal_train_folds.csv must cover exactly the needs marked train in need_split.csv")

    oof_scores = load_score_files(args.oof_scores, args.score_column)
    dev_scores = load_score_files([args.dev_scores], args.score_column)
    expected_train_pairs = {(need_id, candidate_id) for need_id in train_need_ids for candidate_id in pools[need_id]}
    expected_dev_pairs = {(need_id, candidate_id) for need_id in dev_need_ids for candidate_id in pools[need_id]}
    if set(oof_scores) != expected_train_pairs:
        raise ValueError(f"OOF score coverage mismatch: missing={len(expected_train_pairs-set(oof_scores))}, extra={len(set(oof_scores)-expected_train_pairs)}")
    if set(dev_scores) != expected_dev_pairs:
        raise ValueError(f"Dev score coverage mismatch: missing={len(expected_dev_pairs-set(dev_scores))}, extra={len(set(dev_scores)-expected_dev_pairs)}")

    train_rows = []
    for held_fold in sorted(set(folds.values())):
        held_need_ids = {need_id for need_id, fold in folds.items() if fold == held_fold}
        fitting_need_ids = train_need_ids - held_need_ids
        history, global_mean = make_history(labels, fitting_need_ids, args.history_smoothing)
        train_rows.extend(make_rows(held_need_ids, pools, labels, oof_scores, history, global_mean, args.history_smoothing))

    dev_history, dev_global_mean = make_history(labels, train_need_ids, args.history_smoothing)
    dev_rows = make_rows(dev_need_ids, pools, labels, dev_scores, dev_history, dev_global_mean, args.history_smoothing)
    columns = feature_columns(args.bge_features)
    train_pool = make_pool(train_rows, columns, args.target_transform)
    dev_pool = make_pool(dev_rows, columns, args.target_transform)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "meta_train_oof_features.csv", train_rows)
    write_csv(args.output_dir / "meta_dev_features.csv", dev_rows)
    # If labels are already transformed to 2**grade - 1, Base NDCG uses
    # precisely those gains. Exp NDCG would exponentiate them a second time.
    eval_metric = (
        "NDCG:top=5;type=Base"
        if args.target_transform == "ndcg_gain"
        else "NDCG:top=5;type=Exp"
    )
    model = CatBoostRanker(
        loss_function=args.loss_function,
        eval_metric=eval_metric,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_seed,
        task_type=args.task_type,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=dev_pool, early_stopping_rounds=80, use_best_model=True)
    predictions = model.predict(dev_pool)
    ndcg, hcvr, submission = evaluate(dev_rows, predictions)
    model.save_model(args.output_dir / "meta_ranker.cbm")
    sorted_dev_rows = sorted(dev_rows, key=lambda item: (item["need_id"], item["candidate_id"]))
    with (args.output_dir / "meta_dev_scores.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["need_id", "candidate_id", "meta_score"])
        writer.writeheader()
        for row, score in zip(sorted_dev_rows, predictions):
            writer.writerow({
                "need_id": row["need_id"],
                "candidate_id": row["candidate_id"],
                "meta_score": float(score),
            })
    with (args.output_dir / "dev_submission.jsonl").open("w", encoding="utf-8") as file:
        for row in submission:
            file.write(json.dumps(row) + "\n")
    with (args.output_dir / "feature_importance.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["feature", "importance"])
        writer.writeheader()
        # Ranking models default to LossFunctionChange importance, which must
        # be evaluated on the dataset used for fitting.
        for feature, importance in zip(
            columns,
            model.get_feature_importance(data=train_pool),
        ):
            writer.writerow({"feature": feature, "importance": float(importance)})
    print(f"Internal-dev NDCG@5: {ndcg:.6f}")
    print(f"Internal-dev HCVR@5: {hcvr:.6f}")
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Config: loss={args.loss_function}, target={args.target_transform}, bge_features={args.bge_features}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
