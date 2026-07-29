import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from baselines.pairwise import (
    predict,
    read_jsonl,
    read_labels,
    read_pools,
    read_split,
)


DATA = Path(
    "/home/User25/avito_airi_3_case/student_package/data"
)
SPLIT_PATH = Path(
    "/home/User25/avito_3_recovered/splits/need_split.csv"
)
CHECKPOINT = Path(
    "/home/User25/outputs/bge_m3_pairwise/best"
)
OUTPUT_PATH = Path(
    "/home/User25/outputs/bge_m3_hard_negatives/"
    "hard_pairs.train.jsonl"
)

BATCH_SIZE = 32
MAX_LENGTH = 512

# Максимальное количество самых сложных пар
# для одной потребности.
MAX_HARD_PAIRS_PER_NEED = 10


def effective_grade(label):
    """
    Формально недопустимый кандидат получает 0,
    как в текущей лучшей модели с hard_penalty=0.
    """
    if not label["hard_eligible"]:
        return 0.0

    return float(label["grade"])


need_rows = read_jsonl(DATA / "needs.train.jsonl")
provider_rows = read_jsonl(DATA / "providers.jsonl")

needs = {
    row["need_id"]: row["need_text"]
    for row in need_rows
}

providers = {
    row["candidate_id"]: row["profile_summary"]
    for row in provider_rows
}

pools = read_pools(DATA / "candidate_pools.train.csv")
labels = read_labels(DATA / "labels.train.csv")
split_by_need = read_split(SPLIT_PATH)

# Важно: используем только train.
# Dev не участвует в создании новых обучающих пар.
train_need_ids = [
    need_id
    for need_id in needs
    if split_by_need.get(need_id) == "train"
    and need_id in pools
]

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Checkpoint:", CHECKPOINT)
print("Train needs:", len(train_need_ids))

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    dtype=torch.float32,
).to(device)

print("\nОцениваем всех кандидатов train...")

predictions = predict(
    model=model,
    tokenizer=tokenizer,
    need_ids=train_need_ids,
    needs=needs,
    providers=providers,
    pools=pools,
    device=device,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
)

hard_pairs = []
needs_with_inversions = 0
total_inversions = 0

for need_id in train_need_ids:
    score_by_candidate = {
        row["candidate_id"]: float(row["score"])
        for row in predictions[need_id]
    }

    candidates = []

    for candidate_id in pools[need_id]:
        label = labels.get((need_id, candidate_id))

        if label is None:
            continue

        candidates.append({
            "candidate_id": candidate_id,
            "grade": int(label["grade"]),
            "hard_eligible": bool(label["hard_eligible"]),
            "effective_grade": effective_grade(label),
            "score": score_by_candidate[candidate_id],
        })

    possible_pairs = []
    need_inversions = 0

    for positive in candidates:
        for negative in candidates:
            if (
                positive["effective_grade"]
                <= negative["effective_grade"]
            ):
                continue

            # Чем больше hardness, тем сильнее модель ошибается.
            #
            # hardness > 0:
            # отрицательный кандидат уже стоит выше положительного.
            #
            # hardness около 0:
            # модель почти не различает кандидатов.
            hardness = (
                negative["score"]
                - positive["score"]
            )

            if hardness >= 0:
                need_inversions += 1

            grade_gap = (
                positive["effective_grade"]
                - negative["effective_grade"]
            )

            possible_pairs.append({
                "need_id": need_id,
                "pos_candidate_id":
                    positive["candidate_id"],
                "neg_candidate_id":
                    negative["candidate_id"],
                "pos_grade": positive["grade"],
                "neg_grade": negative["grade"],
                "pos_hard_eligible":
                    positive["hard_eligible"],
                "neg_hard_eligible":
                    negative["hard_eligible"],
                "pos_effective_grade":
                    positive["effective_grade"],
                "neg_effective_grade":
                    negative["effective_grade"],
                "pos_score": positive["score"],
                "neg_score": negative["score"],
                "hardness": hardness,
                "weight": grade_gap,
                "is_inversion": hardness >= 0,
            })

    # Сначала реальные ошибки, затем самые близкие пары.
    possible_pairs.sort(
        key=lambda row: (
            -row["hardness"],
            -row["weight"],
            row["pos_candidate_id"],
            row["neg_candidate_id"],
        )
    )

    selected = possible_pairs[
        :MAX_HARD_PAIRS_PER_NEED
    ]

    hard_pairs.extend(selected)

    if need_inversions > 0:
        needs_with_inversions += 1

    total_inversions += need_inversions


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    for row in hard_pairs:
        file.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


selected_inversions = sum(
    row["is_inversion"]
    for row in hard_pairs
)

hardness_values = np.asarray([
    row["hardness"]
    for row in hard_pairs
])

print("\n" + "=" * 72)
print("РЕЗУЛЬТАТ HARD-NEGATIVE MINING")
print("=" * 72)
print("Train needs:", len(train_need_ids))
print("Выбрано hard-пар:", len(hard_pairs))
print(
    "Потребностей хотя бы с одной ошибкой:",
    needs_with_inversions,
)
print(
    "Всего ошибочных перестановок:",
    total_inversions,
)
print(
    "Ошибочных пар среди выбранных:",
    selected_inversions,
)
print(
    "Средняя hardness:",
    f"{hardness_values.mean():.6f}",
)
print(
    "Максимальная hardness:",
    f"{hardness_values.max():.6f}",
)
print("Сохранено:", OUTPUT_PATH)

print("\n10 самых сильных ошибок модели:")

for row in sorted(
    hard_pairs,
    key=lambda item: -item["hardness"],
)[:10]:
    print(
        row["need_id"],
        "|",
        f"grade {row['pos_grade']}"
        f" > {row['neg_grade']}",
        "|",
        f"score {row['pos_score']:.3f}"
        f" < {row['neg_score']:.3f}",
        "|",
        f"hardness={row['hardness']:+.3f}",
    )
