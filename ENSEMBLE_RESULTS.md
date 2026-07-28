# Ensemble results

## Models

- BM25 with Russian stemming
- BAAI/bge-m3
- BAAI/bge-reranker-v2-m3

## Final ensemble weights

- BM25: 0.30
- BGE-M3: 0.30
- Reranker: 0.40

Scores are min-max normalized separately inside each candidate pool.

## Evaluation results

| Dataset | NDCG@5 | HCVR@5 |
|---|---:|---:|
| 5-fold OOF on internal_train | 0.793382 | 0.004723 |
| internal_dev | 0.809356 | 0.006604 |
| mentor validation | 0.834492 | 0.004615 |

The final weights were selected using 5-fold cross-validation on `internal_train`.
The `internal_dev` and mentor validation labels were not used for weight selection.
