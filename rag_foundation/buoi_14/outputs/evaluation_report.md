# Evaluation report

Evaluated questions with verified gold: 3

## bm25

- Hit@1: 0.667
- Hit@3: 0.667
- Hit@5: 0.667

## dense

- Hit@1: 0.000
- Hit@3: 0.333
- Hit@5: 0.333

## hybrid

- Hit@1: 0.000
- Hit@3: 0.333
- Hit@5: 0.333

## hybrid_rerank

- Hit@1: 0.333
- Hit@3: 0.667
- Hit@5: 0.667

## Runtime modes

- Dense: NEURAL_DENSE
- Reranker: NEURAL_CROSS_ENCODER

## Limitations

This is a small three-question verified set; it is suitable for a lab sanity check, not a statistically strong benchmark. Failed queries are retained in the CSV.