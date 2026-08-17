# Final validation report

Validated on 2026-08-15 with `buoi_14/.venv` (Python 3.11.4).

## Artifacts

- [x] Source CSV triplet inspected read-only; controlled fallback to `../ner_kb` documented.
- [x] Normalized corpus: 12,945 unique chunks, 29 documents, 0 missing text.
- [x] BM25 baseline ran.
- [x] Dense multilingual ran in `NEURAL_DENSE` mode; document embeddings are cached locally.
- [x] Hybrid uses BM25 + Dense candidates and RRF; ranks retained.
- [x] Cross-Encoder ran in `NEURAL_CROSS_ENCODER` mode and only processed Hybrid candidates.
- [x] Citation retained through all stages.
- [x] Evaluation ran over 3 verified questions and 4 methods.
- [x] Unit test, Streamlit smoke test and Python compile checks passed.
- [x] Streamlit tested with exact, semantic and Hybrid + Rerank queries.
- [x] Graph hints queried direct relations from Neo4j.
- [x] KG loaded and verified: 13,084 nodes, 26,048 relationships, 0 orphan nodes.
- [x] Six source relationship types were preserved as real Neo4j relationship types.

## Evaluation snapshot

| Method | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|
| BM25 | 0.667 | 0.667 | 0.667 |
| Dense neural | 0.000 | 0.333 | 0.333 |
| Hybrid | 0.000 | 0.333 | 0.333 |
| Hybrid + neural rerank | 0.333 | 0.667 | 0.667 |

These numbers are a lab sanity check; the three-question set is too small for a broad quality claim.

READY FOR DEMO: YES

KG READY: YES
