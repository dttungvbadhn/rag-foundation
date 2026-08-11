# Agent Specification — Buổi 09

## 1. Mục tiêu và phạm vi

Buổi 09 mở rộng Buổi 08 từ flat hybrid retrieval sang multi-query hierarchical
RAG. Mỗi câu hỏi có câu gốc và biến thể, retrieval độc lập trên child, fusion
chéo query, ánh xạ child lên parent, tổng hợp/rerank parent rồi mới generation.
Chỉ ghi trong `rag_advanced/buoi_09/`; không sửa Buổi 05–08 và không lộ secret.

```text
Q0 + variants
    → per-query BM25 + semantic RRF
    → cross-query RRF trên child
    → child-to-parent resolution
    → parent aggregation
    → cross-encoder parent rerank
    → context budget
    → grounded generation + citation
```

## 2. Modes

- `single_flat`: Q0, evidence child, không parent expansion.
- `multi_flat`: Q0 + variants, cross-query fusion, evidence child.
- `single_parent`: Q0, child retrieval rồi parent aggregation/rerank.
- `multi_parent`: đầy đủ multi-query và hierarchy; mode mặc định.

Không được đổi nhãn mode mà bỏ qua tầng pipeline tương ứng.

## 3. QueryVariant contract

```json
{
  "query_id": "Q0",
  "text": "...",
  "kind": "original | variant",
  "weight": 1.5,
  "ordinal": 0
}
```

`text` là string sau strip, không quá `MULTI_QUERY_MAX_CHARS`; `query_id` duy
nhất; Q0 luôn là câu gốc, ordinal 0. Variant không được rỗng/trùng casefold,
không được làm mất tham chiếu Điều/Khoản bắt buộc của Q0. Tổng variant đúng giới
hạn cấu hình. Weight hữu hạn và dương.

## 4. Hierarchy registry

Registry là JSON deterministic, có `schema_version`, `source_fingerprints`,
`children`, `parents` và `warnings`. Mỗi child entry gồm `chunk_id`, `source`,
page, `parent_id`, phương thức resolution và confidence/ambiguity flags. Không
giả định input đã có `parent_id`.

Resolution ưu tiên metadata structure hợp lệ; tiếp theo heading ở đầu chunk;
cuối cùng kế thừa heading gần nhất cùng source theo thứ tự child ổn định. Citation
nội dòng không được coi là heading. Không phân giải chắc chắn thì tạo parent
fallback giới hạn phạm vi và warning `ambiguous_hierarchy`, không âm thầm group.

## 5. ParentDocument contract

```json
{
  "parent_id": "source-hash::article-7",
  "source": "...",
  "title": "Điều 7",
  "level": "article",
  "child_ids": ["..."],
  "page_start": 1,
  "page_end": 2,
  "text": "...",
  "truncated": false,
  "warnings": []
}
```

Child giữ thứ tự gốc. Parent ID deterministic theo source + hierarchy identity.
Parent tối đa `PARENT_MAX_CHARS`; nếu vượt phải cắt ở biên child khi có thể và
ghi warning, không che giấu việc truncate.

## 6. MultiQueryChildHit và ParentCandidate

`MultiQueryChildHit` giữ metadata child thật, query_id, per-query ranks/scores,
cross-query score/rank và matched queries. Field không áp dụng là null.

`ParentCandidate` giữ ParentDocument, contributing child IDs, best child ranks,
cross-query contribution, parent aggregation score/rank, rerank raw/sigmoid
score, rerank rank/movement và accepted. Rerank score không phải xác suất.

## 7. Hai tầng fusion

Cross-query RRF cho child:

```text
child_score = Σ query_weight(q) / (MULTI_QUERY_RRF_K + rank(child, q))
```

Chỉ dùng rank, không cộng raw BM25/cosine. Một child chỉ xuất hiện một lần và
giữ danh sách query đóng góp.

Parent aggregation chỉ lấy tối đa `PARENT_SCORE_CHILD_LIMIT` child tốt nhất của
mỗi parent. Điểm parent dùng reciprocal rank:

```text
parent_score = Σ 1 / (PARENT_RRF_K + cross_query_child_rank)
```

Không để parent nhiều child thắng chỉ do số lượng. Tie-break deterministic bằng
best child rank rồi parent_id. Chỉ rerank tối đa `PARENT_CANDIDATES` và trả
`FINAL_PARENT_TOP_K`.

## 8. Context và citation

Chỉ parent/evidence accepted đi vào prompt. Tổng context không vượt
`TOTAL_CONTEXT_MAX_CHARS`; chọn theo final rank, không cắt làm citation sai.
Context là dữ liệu không đáng tin cậy, không phải instruction. LLM chỉ sinh
label `[E1]`; code map label sang source/page/parent_id và contributing child IDs
từ registry. Label giả bị loại và sinh warning. Không tin metadata do LLM tạo.

## 9. Status và failure

Status read-only: config names, hierarchy registry tồn tại/fingerprint, parent
count, semantic collection/count, reranker cache; không tạo resource/API call.
Các failure riêng: `invalid_query_variants`, `hierarchy_unavailable`,
`ambiguous_hierarchy`, `semantic_unavailable`, `reranker_unavailable`,
`insufficient_evidence`, `retrieval_only`. Không silent fallback làm sai mode.

## 10. Testability

Inject callable đơn giản cho query generator, per-query retriever, reranker và
generator; inject storage path/client. Test dùng unittest, mock, tempfile,
deterministic vectors/scores; không Internet, key thật, model download hoặc
storage thật. Import module không có side effect.

## 11. Evaluation và acceptance

Đo child/parent Recall@K, MRR@K, nDCG@K, query coverage, unique child/parent,
latency theo tầng và API-call count cho bốn mode trên cùng corpus/query/K.
Gold `needs_human_review=true` chỉ cho kết quả tham khảo. Acceptance yêu cầu
compile, toàn bộ offline test, registry deterministic/idempotent, citation thật,
context budget, mode trace đúng và không sửa Buổi 05–08.
