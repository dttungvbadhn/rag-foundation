# Buổi 09 — Multi-query & Parent–Child Retrieval

## Mục tiêu

Buổi 08 dùng BM25 + semantic, inner RRF và cross-encoder để xếp hạng các chunk
phẳng. Buổi 09 giữ baseline đó nhưng bổ sung query fan-out, tầng RRF thứ hai giữa
các query và mô hình **retrieve child, return parent**. Parent được ghép
deterministic từ text gốc, không được LLM tóm tắt.

Đây là demo kỹ thuật, **không phải tư vấn pháp lý**. OCR, hierarchy và nhãn đánh
giá có thể sai; mọi gold label hiện đều `needs_human_review=true`.

## Pipeline

```text
Q0 ─┐
Q1 ─┼─> Hybrid/query (BM25 + semantic → inner RRF)
Qn ─┘               │
                    v
          Cross-query RRF trên child rank
                    │
                    v
          child_id → hierarchy registry → parent
                    │
                    v
          Parent-RRF → parent cross-encoder
                    │
                    v
          confidence gate → grounded generation → citation
```

Multi-query có hai tầng fusion độc lập:

```text
inner_rrf(d) = bm25_weight/(RRF_K + bm25_rank)
             + semantic_weight/(RRF_K + semantic_rank)

multi_query_rrf(d) = Σ query_weight(q)/(MULTI_QUERY_RRF_K + inner_rank_q(d))
```

Parent aggregation không dùng raw score:

```text
parent_rrf(p) = Σ 1/(PARENT_RRF_K + multi_query_rank(child))
```

Chỉ tối đa `PARENT_SCORE_CHILD_LIMIT` child tốt nhất tính Parent-RRF, nhưng toàn
bộ supporting child vẫn được giữ để giải thích.

## Bốn mode

| Mode | Query | Evidence cuối |
|---|---|---|
| `single_flat` | Q0 | child reranked |
| `multi_flat` | Q0 + variants | child sau MQ-RRF và rerank bằng Q0 |
| `single_parent` | Q0 | parent mở rộng và rerank bằng Q0 |
| `multi_parent` | Q0 + variants | parent sau hai tầng fusion và rerank bằng Q0 |

Generated query chỉ phục vụ retrieval. Reranker và answer prompt luôn dùng câu
hỏi gốc Q0.

## Cấu trúc

```text
rag_advanced/buoi_09/
├── rag.py                 # semantic baseline snapshot
├── advanced_rag.py        # Advanced RAG baseline snapshot
├── hierarchical_rag.py    # hierarchy, multi-query, parent pipeline
├── evaluate.py            # retrieval-only evaluator
├── app.py                 # Streamlit explorer
├── eval/questions.json
├── storage/hierarchy/
├── storage/chroma/
├── storage/huggingface/
├── reports/
└── tests/
```

Buổi 09 đọc chunk hierarchical từ Buổi 05 nhưng không sửa Buổi 05–08.

## Cài đặt và `.env`

Chạy từ thư mục gốc `RAG`. Dùng interpreter Buổi 05:

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m pip install -r .\rag_advanced\buoi_09\requirements.txt
Copy-Item .\rag_advanced\buoi_09\.env.example .\rag_advanced\buoi_09\.env
```

Điền `GEMINI_API_KEY` trong `.env`; không commit file này. Các nhóm biến:

- Gemini embedding/generation model và dimension.
- BM25/semantic candidate K và inner RRF weights.
- `MULTI_QUERY_COUNT`, per-query K, cross-query weights/K.
- `PARENT_MAX_CHARS`, Parent-RRF K, candidate/final parent K.
- `TOTAL_CONTEXT_MAX_CHARS`.
- reranker model/device/batch/min score.

`candidate K` điều khiển độ rộng retrieval; `PARENT_CANDIDATES` giới hạn parent
trước rerank; `FINAL_PARENT_TOP_K` giới hạn sau rerank. Context budget chỉ loại
nguyên parent, không cắt giữa child. Parent đầu tiên quá lớn vẫn được giữ kèm
warning.

## Hierarchy store

Hierarchy resolution ưu tiên metadata → heading đầu chunk → carry-forward trong
cùng source → document fallback. Inline `Điều N` giữa câu không tự động là
heading. Conflict hoặc không chắc chắn tạo `ambiguous/warnings`, không bị che.

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py hierarchy-audit
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py build-hierarchy
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py hierarchy-status
```

Manifest chứa input fingerprints và config identity. Store stale không được tự
build trong query; hãy audit rồi chủ động build lại.

## Query expansion và API budget

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py expand-query --question "Điều kiện vay vốn là gì?"
```

Q0 được code giữ nguyên. Gemini tạo Q1..Qn trong một Generation API call và cache
chỉ trong process. Multi answer tối đa hai Generation calls: một expansion, một
answer. Embedding calls được đếm riêng. Single mode chỉ cần answer Generation
call.

## Retrieval, query và compare

```powershell
# Fan-out child retrieval
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py multi-child --question "Điều kiện vay vốn là gì?"

# Retrieve child, return parent (chưa answer)
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py parent-retrieve --mode multi_parent --question "Điều kiện vay vốn là gì?"

# Answer
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py query --mode multi_parent --question "Điều kiện vay vốn là gì?"

# Bốn mode retrieval/rerank, không answer generation
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\hierarchical_rag.py compare --question "Điều kiện vay vốn là gì?"
```

Parent rerank score là score chuẩn hóa của model, không phải xác suất đúng.
Evidence chỉ được gửi tới generation khi đạt `RERANK_MIN_SCORE`. Citation `[P1]`
được code map về source/page/parent/anchor child thật; label giả bị loại.

## Semantic index và status

Status hierarchy là read-only. Semantic status/prepare dùng baseline snapshot:

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\advanced_rag.py status --strategy hierarchical
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\advanced_rag.py prepare-semantic --strategy hierarchical
```

Prepare semantic gọi Gemini embedding thật và ghi Chroma Buổi 09; không có vector
giả.

## Test, evaluation và Streamlit

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m unittest discover -s .\rag_advanced\buoi_09\tests -v
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_advanced\buoi_09\evaluate.py --k 5
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m streamlit run .\rag_advanced\buoi_09\app.py
```

Evaluator so sánh bốn mode retrieval-only và không gọi answer generation. Report
gồm Child/Parent Recall@K, MRR@K, binary nDCG@K, source/parent coverage, query và
embedding call counts, context chars, expansion factor, mean/p50 latency. Report
được ghi atomically vào `reports/`; `latest_report.json` chỉ xuất hiện sau khi
report hoàn chỉnh hợp lệ.

Không được tuyên bố `multi_parent` thắng khi nhãn còn human review hoặc metric
không hỗ trợ kết luận.

## Câu hỏi so sánh thủ công

1. `Điều 8 quy định những nhu cầu vốn nào không được cho vay?`
2. `Khách hàng cần đáp ứng những yêu cầu gì để được tổ chức tín dụng xem xét cho vay?`
3. `Điều kiện vay vốn và những nhu cầu vốn không được cho vay được quy định như thế nào?`
4. `Quy định về cơ cấu lại thời hạn trả nợ gồm điều kiện, thời gian và trách nhiệm nào?`
5. `Lãi suất tiết kiệm cao nhất trên thị trường hôm nay là bao nhiêu?`

Câu 5 dùng để quan sát false positive. Nếu vẫn có evidence đạt gate, phải ghi nhận
và hiệu chỉnh; không sửa output thủ công.

## Troubleshooting

- **Hierarchy stale:** input/config khác manifest; chạy audit rồi build lại.
- **Collection missing/mismatch:** chạy status, sau đó prepare semantic đúng model,
  dimension và strategy.
- **Thiếu API key:** tạo `.env`; không dán key vào log/chat.
- **Reranker unavailable:** kiểm tra Internet lần tải đầu, disk/RAM, cache và
  `RERANK_DEVICE`; không có silent fallback.
- **Latency cao:** multi-query tăng embedding/retrieval calls; giảm query/candidate
  K sau khi đo recall.
- **Context lớn:** giảm parent/final K hoặc `PARENT_MAX_CHARS`, sau đó build lại
  hierarchy; không truncate pháp lý âm thầm.
- **Ambiguous:** xem Parent–Child Explorer và registry warnings; cần human review.
- **OCR lỗi:** metadata/headings có thể sai; không coi output là kết luận pháp lý.

Nội dung child/parent được gửi tới Gemini và reranker khi chạy thật. Chỉ sử dụng
dữ liệu mà người vận hành được phép gửi tới dịch vụ bên ngoài.
