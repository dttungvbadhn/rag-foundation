# Advanced RAG — Buổi 08

## Mục tiêu

Buổi 07 là semantic RAG baseline. Buổi 08 bổ sung BM25 để bắt từ khóa pháp lý chính xác, RRF để hợp nhất lexical/semantic ranking và cross-encoder để đọc đồng thời câu hỏi–đoạn văn rồi xếp hạng lại. Đây là demo học tập, **không phải tư vấn pháp lý**.

```text
Chunk Buổi 05 ─┬─> BM25 ───────┐
               └─> Semantic ───┴─> RRF ─> Cross-encoder ─> Gate ─> Generation/Citation
```

Buổi 08 sao chép semantic baseline vào `rag.py` và chạy độc lập; không import runtime, `.env` hoặc storage của Buổi 07.

## Cấu trúc

```text
buoi_08/
├── rag.py                 # semantic baseline độc lập
├── advanced_rag.py        # BM25, semantic, RRF, reranker, answer/compare
├── evaluate.py            # Recall/MRR/nDCG và latency
├── app.py                 # dashboard 4 tab
├── eval/questions.json    # gold starter cần human review
├── tests/                 # unittest offline
├── reports/               # evaluation JSON (gitignore)
└── storage/               # Chroma và Hugging Face cache (gitignore)
```

## Cài đặt

Chạy từ thư mục gốc `RAG`. Dùng lại `.venv` Buổi 05:

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m pip install -r .\rag_foundation\buoi_08\requirements.txt
Copy-Item .\rag_foundation\buoi_08\.env.example .\rag_foundation\buoi_08\.env
```

Điền `GEMINI_API_KEY` trong `.env`, không đưa key vào Git. Các biến candidate K điều khiển số đoạn mỗi tầng; `FINAL_TOP_K` là số đoạn còn lại sau rerank. Model mặc định `BAAI/bge-reranker-v2-m3` có thể lớn, lần đầu cần Internet, dung lượng đĩa và RAM; CPU có thể chậm.

## Lệnh sử dụng

```powershell
$pythonBuoi05 = ".\rag_foundation\buoi_05\.venv\Scripts\python.exe"
& $pythonBuoi05 .\rag_foundation\buoi_08\advanced_rag.py status --strategy hierarchical
& $pythonBuoi05 .\rag_foundation\buoi_08\advanced_rag.py prepare-semantic --strategy hierarchical
& $pythonBuoi05 .\rag_foundation\buoi_08\advanced_rag.py bm25 --strategy hierarchical --question "Điều 7 quy định gì?"
& $pythonBuoi05 .\rag_foundation\buoi_08\advanced_rag.py hybrid --strategy hierarchical --question "Điều 7 quy định gì?"
& $pythonBuoi05 .\rag_foundation\buoi_08\advanced_rag.py rerank --strategy hierarchical --question "Điều 7 quy định gì?"
& $pythonBuoi05 .\rag_foundation\buoi_08\advanced_rag.py query --mode hybrid_rerank --strategy hierarchical --question "Điều 7 quy định gì?"
& $pythonBuoi05 .\rag_foundation\buoi_08\advanced_rag.py compare --strategy hierarchical --question "Điều 7 quy định gì?"
& $pythonBuoi05 -m unittest discover -s .\rag_foundation\buoi_08\tests -v
& $pythonBuoi05 .\rag_foundation\buoi_08\evaluate.py --strategy hierarchical --k 5
& $pythonBuoi05 -m streamlit run .\rag_foundation\buoi_08\app.py
```

Streamlit không tự index, tải reranker hoặc chạy evaluation khi mở trang. Dừng server bằng `Ctrl+C`.

## Hiểu score và K

- **BM25 score:** cao hơn thường khớp lexical tốt hơn; không phải xác suất.
- **Cosine distance:** thấp hơn thường gần semantic hơn; gate dùng `RAG_MAX_DISTANCE`.
- **RRF score:** tổng đóng góp reciprocal rank, không cộng raw BM25 với distance.
- **Rerank score:** sigmoid của cross-encoder logit; cao hơn tốt hơn nhưng không phải xác suất đúng.
- **Candidate K:** giữ đủ candidate trước tầng sau; K lớn tăng recall và chi phí/latency.
- **Final K:** số evidence tối đa sau rerank.

## Evaluation

- Recall@K: tỷ lệ relevant chunk được tìm thấy.
- MRR@K: reciprocal rank của relevant chunk đầu tiên.
- nDCG@K: đánh giá vị trí các relevant chunk với relevance nhị phân.
- Báo cáo còn `needs_human_review=true` không được dùng để tuyên bố mode chiến thắng chính thức.
- Evaluator chỉ retrieval/rerank, không generation. Một query lỗi được ghi `failed`, không bỏ qua.

## Câu hỏi so sánh thủ công

1. Exact legal reference: `Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?`
2. Paraphrase semantic: `Khách hàng gặp khó khăn có thể được điều chỉnh kỳ hạn trả nợ ra sao?`
3. Multi-concept: `Phân loại nợ và trích lập dự phòng được thực hiện như thế nào?`
4. Out-of-scope: `Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?`

Không khẳng định trước mode nào thắng. Với câu ngoài phạm vi, nếu gate vẫn chấp nhận thì ghi nhận false positive thay vì sửa output thủ công.

## Troubleshooting

- **Sai interpreter/thiếu package:** chạy lại pip bằng đúng `.venv` Buổi 05 và file requirements trên.
- **Thiếu API key:** tạo `.env` từ example; không dán key vào chat/log/Git.
- **Semantic collection rỗng:** chủ động chạy `prepare-semantic`.
- **Collection/model/dimension mismatch:** dùng đúng `.env` đã index hoặc chuẩn bị lại đúng collection.
- **Model download lỗi:** kiểm tra Internet, proxy và dung lượng `storage/huggingface/`; không có runtime fallback giả.
- **CPU chậm/thiếu RAM:** giảm `RERANK_CANDIDATES`, `RERANK_BATCH_SIZE`; không khẳng định rerank đã chạy nếu model lỗi.
- **Gemini/rate limit:** thử lại sau và kiểm tra quota; retrieval evidence vẫn cần được giữ rõ ràng.

## Giới hạn

Gold labels chưa được chuyên gia pháp lý duyệt; corpus workshop nhỏ; threshold cần hiệu chỉnh; retrieval có thể bỏ sót hoặc tạo false positive. Nội dung chunk được gửi tới Gemini để embedding/generation, vì vậy chỉ dùng dữ liệu mà người vận hành được phép gửi tới dịch vụ bên ngoài.
