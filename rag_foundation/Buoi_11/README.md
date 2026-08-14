# Graph RAG Lab 2 — Bước 1

Module này kết nối tới Neo4j cục bộ và kiểm tra database `kb-hops` bằng truy vấn chỉ đọc.

```powershell
..\buoi_05\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\buoi_05\.venv\Scripts\python.exe neo4j_connection.py status
```

Sao chép `.env.example` thành `.env` và điền mật khẩu thực tế. Không commit `.env`.

## Bước 2 — Vector + multi-hop

`search_context()` trong `graph_retrieval.py` dùng model
`thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5` (vector 384 chiều), lấy `TOP_K`
seed từ Neo4j vector index rồi duyệt vô hướng qua `CAN_CU`, `THAY_THE`, `HOP_NHAT`
đến `MAX_HOPS`. Model chỉ được tải khi thực sự gọi retrieval.

Tên vector index và thuộc tính phải khớp schema/index đã tạo ở Bài 1. Vector lưu trong
index cũng phải được tạo bằng cùng model và cùng số chiều; nếu không, điểm tương đồng
không có ý nghĩa hoặc Neo4j sẽ báo sai dimension.

Ví dụ Python:

```python
from graph_retrieval import RetrievalConfig, search_context

result = search_context(
    "Điều kiện vay vốn được quy định thế nào?",
    retrieval_config=RetrievalConfig(top_k=5, max_hops=2),
)
for item in result["context"]:
    print(item["id"], item["text"])
```

## Bước 3 — Gemini QA

`answer_question()` trong `qa_pipeline.py` ghép seed và related evidence, gắn mã `[S1]`,
`[R1]` rồi gọi `gemini-flash-latest` đúng một lần. System prompt mô tả schema đồ thị,
phân cấp Chương/Mục/Điều/Khoản/Điểm và ý nghĩa ba loại quan hệ. Mô hình được yêu cầu
không suy đoán, chỉ trả lời theo evidence và trích mã nguồn.

```python
from qa_pipeline import answer_question

result = answer_question("Điều kiện vay vốn được quy định thế nào?")
print(result["status"])
print(result["answer"])
```

Nếu retrieval rỗng, pipeline trả `insufficient_evidence` và không gọi Gemini. Nếu thiếu
`GEMINI_API_KEY`, evidence vẫn được giữ trong kết quả với status
`generation_unavailable`. `MAX_CONTEXT_CHARS` giới hạn lượng văn bản gửi ra API; pipeline
chỉ nhận cả evidence record, không cắt giữa đoạn.

## Bước 4 — Kiểm thử và đánh giá

Năm câu hỏi trong `eval/questions.json` bao phủ chuỗi thay thế, văn bản hợp nhất, sửa đổi,
căn cứ pháp lý và văn bản điều chỉnh hoạt động. Chạy retrieval-only mặc định:

```powershell
..\buoi_05\.venv\Scripts\python.exe evaluate.py --top-k 5 --max-hops 2
```

Chỉ thêm `--generate` khi chủ động cho phép gọi Gemini. Báo cáo JSON ghi seed/related IDs,
hop lớn nhất, độ phủ loại quan hệ mong đợi, latency, lỗi và số generation call. Các nhãn
quan hệ chỉ là kiểm tra đường đi, không phải gold answer; cả năm câu đều đánh dấu
`needs_human_review=true`, vì vậy không được dùng báo cáo để khẳng định câu trả lời đúng
về mặt pháp lý khi chưa được chuyên gia duyệt.
