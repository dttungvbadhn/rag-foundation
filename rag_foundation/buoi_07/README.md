# RAG Foundation — Buổi 07

## 1. Mục tiêu

Buổi 07 xây dựng một pipeline Retrieval-Augmented Generation (RAG) tối giản:
validate chunk JSON, tạo Gemini embedding, lưu vector bằng ChromaDB persistent,
retrieval theo cosine distance, confidence gate, generation có grounding, citation
từ metadata thật và giao diện Streamlit.

Đây là project học tập. Kết quả không phải tư vấn pháp lý hoặc kết luận chuyên môn.

## 2. Quan hệ với Buổi 05 và Buổi 06

- Buổi 05 cung cấp `.venv` và dữ liệu chunk JSON đã chuẩn bị tại
  `buoi_05/output/chunks/`.
- Buổi 07 chỉ đọc dữ liệu Buổi 05; không OCR, parse PDF hoặc chunk lại.
- Buổi 06 là tài liệu tham khảo về pipeline trước đó; Buổi 07 không sửa hoặc sao
  chép trực tiếp code Buổi 06.
- Buổi 07 không tạo virtual environment riêng.

## 3. Pipeline

```text
JSON Buổi 05
    │
    ▼
Loader + Validator ──► Gemini document embedding ──► Chroma persistent
                                                        │
Câu hỏi ──► Gemini query embedding ──► Retrieval cosine │
                                                        ▼
            Citation từ metadata ◄── Generation ◄── Confidence gate
                    │
                    ▼
              CLI / Streamlit
```

Confidence gate chỉ đưa evidence có distance đạt ngưỡng vào generation. Evidence
không đạt vẫn được giữ trong kết quả để kiểm tra.

## 4. Cấu trúc thư mục

```text
buoi_07/
├── .env.example
├── .gitignore
├── README.md
├── SPEC_buoi_07.md
├── app.py
├── buoi_07.md
├── rag.py
├── requirements.txt
├── storage/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_rag.py
    └── fixtures/
        └── chunks_sample.json
```

Index thật, nếu được tạo, nằm trong `storage/chroma/` và bị `.gitignore` bỏ qua.

## 5. Điều kiện đầu vào

- Terminal đứng tại thư mục gốc `RAG`, nơi chứa trực tiếp `rag_foundation/`.
- Python 3.11 trở lên trong `.venv` Buổi 05 hoạt động cùng pip.
- `rag_foundation/buoi_05/output/chunks/` có ít nhất một JSON hợp lệ.
- Mỗi chunk có `chunk_id`, `strategy`, `source`, `page_start`, `page_end`, `text`.
- Muốn index/query thật cần Gemini API key hợp lệ và quyền gửi dữ liệu đó tới
  dịch vụ Gemini.

## 6. Python và cài package

Project bắt buộc dùng `.venv` Buổi 05, không dùng `python` hoặc `pip` chung chung.

Windows PowerShell:

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m pip install -r .\rag_foundation\buoi_07\requirements.txt
```

Linux/macOS:

```bash
./rag_foundation/buoi_05/.venv/bin/python -m pip install -r ./rag_foundation/buoi_07/requirements.txt
```

## 7. Cấu hình `.env`

Tạo file cá nhân từ mẫu. `.env` đã được `.gitignore` và không được commit.

Windows PowerShell:

```powershell
Copy-Item .\rag_foundation\buoi_07\.env.example .\rag_foundation\buoi_07\.env
```

Linux/macOS:

```bash
cp ./rag_foundation/buoi_07/.env.example ./rag_foundation/buoi_07/.env
```

Các biến:

| Biến | Ý nghĩa |
|---|---|
| `GEMINI_API_KEY` | Key gọi Gemini; không in, chia sẻ hoặc commit |
| `GEMINI_EMBEDDING_MODEL` | Model tạo document/query embedding |
| `GEMINI_EMBEDDING_DIM` | Số chiều vector, từ 128 đến 3072 |
| `GEMINI_GENERATION_MODEL` | Model tổng hợp câu trả lời |
| `DEFAULT_TOP_K` | Số evidence mặc định, từ 1 đến 20 |
| `RAG_MAX_DISTANCE` | Ngưỡng cosine distance không âm cho confidence gate |

Code nạp `.env` bằng đường dẫn tuyệt đối suy ra từ `Path(__file__).resolve()`;
không phụ thuộc current working directory.

## 8. Lệnh Windows PowerShell

### Validate

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py validate --strategy hierarchical
```

### Status read-only

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py status --strategy hierarchical
```

### Index

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py index --strategy hierarchical
```

### Reset đúng collection đích rồi index lại

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py index --strategy hierarchical --reset
```

`--reset` chỉ xóa collection ứng với strategy/model/dimension hiện tại và chỉ
thực hiện sau khi toàn bộ embedding mới đã được validate.

### Query CLI

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe .\rag_foundation\buoi_07\rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"
```

### Chạy test

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m unittest discover -s .\rag_foundation\buoi_07\tests -v
```

### Streamlit

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m streamlit run .\rag_foundation\buoi_07\app.py
```

## 9. Lệnh Linux/macOS

### Validate

```bash
./rag_foundation/buoi_05/.venv/bin/python ./rag_foundation/buoi_07/rag.py validate --strategy hierarchical
```

### Status read-only

```bash
./rag_foundation/buoi_05/.venv/bin/python ./rag_foundation/buoi_07/rag.py status --strategy hierarchical
```

### Index

```bash
./rag_foundation/buoi_05/.venv/bin/python ./rag_foundation/buoi_07/rag.py index --strategy hierarchical
```

### Reset đúng collection đích rồi index lại

```bash
./rag_foundation/buoi_05/.venv/bin/python ./rag_foundation/buoi_07/rag.py index --strategy hierarchical --reset
```

### Query CLI

```bash
./rag_foundation/buoi_05/.venv/bin/python ./rag_foundation/buoi_07/rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"
```

### Chạy test

```bash
./rag_foundation/buoi_05/.venv/bin/python -m unittest discover -s ./rag_foundation/buoi_07/tests -v
```

### Streamlit

```bash
./rag_foundation/buoi_05/.venv/bin/python -m streamlit run ./rag_foundation/buoi_07/app.py
```

Dừng Streamlit bằng `Ctrl+C` tại terminal đã khởi chạy server.

## 10. Khái niệm chính

- **Strategy:** cách chunk được tạo (`hierarchical`, `semantic`, `fixed-size`).
  Mỗi collection chỉ chứa một strategy.
- **Embedding model:** model Gemini biến document/câu hỏi thành vector.
- **Embedding dimension:** số phần tử của vector. Index và query bắt buộc khớp.
- **Collection identity:** tên dạng
  `nhnn-<strategy>-<dimension>-<model_hash>`, phân biệt strategy, dimension và
  hash ổn định của model.
- **Top-k:** số kết quả retrieval tối đa; hệ thống tự giới hạn theo collection count.
- **Cosine distance:** độ xa giữa query và chunk; thấp hơn thường liên quan hơn.
  Đây không phải xác suất hay độ tin cậy tuyệt đối.
- **RAG_MAX_DISTANCE:** ngưỡng demo để đánh dấu evidence được chấp nhận.
- **Confidence gate:** chỉ evidence có `distance <= RAG_MAX_DISTANCE` được đưa
  vào generation. Ngưỡng cần được đánh giá và hiệu chỉnh trên dữ liệu thực tế.
- **Retrieval-only:** đã lấy được evidence nhưng generation lỗi hoặc trả text
  rỗng; evidence vẫn được trả về, citation để trống.
- **Citation:** code thay label như `[E1]` bằng source, trang và `chunk_id` lấy từ
  metadata Chroma thật. Hệ thống không tin metadata do LLM tự tạo.

## 11. Kế hoạch kiểm tra thủ công

Chỉ thực hiện sau khi index dữ liệu thật. Kết quả phải dựa trên collection đang
dùng; không khẳng định trước rằng câu A hoặc B chắc chắn có answer.

### A. Có khả năng thuộc tài liệu

```text
Cơ cấu lại thời hạn trả nợ được quy định như thế nào?
```

### B. Có khả năng thuộc tài liệu

```text
Việc phân loại nợ và trích lập dự phòng được thực hiện như thế nào?
```

### C. Ngoài phạm vi

```text
Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?
```

Kỳ vọng mong muốn cho C: evidence không đạt threshold, generation không được gọi
và hệ thống trả `Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung
cấp.`; không bịa tên ngân hàng hoặc lãi suất. Đây không phải kết quả được bảo đảm
trước khi hiệu chỉnh threshold. Nếu C vẫn đạt threshold, phải ghi nhận false
positive của retrieval/gate, không đánh dấu PASS giả và không sửa answer thủ công
để che lỗi.

## 12. Troubleshooting

### Thiếu package

Chạy lại lệnh cài `requirements.txt` bằng đúng interpreter Buổi 05. Kiểm tra bằng:

```powershell
.\rag_foundation\buoi_05\.venv\Scripts\python.exe -m pip --version
```

### Sai interpreter

Nếu traceback trỏ tới Python khác, gọi interpreter bằng đường dẫn đầy đủ như các
lệnh phía trên; không dùng `python` hoặc `pip` chung chung.

### Thiếu API key

Điền `GEMINI_API_KEY` vào `buoi_07/.env`. Không dán key vào chat, source code,
terminal history công khai hoặc commit Git. `validate` và `status` vẫn chạy khi
thiếu key; `index` và `query` thật sẽ dừng.

### Collection rỗng hoặc chưa tồn tại

Chạy `status`, sau đó chạy `index` cho đúng strategy. UI không tự index khi mở.

### Model hoặc dimension mismatch

Không query nhầm collection cũ. Xác minh `.env`, rồi dùng `index --reset` cho
đúng collection identity. Không xóa toàn bộ storage.

### JSON lỗi

Command `validate` báo tên file và vị trí record. Sửa dữ liệu nguồn ở đúng quy
trình tạo dữ liệu; Buổi 07 không tự sửa JSON Buổi 05.

### Embedding lỗi hoặc rate limit

Kiểm tra key, model, quota và rate limit. Pipeline dừng toàn batch trước upsert;
không tạo vector giả và không reset collection hợp lệ cũ khi embedding lỗi.
Chờ theo chính sách dịch vụ rồi chạy lại command index.

## 13. Giới hạn và cảnh báo

- Đây là demo học tập, không phải hệ thống production và không phải tư vấn pháp lý.
- Threshold cần hiệu chỉnh bằng tập câu hỏi đánh giá; giá trị mặc định không bảo
  đảm chặn hết false positive hoặc giữ hết true positive.
- Retrieval có thể bỏ sót thông tin hoặc trả evidence chưa đủ.
- Không có OCR, reranker, hybrid search, hội thoại nhiều lượt, RBAC hay deployment.
- Chất lượng phụ thuộc dữ liệu chunk, model, dimension và cấu hình retrieval.
- Nội dung chunk được gửi tới Gemini khi tạo embedding và generation. Chỉ dùng dữ
  liệu mà người vận hành được phép gửi tới dịch vụ bên ngoài, tuân thủ yêu cầu
  bảo mật, quyền riêng tư và chính sách dữ liệu áp dụng.
