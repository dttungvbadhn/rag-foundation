# Agent Specification — Buổi 07

## Workspace

- Được đọc `buoi_05/output/chunks/`, `.venv` của Buổi 05, `buoi_06/` và `buoi_07/`.
- Chỉ được ghi trong `buoi_07/`.
- Không sửa bất kỳ code, output, dữ liệu hoặc cấu hình nào của Buổi 05 và Buổi 06.

## Python

- Dùng interpreter trong `.venv` của Buổi 05.
- Không tạo môi trường ảo mới.

## Input

- Đầu vào là các file JSON trong `buoi_05/output/chunks/`.
- Buổi 05 là nguồn dữ liệu đã được chuẩn bị.
- Không OCR, parse PDF hoặc chia chunk lại.

## Packages

Chỉ dùng các package trực tiếp được quy định trong `requirements.txt`: Streamlit,
Google Gen AI SDK, ChromaDB và python-dotenv. Không tự ý thêm dependency trực tiếp.

## Pipeline

Triển khai tuần tự: validate dữ liệu, tạo embedding, lưu Chroma persistent,
retrieval, confidence gate, generation, citation, giao diện Streamlit và unittest
offline.

## Data Contract

Mỗi chunk phải có đủ các field:

- `chunk_id`
- `strategy`
- `source`
- `page_start`
- `page_end`
- `text`

## Index Contract

- Mỗi strategy nằm trong một collection riêng.
- Model và dimension dùng khi index/query phải khớp nhau.
- Dùng embedding thật; không dùng vector giả.
- Từ chối NaN, Infinity, boolean và zero vector.
- Chroma dùng cosine và `embedding_function=None`.
- Thao tác index phải idempotent.
- Thao tác xem status là read-only.
- Phải validate embedding hoàn tất trước khi reset hoặc upsert dữ liệu.

## Retrieval Contract

- Retrieval phải trả evidence thật kèm distance.
- Chỉ evidence đạt threshold mới được đưa vào generation.
- Nếu evidence yếu thì không gọi generation.

## Citation Contract

- Citation phải được dựng từ metadata thật.
- Không tin `source`, `page` hoặc `chunk_id` do LLM tự tạo.
- Kết quả phải có `citations` và `warnings`; code thay label hợp lệ bằng citation thật.

## Security

- Không ghi, log hoặc hiển thị secret.
- Không commit `.env` hay API key thật.

## Testing

- Dùng `unittest`.
- Mock API bên ngoài và dùng thư mục lưu trữ tạm thời.
- Test phải chạy offline, không cần Internet hoặc key thật.

## Coding Style

- Giữ số file, class và function ở mức tối thiểu.
- Không xây dựng kiến trúc phức tạp khi bài học chưa yêu cầu.
- Từ các bước triển khai sau, suy ra đường dẫn từ `Path(__file__).resolve()`;
  không hard-code đường dẫn theo máy.
