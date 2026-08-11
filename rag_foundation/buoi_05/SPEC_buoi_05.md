# SPEC Buổi 05 — OCR PDF tiếng Việt và so sánh chunking

## 1. Mục tiêu và phạm vi

Xây dựng một demo độc lập trong `RAG/rag_foundation/buoi_05/` để:

1. Đọc PDF tiếng Việt trong thư mục `datademo/`.
2. Trích xuất text layer; chỉ dùng OCR cục bộ/fallback khi trang không có text sử dụng được.
3. Chuẩn hóa toàn bộ văn bản đầu ra về Unicode NFC.
4. Tạo và so sánh báo cáo của ba chiến lược chunking: fixed-size, semantic và hierarchical.

Demo ưu tiên mã dễ đọc, từng bước rõ ràng, phù hợp người mới học RAG. Không phức tạp hóa bằng thành phần không cần thiết.

## 2. Đầu vào

- Nguồn dữ liệu: mọi PDF tiếng Việt trong `RAG/rag_foundation/buoi_05/datademo/`.
- Dữ liệu chỉ là PDF công khai hoặc mô phỏng; không dùng dữ liệu nội bộ hay nhạy cảm.
- PDF gốc là dữ liệu chỉ đọc: code không được sửa, đổi tên hoặc ghi đè các file PDF.

## 3. Đầu ra

Mỗi lần xử lý tạo kết quả dưới `RAG/rag_foundation/buoi_05/output/`, gồm:

- Text theo trang đã chuẩn hóa Unicode NFC.
- Metadata tối thiểu cho mỗi trang hoặc đoạn text:
  - `source`: tên file PDF nguồn;
  - `page`: số trang (đánh số từ 1);
  - `ocr_used`: `true` nếu trang đã dùng OCR, ngược lại là `false`;
  - `language`: `vi`.
- Báo cáo so sánh riêng cho cả ba chiến lược chunking. Báo cáo nêu số lượng chunk, độ dài nhỏ nhất/lớn nhất/trung bình và một metadata mẫu của từng chiến lược.

Mỗi chunk cần có tối thiểu `chunk_id`, `strategy`, `source`, `page` hoặc dải trang, và `text`.

## 4. Luồng OCR đơn giản

1. Duyệt PDF từ `datademo/`.
2. Dùng PyMuPDF để lấy text layer của từng trang.
3. Nếu text layer trống hoặc không dùng được, đánh dấu `ocr_used: true` và dùng cơ chế OCR cục bộ đã được cấu hình cho demo.
4. Chuẩn hóa text bằng Unicode NFC trước khi lưu và trước khi chunking.
5. Lưu text, metadata và báo cáo vào `output/`; không sửa PDF nguồn.

Nếu không thể OCR một trang, chương trình phải ghi lỗi/cảnh báo an toàn vào metadata hoặc báo cáo thay vì bịa nội dung.

## 5. Ba chiến lược chunking cần so sánh

### Fixed-size

- Cắt theo số ký tự hoặc token cố định.
- Có overlap giữa hai chunk liên tiếp để giữ ngữ cảnh.
- Báo cáo rõ cấu hình kích thước và overlap đã dùng.

### Semantic

- Ưu tiên ranh giới đoạn văn tự nhiên: ngắt đoạn, kết đoạn và cách dòng.
- Tránh cắt giữa câu khi có thể.
- Khi đoạn quá dài, có thể mới dùng giới hạn độ dài làm phương án phụ.

### Hierarchical

- Nhận diện thứ bậc văn bản pháp quy khi có: `Chương` → `Mục` → `Điều/Khoản` → `Điểm`.
- Mỗi mốc `Chương`, `Mục`, `Điều/Khoản` hoặc `Điểm` được nhận diện phải là điểm bắt đầu của một chunk.
- Metadata chunk cần lưu đường dẫn cấu trúc hiện có, ví dụ chương/mục/điều liên quan.
- Nếu PDF không có cấu trúc rõ ràng, giữ văn bản nguyên trạng theo mức phù hợp và ghi cảnh báo; không tự suy diễn heading.

## 6. Cấu hình `.env` và bảo mật

- File cấu hình nằm tại `RAG/rag_foundation/buoi_05/src/.env`.
- Key trong `.env` được dùng như hợp đồng cấu hình cho môi trường chạy; chỉ được kiểm tra sự tồn tại/tên key khi cần.
- Không được mở, log, in, trả về API, hoặc đưa giá trị key vào exception, metadata, báo cáo hay giao diện.
- Không commit secret vào mã nguồn hay file kết quả.

## 7. Các giới hạn bắt buộc

- Không tạo embedding.
- Không tạo, ghi hoặc truy vấn vector database.
- Không gọi LLM, API LLM hoặc dịch vụ sinh nội dung trong Buổi 5.
- Không sửa PDF gốc.
- Mã mới chỉ nằm trong `RAG/rag_foundation/buoi_05/`.

## 8. Tiêu chí hoàn thành

- Có text tiếng Việt ở dạng Unicode NFC và metadata yêu cầu.
- Có báo cáo đầy đủ cho fixed-size, semantic và hierarchical.
- Báo cáo cho thấy khác biệt về cách cắt, số chunk và độ dài chunk.
- Không có embedding, vector database, cuộc gọi LLM hoặc secret trong đầu ra.
