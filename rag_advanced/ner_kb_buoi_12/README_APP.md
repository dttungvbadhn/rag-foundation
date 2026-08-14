# Vietnamese Legal Knowledge Graph

Ứng dụng Streamlit để khám phá 30 văn bản pháp luật, entity, evidence và
relationship đã được validate trong Neo4j.

## Chức năng

- Dashboard thống kê node và relationship.
- Đối chiếu số liệu CSV với Neo4j.
- Tìm kiếm theo số hiệu, tiêu đề và toàn văn sạch.
- Xem metadata, entity, confidence và evidence.
- Xem các relationship vào/ra của từng văn bản.
- Trực quan hóa toàn graph hoặc neighborhood 1–3 hop.
- Xem và tải báo cáo validation.

## Cài đặt

Chạy từ thư mục `D:\Rag_2`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\ner_kb\requirements-app.txt
```

Sao chép `.env.example` thành `.env` và điền cấu hình Neo4j. Không commit
`.env`.

## Khởi động

```powershell
streamlit run .\ner_kb\app.py
```

Ứng dụng vẫn đọc và tìm kiếm CSV khi Neo4j không kết nối được; trang khám phá
graph cần kết nối Neo4j.
