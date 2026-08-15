# Wiki Risk Graph

MVP chuyển 4 CSV mô phỏng thành graph `KiemSoat -> RuiRo -> SuKienRuiRo`, Wiki Markdown cho Obsidian và dữ liệu có thể nạp vào Neo4j. Pipeline không sửa CSV nguồn, không tự sinh quan hệ và giữ nguyên bằng chứng/trạng thái xác minh.

## Yêu cầu

- Python 3.10 trở lên.
- Obsidian (tùy chọn, để xem Wiki/Graph View).
- Neo4j (tùy chọn, chỉ cần cho bước nạp graph).

## Chạy pipeline

Từ thư mục gốc dự án:

```powershell
python scripts/inspect_data.py
python scripts/build_entities.py
python scripts/build_wiki.py
python scripts/validate_wiki.py
```

Sau đó mở thư mục `wiki/` bằng **Open folder as vault** trong Obsidian và mở `Home.md`.

## Chạy giao diện Streamlit

Cài thư viện và khởi động ứng dụng:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Ứng dụng gồm đồ thị có bộ lọc, trang tra cứu hồ sơ và bằng chứng quan hệ, kiểm tra khoảng trống dữ liệu, bảng dữ liệu và nút tải CSV.

## Kiểm thử tự động

Chạy toàn bộ pipeline và 9 kiểm thử về schema, số lượng node/edge, tham chiếu, hướng quan hệ, Wiki, broken link, validation, tính idempotent và tài nguyên Neo4j:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Kiểm thử kết nối/nạp Neo4j thật chỉ thực hiện sau khi server Neo4j đang chạy và `.env` đã được cấu hình.

## Nạp Neo4j (tùy chọn)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Sửa `.env` bằng thông tin Neo4j thật, sau đó chạy:

```powershell
cypher-shell -f cypher/schema.cypher
python scripts/load_neo4j.py
```

Các truy vấn minh họa nằm trong `cypher/demo_queries.cypher`. Loader dùng `MERGE`, truy vấn có tham số, allowlist label/relationship và không chứa mật khẩu trong mã nguồn.

## Đầu ra

- `outputs/entities.csv`, `outputs/relations.csv`: node và edge chuẩn hóa.
- `wiki/`: vault Markdown với liên kết chéo Obsidian.
- `outputs/wiki_validation_report.md`: broken link, duplicate, orphan và khoảng trống dữ liệu.
- `cypher/`: constraint và demo query Neo4j.

`owner_unit_id` và `owner_role_id` chỉ được hiển thị dưới dạng mã vì dữ liệu nguồn chưa có bảng master tên đơn vị/vai trò.
