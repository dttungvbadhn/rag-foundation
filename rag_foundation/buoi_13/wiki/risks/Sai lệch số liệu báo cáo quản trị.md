---
id: "RR-010"
type: "RuiRo"
verification_status: "VERIFIED"
data_origin: "SYNTHETIC"
---

# Sai lệch số liệu báo cáo quản trị

- **ID:** RR-010
- **Mô tả:** Dữ liệu nguồn không được đối chiếu
- **Phân loại:** Rui ro bao cao
- **Nguyên nhân:** Thay đổi dữ liệu không có kiểm soát
- **Sự kiện:** Báo cáo quản trị có số liệu sai
- **Tác động:** Quyết định quản trị sai lệch
- **Mức rủi ro vốn có:** Trung binh
- **Mức rủi ro còn lại:** Thap
- **Mã đơn vị sở hữu:** DV-FINANCE

## Kiểm soát liên quan

- [[controls/Đối chiếu dữ liệu nguồn trước khi phát hành báo cáo|Đối chiếu dữ liệu nguồn trước khi phát hành báo cáo]]
  - relationship_type: `MITIGATES`
  - evidence_quote: Dữ liệu mô phỏng: đối chiếu nguồn giảm sai lệch báo cáo
  - verification_status: `VERIFIED`

## Sự kiện rủi ro liên quan

- [[events/Báo cáo quản trị sử dụng dữ liệu nguồn chưa đối chiếu|Báo cáo quản trị sử dụng dữ liệu nguồn chưa đối chiếu]]
  - relationship_type: `OBSERVED_AS`
  - evidence_quote: Dữ liệu mô phỏng: sự kiện sai lệch báo cáo
  - verification_status: `VERIFIED`
