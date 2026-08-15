---
id: "RR-004"
type: "RuiRo"
verification_status: "VERIFIED"
data_origin: "SYNTHETIC"
---

# Lộ thông tin khách hàng

- **ID:** RR-004
- **Mô tả:** Quyền truy cập dữ liệu không được kiểm soát phù hợp
- **Phân loại:** Rui ro cong nghe thong tin
- **Nguyên nhân:** Cấp quyền vượt nhu cầu công việc
- **Sự kiện:** Dữ liệu khách hàng bị truy cập hoặc chia sẻ trái phép
- **Tác động:** Vi phạm bảo mật và tổn hại uy tín
- **Mức rủi ro vốn có:** Cao
- **Mức rủi ro còn lại:** Trung binh
- **Mã đơn vị sở hữu:** DV-IT

## Kiểm soát liên quan

- [[controls/Rà soát quyền truy cập định kỳ|Rà soát quyền truy cập định kỳ]]
  - relationship_type: `MITIGATES`
  - evidence_quote: Dữ liệu mô phỏng: rà soát quyền hạn giảm lộ dữ liệu
  - verification_status: `VERIFIED`

## Sự kiện rủi ro liên quan

- [[events/Tài khoản có quyền truy cập dữ liệu vượt phạm vi công việc|Tài khoản có quyền truy cập dữ liệu vượt phạm vi công việc]]
  - relationship_type: `OBSERVED_AS`
  - evidence_quote: Dữ liệu mô phỏng: sự kiện quyền truy cập quá mức
  - verification_status: `VERIFIED`
