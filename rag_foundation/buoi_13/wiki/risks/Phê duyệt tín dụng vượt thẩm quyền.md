---
id: "RR-002"
type: "RuiRo"
verification_status: "VERIFIED"
data_origin: "SYNTHETIC"
---

# Phê duyệt tín dụng vượt thẩm quyền

- **ID:** RR-002
- **Mô tả:** Kiểm tra hạn mức phê duyệt không hiệu lực
- **Phân loại:** Rui ro tin dung
- **Nguyên nhân:** Phân quyền trên hệ thống không cập nhật
- **Sự kiện:** Khoản vay được phê duyệt vượt thẩm quyền
- **Tác động:** Tăng nợ xấu và vi phạm quy định
- **Mức rủi ro vốn có:** Cao
- **Mức rủi ro còn lại:** Trung binh
- **Mã đơn vị sở hữu:** DV-CREDIT

## Kiểm soát liên quan

- [[controls/Kiểm tra hạn mức phê duyệt trên hệ thống|Kiểm tra hạn mức phê duyệt trên hệ thống]]
  - relationship_type: `MITIGATES`
  - evidence_quote: Dữ liệu mô phỏng: kiểm tra hạn mức ngăn phê duyệt vượt thẩm quyền
  - verification_status: `VERIFIED`

## Sự kiện rủi ro liên quan

- [[events/Hồ sơ tín dụng được phê duyệt vượt hạn mức của người phê duyệt|Hồ sơ tín dụng được phê duyệt vượt hạn mức của người phê duyệt]]
  - relationship_type: `OBSERVED_AS`
  - evidence_quote: Dữ liệu mô phỏng: sự kiện vượt thẩm quyền
  - verification_status: `VERIFIED`
