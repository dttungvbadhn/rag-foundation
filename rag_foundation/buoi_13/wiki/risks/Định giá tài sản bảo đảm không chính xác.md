---
id: "RR-008"
type: "RuiRo"
verification_status: "VERIFIED"
data_origin: "SYNTHETIC"
---

# Định giá tài sản bảo đảm không chính xác

- **ID:** RR-008
- **Mô tả:** Dữ liệu định giá không độc lập hoặc hết hạn
- **Phân loại:** Rui ro tin dung
- **Nguyên nhân:** Thiếu rà soát lại giá trị tài sản
- **Sự kiện:** Tài sản bảo đảm được định giá cao hơn thực tế
- **Tác động:** Tăng tổn thất khi xử lý nợ
- **Mức rủi ro vốn có:** Cao
- **Mức rủi ro còn lại:** Trung binh
- **Mã đơn vị sở hữu:** DV-CREDIT

## Kiểm soát liên quan

- [[controls/Rà soát độc lập định giá tài sản bảo đảm|Rà soát độc lập định giá tài sản bảo đảm]]
  - relationship_type: `MITIGATES`
  - evidence_quote: Dữ liệu mô phỏng: rà soát độc lập giảm sai định giá
  - verification_status: `VERIFIED`

## Sự kiện rủi ro liên quan

- [[events/Rà soát phát hiện giá trị tài sản bảo đảm đã hết hiệu lực|Rà soát phát hiện giá trị tài sản bảo đảm đã hết hiệu lực]]
  - relationship_type: `OBSERVED_AS`
  - evidence_quote: Dữ liệu mô phỏng: sự kiện sai định giá tài sản
  - verification_status: `VERIFIED`
