---
id: "RR-001"
type: "RuiRo"
verification_status: "VERIFIED"
data_origin: "SYNTHETIC"
---

# Giao dịch chuyển tiền bị hạch toán sai

- **ID:** RR-001
- **Mô tả:** Đối soát giao dịch cuối ngày không đầy đủ
- **Phân loại:** Rui ro van hanh
- **Nguyên nhân:** Thiếu đối chiếu giữa hệ thống thanh toán và sổ cái
- **Sự kiện:** Giao dịch được ghi nhận sai trạng thái
- **Tác động:** Tổn thất tài chính và khiếu nại khách hàng
- **Mức rủi ro vốn có:** Cao
- **Mức rủi ro còn lại:** Trung binh
- **Mã đơn vị sở hữu:** DV-OPS

## Kiểm soát liên quan

- [[controls/Đối soát tự động giao dịch và sổ cái|Đối soát tự động giao dịch và sổ cái]]
  - relationship_type: `MITIGATES`
  - evidence_quote: Dữ liệu mô phỏng: đối soát tự động giảm nguy cơ hạch toán sai
  - verification_status: `VERIFIED`

## Sự kiện rủi ro liên quan

- [[events/Sai lệch trạng thái giao dịch được phát hiện khi đối soát cuối ngày|Sai lệch trạng thái giao dịch được phát hiện khi đối soát cuối ngày]]
  - relationship_type: `OBSERVED_AS`
  - evidence_quote: Dữ liệu mô phỏng: sự kiện đối soát giao dịch
  - verification_status: `VERIFIED`
