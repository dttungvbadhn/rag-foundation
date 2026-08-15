---
id: "RR-006"
type: "RuiRo"
verification_status: "VERIFIED"
data_origin: "SYNTHETIC"
---

# Gian lận giả mạo yêu cầu chuyển tiền

- **ID:** RR-006
- **Mô tả:** Nhận diện và xác thực yêu cầu chưa đủ mạnh
- **Phân loại:** Rui ro gian lan
- **Nguyên nhân:** Nhân viên không xác minh kênh liên lạc
- **Sự kiện:** Yêu cầu chuyển tiền giả mạo được xử lý
- **Tác động:** Tổn thất tài chính
- **Mức rủi ro vốn có:** Cao
- **Mức rủi ro còn lại:** Trung binh
- **Mã đơn vị sở hữu:** DV-OPS

## Kiểm soát liên quan

- [[controls/Xác thực hai kênh với lệnh chuyển tiền ngoại lệ|Xác thực hai kênh với lệnh chuyển tiền ngoại lệ]]
  - relationship_type: `MITIGATES`
  - evidence_quote: Dữ liệu mô phỏng: xác thực hai kênh giảm gian lận chuyển tiền
  - verification_status: `VERIFIED`

## Sự kiện rủi ro liên quan

- [[events/Yêu cầu chuyển tiền giả mạo được xử lý trước khi bị thu hồi|Yêu cầu chuyển tiền giả mạo được xử lý trước khi bị thu hồi]]
  - relationship_type: `OBSERVED_AS`
  - evidence_quote: Dữ liệu mô phỏng: sự kiện giả mạo chuyển tiền
  - verification_status: `VERIFIED`
