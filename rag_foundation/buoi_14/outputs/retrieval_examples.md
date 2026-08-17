# Retrieval examples

Ba truy vấn đã chạy thật bằng Dense multilingual và Cross-Encoder ngày 2026-08-15.

- Exact/mixed: `Điều 72 hiệu lực thi hành` — Hybrid + neural reranker đưa đúng Điều 72 lên hạng 1.
- Semantic: `Ai chịu trách nhiệm bảo đảm an toàn tiền và tài sản?` — Dense multilingual chạy thật; Hybrid hợp nhất hai bảng rank bằng RRF và giữ citation.
- Mixed: `phạm vi điều chỉnh giao nhận tiền mặt` — Cross-Encoder chỉ rerank tập candidate do Hybrid trả về.

Evaluation nhỏ cho thấy Hybrid + Rerank đạt Hit@5 = 0.667, nhưng ba câu hỏi chưa đủ để kết luận thống kê rằng một cấu hình luôn tốt hơn.
