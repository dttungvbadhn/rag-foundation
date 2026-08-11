# Advanced RAG Specification — Buổi 08

## 1. Workspace và Security

- Chỉ ghi trong `rag_foundation/buoi_08/`; không sửa Buổi 05–07.
- Suy ra mọi đường dẫn từ `Path(__file__).resolve()`, không hard-code đường dẫn máy.
- Không log, hiển thị hoặc commit secret. `.env` và `storage/chroma/` phải bị ignore.
- Không gửi dữ liệu tới dịch vụ ngoài nếu chưa được người vận hành cho phép.

## 2. Quan hệ với Buổi 05 và Buổi 07

- JSON chunk do Buổi 05 chuẩn bị là nguồn dữ liệu thật; không OCR hoặc chunk lại.
- `rag.py` là bản sao semantic baseline từ Buổi 07 và chạy độc lập trong Buổi 08.
- Không import runtime từ Buổi 07, không dùng `.env` hoặc storage Buổi 07.
- Advanced pipeline phải tái sử dụng contract baseline, không tạo semantic pipeline lệch chuẩn.

## 3. Data Contract

Mỗi chunk bắt buộc có `chunk_id`, `strategy`, `source`, `page_start`, `page_end`,
`text`. Kiểu dữ liệu, trang, strategy, text rỗng và duplicate ID tuân thủ validator
baseline. Không sửa object nguồn tại chỗ.

## 4. BM25 Tokenizer/Retrieval Contract

- Tokenizer tiếng Việt phải deterministic, có quy tắc normalize được ghi rõ và test.
- Không làm mất số Điều, Khoản hoặc ký hiệu có ý nghĩa pháp lý.
- BM25 chỉ index chunk thuộc strategy được chọn và trả `chunk_id`, score, rank.
- Tham số `k1`, `b`, số candidate phải cấu hình và validate; score cao hơn tốt hơn.
- Index/retrieval không phụ thuộc Internet và cho kết quả ổn định cùng input.

## 5. Semantic Candidate Contract

- Dùng collection identity, model và dimension từ semantic baseline.
- Query embedding phải qua cùng validator vector và dùng `embedding_function=None`.
- Candidate phải có document, metadata thật, cosine distance và semantic rank.
- Không trộn collection/strategy hoặc tự tạo metadata còn thiếu.

## 6. RRF Fusion Contract

- Hợp nhất BM25 và semantic candidate theo `chunk_id` bằng Reciprocal Rank Fusion.
- Công thức và hằng số `k` phải minh bạch, cấu hình được và deterministic.
- Chunk xuất hiện ở một hoặc cả hai nguồn đều được giữ với trace rank/đóng góp.
- Sắp xếp tie ổn định; không cộng trực tiếp BM25 score với cosine distance.

## 7. Cross-Encoder Reranker Contract

- Chỉ rerank tập candidate sau fusion, không quét toàn corpus.
- Model/version, device và batch size phải được báo rõ; score cao hơn tốt hơn.
- Không tải model khi import module hoặc chạy status.
- Cho phép inject/mock scorer trong test; không có random/hash fallback runtime.
- Lỗi model phải fail rõ hoặc trả retrieval trước rerank theo policy được khai báo.

## 8. Final Evidence và Citation Contract

- Final evidence giữ text, metadata thật, rank/score/trace và trạng thái accepted.
- Citation chỉ dựng bằng code từ `source`, trang và `chunk_id` trong metadata.
- Không tin source/trang/chunk ID do LLM tự tạo; label giả bị loại kèm warning.
- Evidence yếu không được đưa vào generation nhưng vẫn có thể hiển thị để audit.

## 9. Pipeline Trace Contract

Mỗi query phải có trace tối thiểu: query ID, strategy, cấu hình candidate, BM25
rank/score, semantic rank/distance, RRF contribution/score, reranker score/rank,
final rank, gate decision, timings và warnings. Trace không chứa secret hoặc raw key.

## 10. Evaluation Metrics Contract

- Đọc schema eval đã validate và tách rõ query in-scope/out-of-scope.
- Báo Recall@k, MRR, nDCG@k và tỷ lệ gate cho từng pipeline/configuration.
- Không tính label `needs_human_review=true` là gold chuyên gia đã duyệt.
- Report phải ghi dataset/version, model, dimension, threshold và thời điểm chạy.
- Không chọn threshold bằng chính tập dùng để báo kết quả cuối mà không cảnh báo bias.

## 11. Offline Testing Contract

- Dùng `unittest`, mock API/model và temporary storage; không cần key hoặc Internet.
- Test tokenizer, BM25 ranking, semantic candidate contract, RRF, tie ordering,
  reranker injection, trace, metrics, gate và citation.
- Fixture mô phỏng không chứa dữ liệu nhạy cảm; test độc lập và cleanup tự động.

## 12. UI Comparison Contract

- UI so sánh semantic baseline với Advanced RAG trên cùng query/strategy/top-k.
- Hiển thị riêng candidate/rank/distance/score, final evidence, citation và warnings.
- Giải thích BM25 score, cosine distance, RRF và reranker score không phải xác suất.
- Không tự index, tải model hoặc gọi API khi mở UI.
- Không gọi API khi thiếu key, query rỗng hoặc collection chưa sẵn sàng.
- Session state tối thiểu; không thêm login, analytics hoặc dashboard ngoài phạm vi.
