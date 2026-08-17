# Buổi 14 — Hybrid Search, Reranking và Mini KG

Nguồn được tìm có kiểm soát cạnh `rag_foundation` trước, sau đó tại workspace gốc. Workspace hiện tại không có `kb+hops`, vì vậy lần chạy đã xác minh dùng bộ ba CSV tại `../../../ner_kb`; các file nguồn không bị thay đổi.

## Thiết lập và chạy tuần tự

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/inspect_project.py
python scripts/prepare_corpus.py
python scripts/baseline_retrieval.py --query "Điều 72 hiệu lực thi hành" --top-k 5
python scripts/hybrid_search.py --query "an toàn tiền và tài sản" --candidate-k 20 --top-k 5
python scripts/rerank.py --query "phạm vi điều chỉnh giao nhận tiền mặt" --candidate-k 20 --top-k 5
python scripts/compare_retrieval.py
python scripts/load_mini_kg.py
python scripts/query_demo.py --query "Điều 72 hiệu lực thi hành" --method hybrid_rerank --top-k 5
streamlit run app.py
```

Trên Windows có thể chạy ổn định bằng:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

Giữ terminal này mở trong lúc sử dụng. App cố định tại `http://localhost:8502`; file watcher được tắt vì pipeline dùng Transformers/PyTorch và không cần hot reload.

Dừng Streamlit bằng `Ctrl+C`. Giao diện có bốn method, Top-k, citation, các rank BM25/Dense/RRF và bảng trước/sau rerank. Graph hints vẫn hiện ID khi Neo4j không sẵn sàng.

## Model và fallback

Pipeline dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` cho Dense và `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` cho reranking. Nếu model không thể tải, Dense mang nhãn `dense_hashing_fallback` và reranker mang nhãn `hybrid_rerank_lexical_fallback`; các fallback này không được xem là neural.

## Neo4j

Sao chép `.env.example` thành `.env` và điền credentials. Loader chỉ `MERGE`, gắn `lab_session='buoi_14'`, không có lệnh xóa toàn graph. Các quan hệ nguồn dị thể được lưu dưới `SOURCE_RELATION` với `type`, provenance và evidence nguyên gốc.
