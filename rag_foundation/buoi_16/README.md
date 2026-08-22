# Buổi 16 — Đánh giá hệ thống RAG bằng Ragas

Thư mục độc lập cho bài thực hành Buổi 16. Pipeline dùng `SecureRetriever`, tạo Golden Dataset, sinh câu trả lời RAG, đánh giá bốn metric Ragas và xuất báo cáo.

## Cấu trúc chính

- `scripts/evaluate_rag_pipeline.py`: pipeline đánh giá một lệnh.
- `src/secure_retriever.py`: retrieval có RBAC trước fusion/reranking.
- `data/processed/`: corpus normalized và secure.
- `data/eval/`: Golden Dataset, checkpoint và kết quả đánh giá.
- `outputs/ragas_evaluation_report.md`: báo cáo Ragas.

## Thiết lập và chạy

```powershell
cd D:\Rag_2\RAG\rag_foundation\buoi_16
Copy-Item .env.example .env
# Điền HF_TOKEN vào .env, không commit file này.
python -m pip install -r requirements.txt
python scripts/evaluate_rag_pipeline.py --reuse-qa
```

Smoke test một mẫu:

```powershell
python scripts/evaluate_rag_pipeline.py --reuse-qa --sample-limit 1
```

Generator là `Qwen/Qwen3.5-9B:deepinfra`; evaluator độc lập là `openai/gpt-oss-20b:deepinfra` qua Hugging Face Router.

