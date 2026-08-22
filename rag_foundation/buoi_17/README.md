# Buổi 17 — Secure RAG & Compliance

Project tái sử dụng nguyên trạng `buoi_16/src/SecureRetriever` và corpus bảo mật read-only tại `../buoi_16/data/processed/chunks_secure.csv`.

## Chạy

```powershell
cd RAG/rag_foundation/buoi_17
..\buoi_14\.venv\Scripts\python.exe -m pytest tests -q
..\buoi_14\.venv\Scripts\streamlit.exe run app.py
```

Không cần API key để chạy luồng mặc định: câu trả lời extractive chỉ lấy từ context đã qua RBAC. Corpus hiện tại không có tài liệu nội bộ được chứng minh, vì vậy Compliance Gap Checker cố ý trả `CHUA_DU_BANG_CHUNG` và luôn yêu cầu human review.

Encryption chỉ là demo at-rest. Hệ thống thật còn cần TLS, KMS, rotation, backup và IAM.
