from __future__ import annotations

from .audit_logger import log_event
from .secure_retrieval_adapter import SecureRetrievalAdapter

NO_INFO = "Không tìm thấy đủ thông tin trong phạm vi tài liệu được phép truy cập."


def lookup(question: str, user_role: str, top_k: int = 5, user_id: str = "demo01", adapter=None) -> dict:
    adapter = adapter or SecureRetrievalAdapter()
    method = "hybrid_rerank"
    try:
        payload = adapter.retrieve(question, user_role, top_k, method)
        results = payload["results"]
        # Extractive answer is deliberately bounded to authorized context and works without an API key.
        answer = results[0]["text"].strip() if results and results[0]["text"].strip() else NO_INFO
        status = "SUCCESS" if results else "DENIED"
        request_id = log_event(user_id_demo=user_id, user_role=user_role, action="INTERNAL_LOOKUP",
                               query=question, retrieval_method=method, results=results,
                               filtered_count=payload["filtered_count"], status=status)
        return {"answer": answer, "citations": [r["citation"] for r in results],
                "results": results, "access_scope": user_role,
                "access_decision": payload["access_decision"], "request_id": request_id}
    except Exception as exc:
        status = "DENIED" if isinstance(exc, PermissionError) else "ERROR"
        request_id = log_event(user_id_demo=user_id, user_role=user_role, action="INTERNAL_LOOKUP",
                               query=question, retrieval_method=method, results=[], filtered_count=0,
                               status=status, error=str(exc))
        return {"answer": NO_INFO, "citations": [], "results": [], "access_scope": user_role,
                "access_decision": "DENY", "request_id": request_id, "error": str(exc)}

