from __future__ import annotations

from .bm25_retriever import BM25Retriever
from .common import ROOT, read_csv
from .dense_retriever import DenseRetriever
from .reranker import Reranker
from .security import can_access, parse_roles, validate_user_roles


class SecureRetriever:
    """Security-filtered retrieval; filtering happens before fusion/reranking."""

    def __init__(self):
        path = ROOT / "data" / "processed" / "chunks_secure.csv"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy secure corpus: {path}")
        self.corpus = read_csv(path)
        self.bm25 = BM25Retriever(self.corpus)
        self.dense = DenseRetriever(self.corpus)
        self.reranker = Reranker()

    @staticmethod
    def _allowed(row, user_roles):
        return can_access(row.get("allowed_roles"), user_roles)

    @staticmethod
    def _normalize(rows, method):
        normalized = []
        for rank, item in enumerate(rows, 1):
            row = dict(item)
            row["rank"] = rank
            row["retrieval_method"] = method
            row["allowed_roles"] = parse_roles(row.get("allowed_roles"))
            row["score"] = row.get("score", row.get("retrieval_score", row.get("rrf_score", 0)))
            normalized.append(row)
        return normalized

    def retrieve(self, query, user_roles, method="hybrid", top_k=5, candidate_k=20):
        roles = validate_user_roles(user_roles)
        if method == "bm25":
            ranked = self.bm25.search(query, len(self.corpus))
            return self._normalize([x for x in ranked if self._allowed(x, roles)][:top_k], "secure_bm25")
        if method == "dense":
            ranked = self.dense.search(query, len(self.corpus))
            return self._normalize([x for x in ranked if self._allowed(x, roles)][:top_k], "secure_dense")
        if method not in {"hybrid", "hybrid_rerank"}:
            raise ValueError("method phải là bm25, dense, hybrid hoặc hybrid_rerank")

        bm25 = [x for x in self.bm25.search(query, len(self.corpus)) if self._allowed(x, roles)][:candidate_k]
        dense = [x for x in self.dense.search(query, len(self.corpus)) if self._allowed(x, roles)][:candidate_k]
        merged = {}
        for source, items in (("bm25", bm25), ("dense", dense)):
            for rank, item in enumerate(items, 1):
                row = merged.setdefault(item["chunk_id"], dict(item))
                row[f"{source}_rank"] = rank
                row["rrf_score"] = row.get("rrf_score", 0.0) + 1 / (60 + rank)
        candidates = sorted(merged.values(), key=lambda x: (-x["rrf_score"], x["chunk_id"]))
        if method == "hybrid_rerank":
            safe = [x for x in candidates[:candidate_k] if self._allowed(x, roles)]
            return self._normalize(self.reranker.rerank(query, safe, top_k), "secure_hybrid_rerank")
        return self._normalize(candidates[:top_k], "secure_hybrid")
