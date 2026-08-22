from __future__ import annotations
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker

class RetrievalPipeline:
    def __init__(self):
        self.hybrid=HybridRetriever(); self.reranker=Reranker()
    def retrieve(self,question,method="hybrid",top_k=5,candidate_k=20):
        if method=="bm25": return self.hybrid.bm25.search(question,top_k)
        if method=="dense": return self.hybrid.dense.search(question,top_k)
        if method=="hybrid": return self.hybrid.search(question,top_k,candidate_k)
        if method=="hybrid_rerank":
            candidates=self.hybrid.search(question,candidate_k,candidate_k)
            return self.reranker.rerank(question,candidates,top_k)
        raise ValueError("method phải là bm25, dense, hybrid hoặc hybrid_rerank")
