from __future__ import annotations
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever

class HybridRetriever:
    def __init__(self, corpus=None, rrf_k=60):
        self.bm25=BM25Retriever(corpus); self.dense=DenseRetriever(self.bm25.corpus); self.rrf_k=rrf_k
    def search(self,question,top_k=5,candidate_k=20):
        bm=self.bm25.search(question,candidate_k); de=self.dense.search(question,candidate_k); merged={}
        for method,items in (("bm25",bm),("dense",de)):
            for item in items:
                cid=item["chunk_id"]; row=merged.setdefault(cid,dict(item)); row[f"{method}_rank"]=item["rank"]
                row["rrf_score"]=row.get("rrf_score",0)+1/(self.rrf_k+item["rank"])
        ranked=sorted(merged.values(),key=lambda x:(-x["rrf_score"],x["chunk_id"]))[:top_k]
        for rank,row in enumerate(ranked,1): row.update(final_rank=rank,rank=rank,retrieval_method="hybrid",score=row["rrf_score"])
        return ranked
