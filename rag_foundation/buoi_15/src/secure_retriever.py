from __future__ import annotations
import csv,os
from pathlib import Path
from .common import ROOT,read_csv
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .reranker import Reranker
from .security import can_access,parse_roles,validate_user_roles

class SecureRetriever:
    def __init__(self):
        path=ROOT/"data/processed/chunks_secure.csv"
        if not path.exists(): raise FileNotFoundError("Chạy scripts/assign_security_tags.py trước")
        self.corpus=read_csv(path)
        self.bm25=BM25Retriever(self.corpus); self.dense=DenseRetriever(self.corpus); self.reranker=Reranker()

    def _allowed(self,row,user_roles): return can_access(row.get("allowed_roles"),user_roles)
    def _normalize(self,rows,method):
        for rank,row in enumerate(rows,1):
            row["rank"]=rank;row["retrieval_method"]=method;row["allowed_roles"]=parse_roles(row.get("allowed_roles"))
            row["score"]=row.get("score",row.get("retrieval_score",row.get("rrf_score",row.get("rerank_score",0))))
        return rows

    def bm25_search(self,query,user_roles,top_k=5):
        roles=validate_user_roles(user_roles); ranked=self.bm25.search(query,len(self.corpus)); allowed=[x for x in ranked if self._allowed(x,roles)]
        return self._normalize(allowed[:top_k],"secure_bm25")

    def dense_search(self,query,user_roles,top_k=5):
        roles=validate_user_roles(user_roles); ranked=self.dense.search(query,len(self.corpus)); allowed=[x for x in ranked if self._allowed(x,roles)]
        return self._normalize(allowed[:top_k],"secure_dense")

    def hybrid_search(self,query,user_roles,top_k=5,candidate_k=20):
        roles=validate_user_roles(user_roles)
        bm=[x for x in self.bm25.search(query,len(self.corpus)) if self._allowed(x,roles)][:candidate_k]
        de=[x for x in self.dense.search(query,len(self.corpus)) if self._allowed(x,roles)][:candidate_k]
        merged={}
        for method,items in (("bm25",bm),("dense",de)):
            for rank,item in enumerate(items,1):
                row=merged.setdefault(item["chunk_id"],dict(item));row[f"{method}_rank"]=rank;row["rrf_score"]=row.get("rrf_score",0)+1/(60+rank)
        ranked=sorted(merged.values(),key=lambda x:(-x["rrf_score"],x["chunk_id"]))[:top_k]
        return self._normalize(ranked,"secure_hybrid")

    def retrieve(self,query,user_roles,method="hybrid",top_k=5,candidate_k=20):
        validate_user_roles(user_roles)
        if method=="bm25": return self.bm25_search(query,user_roles,top_k)
        if method=="dense": return self.dense_search(query,user_roles,top_k)
        if method=="hybrid": return self.hybrid_search(query,user_roles,top_k,candidate_k)
        if method=="hybrid_rerank":
            candidates=self.hybrid_search(query,user_roles,candidate_k,candidate_k)
            # Security invariant: every candidate is checked again before external model processing.
            safe=[x for x in candidates if self._allowed(x,user_roles)]
            if len(safe)!=len(candidates): raise AssertionError("Unauthorized candidate reached reranker boundary")
            return self._normalize(self.reranker.rerank(query,safe,top_k),"secure_hybrid_rerank")
        raise ValueError("method không hợp lệ")

    def filtered_count(self,user_roles):
        roles=validate_user_roles(user_roles); return sum(not self._allowed(x,roles) for x in self.corpus)

    def graph_search(self,user_roles,limit=20):
        roles=validate_user_roles(user_roles)
        try:
            from dotenv import load_dotenv
            from neo4j import GraphDatabase
            for p in (ROOT/".env",ROOT.parents[2]/".env"):
                if p.exists(): load_dotenv(p,override=False)
            with GraphDatabase.driver(os.environ["NEO4J_URI"],auth=(os.getenv("NEO4J_USER","neo4j"),os.environ["NEO4J_PASSWORD"])) as driver:
                with driver.session(database=os.getenv("NEO4J_DATABASE","neo4j")) as s:
                    return s.run("MATCH (v:VanBan {lab_session:'buoi_15'})-[:CONTAINS]->(d:DieuKhoan {lab_session:'buoi_15'}) WHERE any(role IN coalesce(d.allowed_roles,v.allowed_roles,[]) WHERE role IN $user_roles) RETURN v.id AS document_id,d.id AS chunk_id,d.article AS article,d.allowed_roles AS allowed_roles LIMIT $limit",user_roles=roles,limit=limit).data()
        except Exception: return []
