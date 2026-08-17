from __future__ import annotations
import os
import math
from .common import ROOT
from .common import tokenize

class Reranker:
    def __init__(self, model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.model=None; self.model_name=model_name; self.mode="LEXICAL_FALLBACK"
        os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
        try:
            from sentence_transformers import CrossEncoder
            self.model=CrossEncoder(model_name, local_files_only=True); self.mode="NEURAL_CROSS_ENCODER"
        except (ImportError,OSError): pass
    def rerank(self,question,candidates,top_k=5):
        if self.model:
            scores=self.model.predict([(question,x["text"]) for x in candidates]).tolist()
        else:
            q=set(tokenize(question)); scores=[len(q & set(tokenize(x["text"])))/max(1,len(q)) for x in candidates]
        items=[]
        for candidate,score in zip(candidates,scores):
            raw=float(score)
            display_score=1/(1+math.exp(-max(-60.0,min(60.0,raw)))) if self.model else raw
            row=dict(candidate); row.update(hybrid_rank=candidate.get("rank",candidate.get("final_rank")),hybrid_score=candidate["rrf_score"],rerank_logit=raw,rerank_score=display_score); items.append(row)
        items=sorted(items,key=lambda x:(-x["rerank_logit"],-x["hybrid_score"]))[:top_k]
        for rank,row in enumerate(items,1): row.update(rank=rank,final_rank=rank,retrieval_method="hybrid_rerank" if self.model else "hybrid_rerank_lexical_fallback",score=row["rerank_score"])
        return items
