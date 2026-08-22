from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from .common import ROOT, citation, load_corpus, tokenize

class DenseRetriever:
    """Multilingual SentenceTransformer when installed; explicit hashing fallback otherwise."""
    def __init__(self, corpus=None, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.corpus=corpus or load_corpus(); self.model_name=model_name; self.model=None; self.mode="HASHING_FALLBACK"
        os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
        try:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(model_name, local_files_only=True); self.mode="NEURAL_DENSE"
        except (ImportError, OSError): pass
        self.embeddings=self._load_or_encode()
    def _load_or_encode(self):
        if self.model is None:
            return self._encode([r["text"] for r in self.corpus])
        import numpy as np
        cache_dir=ROOT/"cache"/"embeddings"; cache_dir.mkdir(parents=True,exist_ok=True)
        signature=hashlib.sha256((self.model_name+"|"+"|".join(r["chunk_id"] for r in self.corpus)).encode()).hexdigest()[:16]
        path=cache_dir/f"documents-{signature}.npy"
        if path.exists(): return np.load(path,mmap_mode="r")
        vectors=self.model.encode([r["text"] for r in self.corpus],normalize_embeddings=True,show_progress_bar=True,batch_size=32)
        np.save(path,vectors); return np.load(path,mmap_mode="r")
    def _hash(self,text,dim=4096):
        counts={}
        toks=tokenize(text)
        for token in toks:
            idx=int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%dim; counts[idx]=counts.get(idx,0.0)+1
        norm=math.sqrt(sum(x*x for x in counts.values())) or 1
        return {i:x/norm for i,x in counts.items()}
    def _encode(self,texts):
        if self.model is not None: return self.model.encode(texts,normalize_embeddings=True)
        return [self._hash(x) for x in texts]
    def search(self,question,top_k=5):
        q=self._encode([question])[0]
        if self.model is not None:
            scored=[(sum(a*b for a,b in zip(q,v)),i) for i,v in enumerate(self.embeddings)]
        else:
            scored=[(sum(value*v.get(key,0.0) for key,value in q.items()),i) for i,v in enumerate(self.embeddings)]
        out=[]
        for rank,(score,i) in enumerate(sorted(scored,reverse=True)[:top_k],1):
            row=dict(self.corpus[i]); row.update(rank=rank,retrieval_score=score,retrieval_method="dense" if self.model else "dense_hashing_fallback",citation=citation(row)); out.append(row)
        return out
