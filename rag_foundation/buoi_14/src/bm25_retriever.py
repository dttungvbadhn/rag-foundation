from __future__ import annotations
import math
from collections import Counter
from .common import citation, load_corpus, tokenize

class BM25Retriever:
    def __init__(self, corpus=None, k1=1.5, b=.75):
        self.corpus = corpus or load_corpus(); self.k1=k1; self.b=b
        self.tokens=[tokenize(r["text"]+" "+r.get("title", "")) for r in self.corpus]
        self.freq=[Counter(x) for x in self.tokens]; self.avgdl=sum(map(len,self.tokens))/max(1,len(self.tokens))
        self.df=Counter(t for doc in self.tokens for t in set(doc)); self.n=len(self.corpus)
    def search(self, question, top_k=5):
        scores=[]
        for i,(tokens,tf) in enumerate(zip(self.tokens,self.freq)):
            score=0.0
            for term in tokenize(question):
                if not tf[term]: continue
                idf=math.log(1+(self.n-self.df[term]+.5)/(self.df[term]+.5))
                score += idf*tf[term]*(self.k1+1)/(tf[term]+self.k1*(1-self.b+self.b*len(tokens)/self.avgdl))
            scores.append((score,i))
        out=[]
        for rank,(score,i) in enumerate(sorted(scores,reverse=True)[:top_k],1):
            row=dict(self.corpus[i]); row.update(rank=rank,retrieval_score=score,retrieval_method="bm25",citation=citation(row)); out.append(row)
        return out
