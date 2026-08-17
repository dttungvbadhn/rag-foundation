import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--query",required=True);p.add_argument("--top-k",type=int,default=5);p.add_argument("--candidate-k",type=int,default=20);a=p.parse_args(); h=HybridRetriever(); before=h.search(a.query,a.candidate_k,a.candidate_k); rr=Reranker(); after=rr.rerank(a.query,before,a.top_k)
 print("BEFORE RERANK");[print(x['rank'],x['chunk_id'],x['citation']) for x in before[:a.top_k]]
 print(f"AFTER RERANK ({rr.mode})");[print(x['rank'],x['chunk_id'],f"{x['rerank_score']:.4f}",x['citation']) for x in after]
