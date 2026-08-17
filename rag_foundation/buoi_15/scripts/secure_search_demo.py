import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.secure_retriever import SecureRetriever
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--query",required=True);p.add_argument("--roles",nargs="+",required=True);p.add_argument("--method",choices=["bm25","dense","hybrid","hybrid_rerank"],default="hybrid");p.add_argument("--top-k",type=int,default=5);p.add_argument("--candidate-k",type=int,default=20);a=p.parse_args();r=SecureRetriever();rows=r.retrieve(a.query,a.roles,a.method,a.top_k,a.candidate_k)
 print(f"SECURE RESULTS | roles={a.roles} | filtered={r.filtered_count(a.roles)}")
 for x in rows: print(f"#{x['rank']} {x['chunk_id']} score={x['score']:.4f} allowed={x['allowed_roles']}\n{x['citation']}\n{x['text'][:300]}\n")
