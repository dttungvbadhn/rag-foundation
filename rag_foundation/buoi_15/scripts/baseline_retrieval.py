import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever

def show(title,rows):
    print(f"\n{title}")
    for x in rows: print(f"{x['rank']:>2} | {x['retrieval_score']:.4f} | {x['chunk_id']} | {x['citation']}\n{x['text'][:240]}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--query",required=True); p.add_argument("--top-k",type=int,default=5); a=p.parse_args()
    bm=BM25Retriever(); de=DenseRetriever(bm.corpus); show("BM25 RESULTS",bm.search(a.query,a.top_k)); show(f"DENSE RESULTS ({de.mode})",de.search(a.query,a.top_k))
