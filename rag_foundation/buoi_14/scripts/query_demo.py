import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.pipeline import RetrievalPipeline
from src.graph_hints import get_graph_hints
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--query",required=True);p.add_argument("--method",choices=["bm25","dense","hybrid","hybrid_rerank"],default="hybrid");p.add_argument("--top-k",type=int,default=5);a=p.parse_args(); rows=RetrievalPipeline().retrieve(a.query,a.method,a.top_k)
 for x in rows: print(f"#{x['rank']} {x['chunk_id']} score={x.get('score',x.get('retrieval_score')):.5f}\n{x['citation']}\n{x['text']}\n")
 print("GRAPH HINTS"); hints,status=get_graph_hints(rows);print(status)
 [print(x) for x in (hints or [{"document_id":x["document_id"],"chunk_id":x["chunk_id"]} for x in rows])]
