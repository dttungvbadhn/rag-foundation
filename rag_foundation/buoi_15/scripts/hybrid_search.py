import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.hybrid_retriever import HybridRetriever
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--query",required=True);p.add_argument("--top-k",type=int,default=5);p.add_argument("--candidate-k",type=int,default=20);a=p.parse_args()
 print("HYBRID RESULTS\nRank | Chunk | BM25 rank | Dense rank | RRF | Citation")
 for x in HybridRetriever().search(a.query,a.top_k,a.candidate_k): print(f"{x['rank']} | {x['chunk_id']} | {x.get('bm25_rank','-')} | {x.get('dense_rank','-')} | {x['rrf_score']:.6f} | {x['citation']}")
