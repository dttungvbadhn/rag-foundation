from __future__ import annotations
import csv,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.pipeline import RetrievalPipeline
def main():
 questions=list(csv.DictReader((ROOT/"data/eval/questions.csv").open(encoding="utf-8-sig"))); pipe=RetrievalPipeline(); methods=["bm25","dense","hybrid","hybrid_rerank"]; rows=[]; agg=defaultdict(lambda:defaultdict(int))
 for q in questions:
  for method in methods:
   try: ids=[x["chunk_id"] for x in pipe.retrieve(q["question"],method,5)]; error=""
   except Exception as exc: ids=[]; error=str(exc)
   gold=q["expected_chunk_id"]
   row={"question_id":q["question_id"],"method":method,"expected_chunk_id":gold,"top_5":"|".join(ids),"error":error}
   for k in (1,3,5): row[f"hit_at_{k}"]=int(gold in ids[:k]) if gold else ""; agg[method][k]+=int(gold in ids[:k]) if gold else 0
   rows.append(row)
 out=ROOT/"outputs/retrieval_comparison.csv";out.parent.mkdir(exist_ok=True); fields=list(rows[0]);
 with out.open("w",encoding="utf-8-sig",newline="") as f: w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 n=sum(bool(q["expected_chunk_id"]) for q in questions); report=["# Evaluation report","",f"Evaluated questions with verified gold: {n}",""]
 for m in methods: report += [f"## {m}","",*(f"- Hit@{k}: {agg[m][k]/n:.3f}" for k in (1,3,5)),""]
 report += ["## Runtime modes","",f"- Dense: {pipe.hybrid.dense.mode}",f"- Reranker: {pipe.reranker.mode}","",
            "## Limitations","","This is a small three-question verified set; it is suitable for a lab sanity check, not a statistically strong benchmark. Failed queries are retained in the CSV."]
 (ROOT/"outputs/evaluation_report.md").write_text("\n".join(report),encoding="utf-8")
if __name__=="__main__":main()
