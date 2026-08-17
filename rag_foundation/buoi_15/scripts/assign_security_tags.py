from __future__ import annotations
import csv,json,sys,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import corpus_path,read_csv
from src.security import VALID_ROLES

HR=("nhân sự","lương","tiền lương","tuyển dụng","bổ nhiệm","kỷ luật lao động","người lao động")
RISK=("tín dụng","rủi ro","hạn mức","phê duyệt","khoản vay","cấp tín dụng","nợ xấu","thu hồi nợ")

def classify_text(text):
 text=text.lower()
 if any(k in text for k in HR): return ["Admin","HR"]
 if any(k in text for k in RISK): return ["Admin","Risk_Manager","Staff"]
 return list(VALID_ROLES)

def main():
 rows=read_csv(corpus_path()); counts=Counter();document_text=defaultdict(list)
 for row in rows: document_text[row["document_id"]].append(" ".join((row.get("title",""),row.get("text",""))))
 document_roles={doc_id:classify_text(" ".join(parts)) for doc_id,parts in document_text.items()}
 for row in rows:
  roles=document_roles[row["document_id"]]; row["allowed_roles"]=json.dumps(roles,ensure_ascii=False); counts[tuple(roles)]+=1
 if any(not json.loads(r["allowed_roles"]) for r in rows): raise AssertionError("Có chunk thiếu allowed_roles")
 out=ROOT/"data/processed/chunks_secure.csv"; fields=list(rows[0])
 with out.open("w",encoding="utf-8-sig",newline="") as f: w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 print("SECURITY TAGGING")
 for roles,count in counts.items(): print(f"{list(roles)}: {count}")
 for roles in counts:
  sample=next(r for r in rows if tuple(json.loads(r["allowed_roles"]))==roles); print({k:sample[k] for k in ("chunk_id","document_id","article","allowed_roles")})
 print(f"Rows: {len(rows)} | Missing roles: 0 | Output: {out}")
if __name__=="__main__":main()
