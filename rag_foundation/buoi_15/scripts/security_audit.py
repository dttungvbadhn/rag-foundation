from __future__ import annotations
import json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read_csv
from src.security import parse_roles
from src.secure_retriever import SecureRetriever

def build_cases(rows):
 docs=defaultdict(list)
 for row in rows: docs[row["document_id"]].append(row)
 cases=[]
 for doc_id,items in docs.items():
  roles=parse_roles(items[0]["allowed_roles"])
  if roles==["Admin","HR"] and sum(c["kind"]=="HR" for c in cases)<3: kind,unauth,auth="HR",["Staff"],["HR"]
  elif roles==["Admin","Risk_Manager","Staff"] and sum(c["kind"]=="RISK" for c in cases)<2: kind,unauth,auth="RISK",["Guest"],["Risk_Manager"]
  else: continue
  sample=max(items,key=lambda x:len(x["text"]));query=(sample["title"]+" "+sample["text"])[:500]
  cases.append({"name":f"{kind}-{doc_id}","kind":kind,"query":query,"target_sensitive_document_id":doc_id,"unauthorized_roles":unauth,"authorized_roles":auth})
  if len(cases)==5: break
 return cases

def main():
 rows=read_csv(ROOT/"data/processed/chunks_secure.csv");cases=build_cases(rows)
 if len(cases)<5: raise AssertionError("Không đủ 5 tài liệu nhạy cảm đã xác minh")
 engine=SecureRetriever();results=[]
 for case in cases:
  denied=engine.retrieve(case["query"],case["unauthorized_roles"],"bm25",10)
  allowed=engine.retrieve(case["query"],case["authorized_roles"],"bm25",10)
  leaked=any(x["document_id"]==case["target_sensitive_document_id"] for x in denied)
  visible=any(x["document_id"]==case["target_sensitive_document_id"] for x in allowed)
  passed=not leaked and visible
  results.append({**case,"passed":passed,"leaked":leaked,"visible_when_authorized":visible,"denied_ids":[x["document_id"] for x in denied],"allowed_ids":[x["document_id"] for x in allowed]})
 passed=sum(x["passed"] for x in results);lines=["# Security audit report","",f"- Tests: {len(results)}",f"- PASS: {passed}",f"- FAIL: {len(results)-passed}",""]
 for r in results:
  lines += [f"## {'PASS' if r['passed'] else 'FAIL'} — {r['name']}","",f"- Target document: `{r['target_sensitive_document_id']}`",f"- Unauthorized roles: `{r['unauthorized_roles']}`",f"- Authorized roles: `{r['authorized_roles']}`",f"- Leakage detected: `{r['leaked']}`",f"- Visible when authorized: `{r['visible_when_authorized']}`",""]
 lines += ["## Conclusion","",("BASIC DATA SECURITY: PASS" if passed==len(results) else "BASIC DATA SECURITY: FAIL")]
 out=ROOT/"outputs/security_audit_report.md";out.write_text("\n".join(lines),encoding="utf-8");print(out.read_text(encoding="utf-8"))
 if passed!=len(results): raise SystemExit(1)
if __name__=="__main__":main()
