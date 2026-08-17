from __future__ import annotations
import json,os,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read_csv
from src.security import parse_roles

def main():
 from dotenv import load_dotenv
 from neo4j import GraphDatabase
 for p in (ROOT/".env",ROOT.parents[2]/".env"):
  if p.exists(): load_dotenv(p,override=False)
 uri,password=os.getenv("NEO4J_URI"),os.getenv("NEO4J_PASSWORD")
 if not uri or not password: raise RuntimeError("Thiếu NEO4J_URI/NEO4J_PASSWORD")
 rows=read_csv(ROOT/"data/processed/chunks_secure.csv"); docs=defaultdict(lambda:{"roles":set(),"title":"","type":"","status":""})
 for r in rows:
  r["allowed_roles"]=parse_roles(r["allowed_roles"]);d=docs[r["document_id"]];d["roles"].update(r["allowed_roles"]);d.update(title=r.get("title",""),type=r.get("document_type",""),status=r.get("status",""))
 documents=[{"id":k,"allowed_roles":sorted(v["roles"]),**{x:v[x] for x in ("title","type","status")}} for k,v in docs.items()]
 with GraphDatabase.driver(uri,auth=(os.getenv("NEO4J_USER","neo4j"),password)) as driver:
  driver.verify_connectivity()
  with driver.session(database=os.getenv("NEO4J_DATABASE","neo4j")) as s:
   s.run("UNWIND $rows AS r MERGE (v:VanBan {id:r.id,lab_session:'buoi_15'}) SET v.title=r.title,v.document_type=r.type,v.status=r.status,v.allowed_roles=r.allowed_roles",rows=documents).consume()
   s.run("UNWIND $rows AS r MERGE (d:DieuKhoan {id:r.chunk_id,lab_session:'buoi_15'}) SET d.document_id=r.document_id,d.text=r.text,d.article=r.article,d.allowed_roles=r.allowed_roles WITH d,r MATCH (v:VanBan {id:r.document_id,lab_session:'buoi_15'}) MERGE (v)-[e:CONTAINS {lab_session:'buoi_15'}]->(d)",rows=rows).consume()
   counts=s.run("MATCH (n {lab_session:'buoi_15'}) WHERE n.allowed_roles IS NOT NULL RETURN labels(n) AS labels,count(*) AS total").data()
   sample=s.run("MATCH (v:VanBan {lab_session:'buoi_15'})-[:CONTAINS]->(d:DieuKhoan) RETURN v.id AS document_id,v.allowed_roles AS document_roles,d.id AS chunk_id,d.allowed_roles AS chunk_roles LIMIT 1").data()
 report=ROOT/"outputs/security_kg_report.md";report.write_text("# Secure KG report\n\nStatus: RUN\n\nCounts: `"+str(counts)+"`\n\nSample: `"+str(sample)+"`\n",encoding="utf-8")
 print(report.read_text(encoding="utf-8"))
if __name__=="__main__":main()
