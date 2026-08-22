from __future__ import annotations
import os,sys
from pathlib import Path
from collections import defaultdict
import re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import load_corpus,read_csv,source_dir

def main():
 try:
  from dotenv import load_dotenv
  from neo4j import GraphDatabase
 except ImportError as exc: return report(f"NOT RUN\n\nMissing dependency: {exc}")
 for env_path in (ROOT / ".env", ROOT.parents[2] / ".env"):
  if env_path.exists():
   load_dotenv(env_path, override=False)
 uri=os.getenv("NEO4J_URI"); password=os.getenv("NEO4J_PASSWORD")
 if not uri or not password: return report("NOT RUN\n\nNEO4J_URI or NEO4J_PASSWORD is not configured.")
 auth=(os.getenv("NEO4J_USER","neo4j"),password); database=os.getenv("NEO4J_DATABASE","neo4j"); corpus=load_corpus(); src=source_dir(); docs=read_csv(src/"metadata.csv"); rels=read_csv(src/"relationships.csv")
 try:
  with GraphDatabase.driver(uri,auth=auth) as driver:
   driver.verify_connectivity()
   with driver.session(database=database) as s:
    existing=s.run("MATCH (d:DieuKhoan {lab_session:'buoi_16'}) RETURN count(d) AS total").single()["total"]
    if existing < len(corpus):
     s.run("UNWIND $rows AS r MERGE (v:VanBan {id:r.id,lab_session:'buoi_16'}) SET v.title=r.title,v.document_type=r.loai_van_ban,v.status=r.tinh_trang_hieu_luc",rows=docs).consume()
     s.run("UNWIND $rows AS r MERGE (d:DieuKhoan {id:r.chunk_id,lab_session:'buoi_16'}) SET d.document_id=r.document_id,d.text=r.text,d.article=r.article WITH d,r MATCH (v:VanBan {id:r.document_id,lab_session:'buoi_16'}) MERGE (v)-[e:CONTAINS]->(d) SET e.lab_session='buoi_16'",rows=corpus).consume()
     bydoc={}
     for x in corpus: bydoc.setdefault(x["document_id"],[]).append(x)
     pairs=[{"a":a["chunk_id"],"b":b["chunk_id"]} for xs in bydoc.values() for a,b in zip(xs,xs[1:])]
     s.run("UNWIND $rows AS r MATCH (a:DieuKhoan {id:r.a,lab_session:'buoi_16'}),(b:DieuKhoan {id:r.b,lab_session:'buoi_16'}) MERGE (a)-[e:NEXT]->(b) SET e.lab_session='buoi_16'",rows=pairs).consume()
    # Create the relationship types that actually occur in relationships.csv.
    grouped=defaultdict(list)
    for row in rels: grouped[row["relationship_type"]].append(row)
    for rel_type, rows in grouped.items():
     if not re.fullmatch(r"[A-Z][A-Z0-9_]*", rel_type):
      raise ValueError(f"Unsafe relationship type in source data: {rel_type}")
     query=f"UNWIND $rows AS r MATCH (v:VanBan {{id:r.source,lab_session:'buoi_16'}}) MERGE (t:RelationTarget {{id:r.target,lab_session:'buoi_16'}}) MERGE (v)-[e:{rel_type} {{target_id:r.target,lab_session:'buoi_16'}}]->(t) SET e.method=r.method,e.confidence=r.confidence,e.evidence=r.evidence"
     s.run(query,rows=rows).consume()
    # Controlled migration: remove only the obsolete generic Buoi 14 edges.
    s.run("MATCH ()-[r:SOURCE_RELATION {lab_session:'buoi_16'}]->() DELETE r").consume()
    counts=s.run("MATCH (n {lab_session:'buoi_16'}) RETURN labels(n) AS node_labels,count(*) AS total").data()
    rc=s.run("MATCH ()-[r {lab_session:'buoi_16'}]->() RETURN type(r) AS rel_type,count(*) AS total").data()
    orphans=s.run("MATCH (n {lab_session:'buoi_16'}) WHERE NOT (n)--() RETURN count(n) AS total").single()["total"]
  report("# KG build report\n\nStatus: RUN\n\nNodes: `"+str(counts)+"`\n\nRelationships: `"+str(rc)+"`\n\nOrphan nodes: `"+str(orphans)+"`")
 except Exception as exc: report(f"NOT RUN\n\nNeo4j error: `{exc}`")
def report(text):
 out=ROOT/"outputs/kg_build_report.md";out.parent.mkdir(exist_ok=True);out.write_text(text,encoding="utf-8");print(text)
if __name__=="__main__":main()
