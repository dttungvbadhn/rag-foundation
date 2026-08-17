from __future__ import annotations
import os
from .common import ROOT

def get_graph_hints(results: list[dict], limit_per_result: int = 10) -> tuple[list[dict], str]:
    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        for path in (ROOT / ".env", ROOT.parents[2] / ".env"):
            if path.exists(): load_dotenv(path, override=False)
        uri, password = os.getenv("NEO4J_URI"), os.getenv("NEO4J_PASSWORD")
        if not uri or not password: return [], "Neo4j chưa được cấu hình."
        query = """
        UNWIND $items AS item
        MATCH (v:VanBan {id:item.document_id, lab_session:'buoi_14'})
        OPTIONAL MATCH (v)-[vr {lab_session:'buoi_14'}]->(target)
        WITH item, v, collect(DISTINCT {from:v.id, relation:type(vr), to:target.id})[0..$limit] AS doc_relations
        OPTIONAL MATCH (d:DieuKhoan {id:item.chunk_id, lab_session:'buoi_14'})-[dr {lab_session:'buoi_14'}]->(next)
        RETURN item.document_id AS document_id, item.chunk_id AS chunk_id,
               doc_relations, collect(DISTINCT {from:d.id, relation:type(dr), to:next.id})[0..$limit] AS chunk_relations
        """
        items=[{"document_id":x["document_id"],"chunk_id":x["chunk_id"]} for x in results]
        auth=(os.getenv("NEO4J_USER","neo4j"),password)
        with GraphDatabase.driver(uri,auth=auth) as driver:
            with driver.session(database=os.getenv("NEO4J_DATABASE","neo4j")) as session:
                rows=session.run(query,items=items,limit=limit_per_result).data()
        return rows, "Neo4j connected"
    except Exception as exc:
        return [], f"Neo4j chưa sẵn sàng: {exc}"
