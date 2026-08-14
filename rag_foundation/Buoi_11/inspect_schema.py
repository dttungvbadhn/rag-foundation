"""Read-only Neo4j schema/index inspection for validating Lab 1 compatibility."""

from neo4j import GraphDatabase

from neo4j_connection import Neo4jConfig


def main() -> None:
    config = Neo4jConfig.from_env()
    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    queries = {
        "INDEXES": "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state RETURN name,type,entityType,labelsOrTypes,properties,state",
        "LABELS": "MATCH (n) UNWIND labels(n) AS label RETURN label,count(*) AS count ORDER BY count DESC",
        "RELATIONSHIPS": "MATCH ()-[r]->() RETURN type(r) AS type,count(*) AS count ORDER BY count DESC",
        "SAMPLE_PROPERTIES": "MATCH (n) RETURN labels(n) AS labels, keys(n) AS properties, count(*) AS count ORDER BY count DESC LIMIT 10",
        "LEGAL_EDGES": "MATCH (a)-[r]->(b) WHERE type(r) IN ['CAN_CU','THAY_THE','SUA_DOI_BO_SUNG','HOP_NHAT','VAN_BAN_BO_SUNG'] RETURN type(r) AS rel, labels(a) AS from_labels, a.id AS from_id, a.so_ky_hieu AS from_so, labels(b) AS to_labels, b.id AS to_id, b.so_ky_hieu AS to_so",
    }
    try:
        with driver.session(database=config.database) as session:
            for heading, query in queries.items():
                print(heading)
                for record in session.run(query):
                    print(dict(record))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
