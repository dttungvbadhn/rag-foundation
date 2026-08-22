MATCH p=(v:VanBan {lab_session:'buoi_16'})-[r]->(n) RETURN p LIMIT 100;
MATCH p=(v:VanBan {lab_session:'buoi_16'})-[:CONTAINS]->(d:DieuKhoan) RETURN p LIMIT 100;
MATCH p=(a:DieuKhoan {lab_session:'buoi_16'})-[:NEXT*1..5]->(b:DieuKhoan) RETURN p LIMIT 30;
MATCH ()-[r {lab_session:'buoi_16'}]->() RETURN type(r), count(*) ORDER BY count(*) DESC;
MATCH (n {lab_session:'buoi_16'}) WHERE NOT (n)--() RETURN labels(n), n.id;
