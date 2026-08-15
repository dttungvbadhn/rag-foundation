// A. Xem toàn bộ graph
MATCH (a)-[r]->(b) RETURN a, r, b;

// B. Kiểm soát giảm thiểu một rủi ro
MATCH (c:KiemSoat)-[r:MITIGATES]->(risk:RuiRo {id: $risk_id}) RETURN c, r, risk;

// C. Sự kiện của một rủi ro
MATCH (risk:RuiRo {id: $risk_id})-[r:OBSERVED_AS]->(event:SuKienRuiRo) RETURN risk, r, event;

// D. Đường KiemSoat -> RuiRo -> SuKienRuiRo
MATCH path=(c:KiemSoat)-[:MITIGATES]->(risk:RuiRo)-[:OBSERVED_AS]->(event:SuKienRuiRo) RETURN path;

// E. Rủi ro không có kiểm soát
MATCH (risk:RuiRo) WHERE NOT EXISTS { MATCH (:KiemSoat)-[:MITIGATES]->(risk) } RETURN risk;

// F. Relation chưa VERIFIED
MATCH (a)-[r]->(b) WHERE coalesce(r.verification_status, '') <> 'VERIFIED' RETURN a, r, b;
