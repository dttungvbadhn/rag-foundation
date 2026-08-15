// Uniqueness constraints; safe to run repeatedly.
CREATE CONSTRAINT rui_ro_id IF NOT EXISTS FOR (n:RuiRo) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT kiem_soat_id IF NOT EXISTS FOR (n:KiemSoat) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT su_kien_rui_ro_id IF NOT EXISTS FOR (n:SuKienRuiRo) REQUIRE n.id IS UNIQUE;
