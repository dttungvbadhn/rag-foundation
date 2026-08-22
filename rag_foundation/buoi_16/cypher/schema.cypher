CREATE CONSTRAINT buoi16_vanban IF NOT EXISTS FOR (n:VanBan) REQUIRE (n.id, n.lab_session) IS UNIQUE;
CREATE CONSTRAINT buoi16_dieukhoan IF NOT EXISTS FOR (n:DieuKhoan) REQUIRE (n.id, n.lab_session) IS UNIQUE;
CREATE INDEX buoi16_session IF NOT EXISTS FOR (n:DieuKhoan) ON (n.lab_session);
