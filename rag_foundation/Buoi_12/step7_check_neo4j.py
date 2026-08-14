from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_DIR / ".env")
    load_dotenv(BASE_DIR / ".env")
    uri = os.getenv("NEO4J_URI", "").strip()
    user = os.getenv("NEO4J_USER", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_USER", user),
            ("NEO4J_PASSWORD", password),
            ("NEO4J_DATABASE", database),
        )
        if not value
    ]
    if missing:
        print("NEO4J_CONNECTION=FAIL")
        print(f"missing_config={missing}")
        raise SystemExit(2)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        record = driver.execute_query(
            "RETURN 1 AS ok",
            database_=database,
            routing_="r",
        ).records[0]
        passed = record["ok"] == 1
        print(f"database={database}")
        print(f"read_query_ok={passed}")
        print(f"NEO4J_CONNECTION={'PASS' if passed else 'FAIL'}")
        if not passed:
            raise SystemExit(1)
    except Exception as exc:
        print(f"database={database}")
        print("NEO4J_CONNECTION=FAIL")
        print(f"error={type(exc).__name__}: {str(exc)[:500]}")
        raise SystemExit(1) from exc
    finally:
        driver.close()


if __name__ == "__main__":
    main()
