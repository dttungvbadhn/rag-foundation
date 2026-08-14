"""Cau hinh va kiem tra ket noi Neo4j cho Graph RAG Lab 2."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent


class Neo4jConfigError(ValueError):
    """Raised when required Neo4j settings are missing or invalid."""


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Neo4jConfig":
        load_dotenv(env_file or PROJECT_DIR / ".env", override=False)
        values = {
            "uri": os.getenv("NEO4J_URI", "").strip(),
            "database": os.getenv("NEO4J_DATABASE", "").strip(),
            "user": os.getenv("NEO4J_USER", "").strip(),
            "password": os.getenv("NEO4J_PASSWORD", ""),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            names = ", ".join(f"NEO4J_{name.upper()}" for name in missing)
            raise Neo4jConfigError(f"Thieu bien cau hinh: {names}")
        if not values["uri"].startswith(("neo4j://", "bolt://")):
            raise Neo4jConfigError("NEO4J_URI phai dung neo4j:// hoac bolt://")
        return cls(**values)


def check_connection(config: Neo4jConfig, driver_factory: Any = None) -> dict[str, Any]:
    """Verify connectivity and run one read-only query against the selected database."""
    if driver_factory is None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Chua cai neo4j driver; hay cai requirements.txt") from exc
        driver_factory = GraphDatabase.driver

    driver = driver_factory(config.uri, auth=(config.user, config.password))
    try:
        driver.verify_connectivity()
        with driver.session(database=config.database) as session:
            record = session.run(
                "MATCH (n) RETURN count(n) AS node_count"
            ).single(strict=True)
        return {
            "status": "connected",
            "uri": config.uri,
            "database": config.database,
            "user": config.user,
            "node_count": int(record["node_count"]),
        }
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiem tra ket noi Neo4j Graph RAG")
    parser.add_argument("command", choices=["status"])
    args = parser.parse_args()
    if args.command == "status":
        try:
            result = check_connection(Neo4jConfig.from_env())
        except Exception as exc:
            print(f"Status: connection_failed", file=sys.stderr)
            print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("Hint: khoi dong Neo4j va kiem tra Bolt port 7687.", file=sys.stderr)
            return 1
        print(f"Status: {result['status']}")
        print(f"URI: {result['uri']}")
        print(f"Database: {result['database']}")
        print(f"User: {result['user']}")
        print(f"Nodes: {result['node_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
