import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neo4j_connection import Neo4jConfig, Neo4jConfigError, check_connection


class FakeRecord(dict):
    pass


class FakeResult:
    def single(self, strict=False):
        return FakeRecord(node_count=7)


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query):
        assert query == "MATCH (n) RETURN count(n) AS node_count"
        return FakeResult()


class FakeDriver:
    def __init__(self):
        self.closed = False
        self.verified = False

    def verify_connectivity(self):
        self.verified = True

    def session(self, database):
        assert database == "kb-hops"
        return FakeSession()

    def close(self):
        self.closed = True


class ConnectionTests(unittest.TestCase):
    def test_config_and_read_only_health_check(self):
        env = {
            "NEO4J_URI": "neo4j://localhost:7687",
            "NEO4J_DATABASE": "kb-hops",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Neo4jConfig.from_env()
        driver = FakeDriver()
        result = check_connection(config, lambda *_args, **_kwargs: driver)
        self.assertEqual("connected", result["status"])
        self.assertEqual(7, result["node_count"])
        self.assertTrue(driver.verified)
        self.assertTrue(driver.closed)

    def test_missing_secret_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_file = Path(directory) / "missing.env"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(Neo4jConfigError):
                    Neo4jConfig.from_env(env_file=missing_file)


if __name__ == "__main__":
    unittest.main()
