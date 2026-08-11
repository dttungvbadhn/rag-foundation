"""Config path và import isolation tests."""

import os
from pathlib import Path
import tempfile
import unittest

from rag_foundation.buoi_08 import advanced_rag


class ConfigIsolationTests(unittest.TestCase):
    def test_config_works_from_different_cwd(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                loaded = advanced_rag.load_advanced_config(
                    advanced_rag.BUOI_08_DIR / ".env.example"
                )
            finally:
                os.chdir(original)
        self.assertEqual("gemini-embedding-2", loaded.embedding_model)
        self.assertEqual("BAAI/bge-reranker-v2-m3", loaded.reranker_model)
        self.assertEqual("", loaded.api_key)


if __name__ == "__main__":
    unittest.main()
