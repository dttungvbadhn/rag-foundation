import unittest
from src.hybrid_retriever import HybridRetriever
class Fake:
 def __init__(self,rows): self.rows=rows
 def search(self,q,k): return self.rows[:k]
class TestRRF(unittest.TestCase):
 def test_rrf_merges_without_duplicates(self):
  h=object.__new__(HybridRetriever);h.rrf_k=60;h.bm25=Fake([{"chunk_id":"a","rank":1,"citation":"a"}]);h.dense=Fake([{"chunk_id":"a","rank":1,"citation":"a"},{"chunk_id":"b","rank":2,"citation":"b"}]); out=h.search("q",2,2);self.assertEqual([x["chunk_id"] for x in out],["a","b"]);self.assertIn("bm25_rank",out[0]);self.assertIn("dense_rank",out[0])
if __name__=="__main__":unittest.main()
