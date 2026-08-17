import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

class TestStreamlitApp(unittest.TestCase):
    def test_three_required_query_modes(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        cases = [
            ("Điều 72 hiệu lực thi hành", "BM25"),
            ("Ai chịu trách nhiệm bảo đảm an toàn tiền và tài sản?", "Dense"),
            ("phạm vi điều chỉnh giao nhận tiền mặt", "Hybrid + Rerank"),
        ]
        for query, method in cases:
            app.text_input[0].input(query)
            app.selectbox[0].select(method)
            app.button[0].click().run(timeout=180)
            self.assertFalse(app.exception, f"Streamlit failed for {method}: {app.exception}")
            self.assertGreater(len(app.expander), 0)

if __name__ == "__main__": unittest.main()
