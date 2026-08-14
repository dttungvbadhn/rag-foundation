from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitSmokeTests(unittest.TestCase):
    def test_overview_renders_without_exception(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("Tổng quan Knowledge Graph" in item.value for item in app.header))

    def test_all_navigation_pages_render(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        for page in ("Tìm kiếm văn bản", "Khám phá graph", "Chất lượng dữ liệu"):
            with self.subTest(page=page):
                app.sidebar.radio[0].set_value(page)
                app.run(timeout=30)
                self.assertEqual(list(app.exception), [])


if __name__ == "__main__":
    unittest.main()
