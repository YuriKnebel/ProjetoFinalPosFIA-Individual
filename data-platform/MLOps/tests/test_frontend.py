import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[2]


class FrontendTest(unittest.TestCase):
    def test_streamlit_app_starts_without_exceptions(self) -> None:
        app = AppTest.from_file(
            DATA_PLATFORM_DIR / "MLOps/app/frontend/app.py"
        ).run(timeout=30)
        self.assertEqual(list(app.exception), [])


if __name__ == "__main__":
    unittest.main()
