from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_starts_and_handles_placeholder_submission() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py")).run()
    assert not app.exception
    assert app.title[0].value == "NALTRA"
    assert app.button[0].disabled
    assert "Laya" in app.selectbox[0].options
    assert "Ensemble" in app.selectbox[0].options

    app.text_area[0].set_value("English ve Türkçe").run()
    assert not app.button[0].disabled

    app.button[0].click().run()
    assert not app.exception
    assert "not connected yet" in app.warning[0].value
