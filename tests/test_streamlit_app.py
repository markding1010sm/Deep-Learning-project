from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_baseline_option_loads_real_checkpoint() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.selectbox[0].set_value("GloVe + MLP").run()
    assert app.slider[0].value == 0.5

    app.button[0].click().run(timeout=30)
    assert app.success
    assert not app.error
