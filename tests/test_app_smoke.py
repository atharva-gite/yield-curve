import pandas as pd
from streamlit.testing.v1 import AppTest

import logic


def test_streamlit_app_renders_without_exceptions(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=360, freq="B")
    rows = [logic.ns_curve(logic.MATURITIES, 0.04, -0.015, 0.02, 0.7) for _ in dates]
    frame = pd.DataFrame(rows, index=dates, columns=logic.SERIES_IDS)
    monkeypatch.setattr(logic, "get_yield_data", lambda: logic.YieldDataResult(frame, "test_cache"))
    monkeypatch.setattr(logic, "check_groq_model_availability", lambda: (True, None))
    app = AppTest.from_file("app.py")
    app.run(timeout=60)
    assert not app.exception
