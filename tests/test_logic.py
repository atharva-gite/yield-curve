import numpy as np
import pandas as pd

from logic import (
    MATURITIES,
    SERIES_IDS,
    adf_stationarity_check,
    calculate_drawdown,
    call_llm,
    check_rate_limit,
    compute_pca,
    bond_risk,
    discount_factors,
    fit_smoothing_spline,
    fit_nss,
    kalman_filter_diebold_li,
    leave_one_maturity_cv,
    ns_curve,
    ns_forward,
    nss_curve,
    nss_forward,
    price_coupon_bond,
    resolve_groq_api_key,
    run_curve_spread_backtest,
    check_groq_model_availability,
    GROQ_MODEL_FALLBACK,
    GROQ_MODEL_PRIMARY,
)


def _forward_from_yield(curve, tau: np.ndarray, params: tuple[float, ...], step: float = 1e-4) -> np.ndarray:
    y = curve(tau, *params)
    slope = (curve(tau + step, *params) - curve(tau - step, *params)) / (2.0 * step)
    return y + tau * slope


def test_ns_forward_matches_yield_derivative_identity():
    params = (0.04, -0.015, 0.02, 0.7)
    tau = np.linspace(0.25, 30.0, 80)
    analytic = ns_forward(tau, *params)
    numerical = _forward_from_yield(ns_curve, tau, params)
    assert np.allclose(analytic, numerical, atol=1e-6)


def test_nss_forward_matches_yield_derivative_identity():
    params = (0.041, -0.018, 0.026, -0.012, 0.55, 1.75)
    tau = np.linspace(0.25, 30.0, 80)
    analytic = nss_forward(tau, *params)
    numerical = _forward_from_yield(nss_curve, tau, params)
    assert np.allclose(analytic, numerical, atol=1e-6)


def test_ns_forward_flat_curve_is_beta0_and_short_end_is_beta0_plus_beta1():
    flat = ns_forward(np.array([0.0, 2.0, 10.0, 30.0]), 0.03, 0.0, 0.0, 0.6)
    assert np.allclose(flat, 0.03)
    short_end = ns_forward(np.array([0.0]), 0.04, -0.01, 0.02, 0.5)[0]
    assert np.isclose(short_end, 0.03)
    nss_short = nss_forward(np.array([0.0]), 0.041, -0.018, 0.026, -0.012, 0.55, 1.75)[0]
    assert np.isclose(nss_short, 0.041 - 0.018)


def test_ns_loading_limit_is_finite_near_zero():
    values = ns_curve(np.array([1e-12, 1.0]), 0.03, -0.01, 0.02, 1e-12)
    assert np.isfinite(values).all()
    assert np.isclose(values[0], 0.02, atol=1e-8)


def test_nss_recovers_synthetic_parameters():
    expected = {
        "beta0": 0.041,
        "beta1": -0.018,
        "beta2": 0.026,
        "beta3": -0.012,
        "lambda1": 0.55,
        "lambda2": 1.75,
    }
    y = nss_curve(
        MATURITIES,
        expected["beta0"],
        expected["beta1"],
        expected["beta2"],
        expected["beta3"],
        expected["lambda1"],
        expected["lambda2"],
    )
    result = fit_nss(MATURITIES, y)
    fitted_again = nss_curve(
        MATURITIES,
        result.params["beta0"],
        result.params["beta1"],
        result.params["beta2"],
        result.params["beta3"],
        result.params["lambda1"],
        result.params["lambda2"],
    )
    assert result.rmse < 1e-5
    assert np.max(np.abs(fitted_again - y)) < 3e-5
    assert abs(result.params["beta0"] - expected["beta0"]) < 0.005
    assert abs(result.params["lambda1"] - result.params["lambda2"]) > 0.02


def test_pca_on_three_factor_synthetic_panel_explains_variance():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=320, freq="B")
    level = rng.normal(0, 0.0007, len(dates)).cumsum()
    slope = rng.normal(0, 0.00025, len(dates)).cumsum()
    curve = rng.normal(0, 0.00015, len(dates)).cumsum()
    maturity_scale = (MATURITIES - MATURITIES.mean()) / np.ptp(MATURITIES)
    hump = -((MATURITIES - 5.0) ** 2)
    hump = (hump - hump.mean()) / np.max(np.abs(hump))
    panel = []
    base = 0.03 + 0.002 * maturity_scale
    for i in range(len(dates)):
        panel.append(base + level[i] + slope[i] * maturity_scale + curve[i] * hump)
    frame = pd.DataFrame(panel, index=dates, columns=SERIES_IDS)
    pca = compute_pca(frame)
    assert pca["changes"].shape[0] == len(dates) - 1
    assert pca["loadings"].shape == (len(SERIES_IDS), 3)
    assert float(np.sum(pca["explained_variance"])) > 0.98
    assert {"Level", "Slope", "Curvature"} == set(pca["loadings"].columns)


def test_kalman_filter_outputs_stable_finite_states():
    dates = pd.date_range("2022-01-01", periods=60, freq="B")
    rows = []
    for i in range(len(dates)):
        rows.append(ns_curve(MATURITIES, 0.035 + i * 1e-6, -0.012, 0.018, 0.6))
    frame = pd.DataFrame(rows, index=dates, columns=SERIES_IDS)
    filtered = kalman_filter_diebold_li(frame)
    assert filtered.shape == (60, 3)
    assert np.isfinite(filtered.values).all()
    assert filtered.tail(10).std().max() < 1e-3


def test_discount_and_zero_coupon_bond_price():
    price = price_coupon_bond(
        maturity=2.0,
        coupon_rate=0.0,
        face_value=100.0,
        payment_frequency=1,
        curve_function=lambda times: np.full_like(times, 0.05, dtype=float),
    )
    assert np.isclose(price, 100.0 * np.exp(-0.05 * 2.0))
    assert np.isclose(discount_factors([2.0], [0.05])[0], np.exp(-0.1))


def test_bond_risk_flat_two_year_zero_matches_analytic_dv01():
    def flat(times: np.ndarray) -> np.ndarray:
        return np.full_like(times, 0.05, dtype=float)

    risk = bond_risk(maturity=2.0, coupon_rate=0.0, face_value=100.0, payment_frequency=1, curve_function=flat)
    expected_dv01 = 100.0 * 2.0 * np.exp(-0.10) * 0.0001
    assert np.isclose(risk["modified_duration"], 2.0, atol=1e-3)
    assert np.isclose(risk["dv01"], expected_dv01, rtol=1e-3)


def test_key_rate_bump_moves_the_local_bond_only():
    def flat(times: np.ndarray) -> np.ndarray:
        return np.full_like(times, 0.04, dtype=float)

    two_year = bond_risk(maturity=2.0, coupon_rate=0.0, face_value=100.0, payment_frequency=1, curve_function=flat)
    ten_year = bond_risk(maturity=10.0, coupon_rate=0.0, face_value=100.0, payment_frequency=1, curve_function=flat)
    assert abs(two_year["key_rate_dv01"][10.0]) < 1e-8
    assert ten_year["key_rate_dv01"][10.0] > 0.01
    assert np.isclose(two_year["key_rate_dv01"][2.0], two_year["dv01"])


def test_adf_stationarity_check_detects_stationary_noise():
    rng = np.random.default_rng(11)
    residuals = rng.normal(0, 0.001, 200)
    result = adf_stationarity_check(residuals)
    assert result["p_value"] < 0.05
    assert result["passed"] is True


def test_spline_fit_and_leave_one_maturity_cv():
    y = ns_curve(MATURITIES, 0.04, -0.015, 0.02, 0.7)
    spline = fit_smoothing_spline(MATURITIES, y)
    cv = leave_one_maturity_cv(MATURITIES, y, "spline")
    assert spline.rmse < 1e-3
    assert np.isfinite(cv["errors"]).all()
    assert cv["rmse"] < 0.01


def test_walk_forward_backtest_strategy_and_drawdown_stats():
    dates = pd.date_range("2020-01-01", periods=420, freq="B")
    slope = np.sin(np.linspace(0, 14, len(dates))) * 0.005
    base = 0.03
    frame = pd.DataFrame(index=dates, columns=SERIES_IDS, dtype=float)
    for idx, series_id in enumerate(SERIES_IDS):
        frame[series_id] = base + idx * 0.0005
    frame["DGS2"] = base - slope / 2
    frame["DGS5"] = base + np.cos(np.linspace(0, 8, len(dates))) * 0.001
    frame["DGS10"] = base + slope / 2
    result = run_curve_spread_backtest(
        frame,
        strategy="butterfly_mean_reversion",
        train_window=120,
        test_window=20,
    )
    assert "drawdown" in result
    assert np.isfinite(result["pnl"]).all()
    assert result["max_drawdown"] <= 0
    assert result["max_drawdown_duration"] >= 0
    dd = calculate_drawdown(result["cumulative_pnl"])
    assert "max_duration" in dd


def test_call_llm_builds_context_payload_and_handles_errors(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Message:
                content = "Context-grounded answer."

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("logic.GROQ_CLIENT_FACTORY", lambda api_key: FakeClient())
    result = call_llm([{"role": "user", "content": "What is the curve saying?"}], {"selected_date": "2026-08-20"})
    assert result.ok is True
    assert result.content == "Context-grounded answer."
    assert captured["messages"][0]["role"] == "system"
    assert "ACTIVE_SESSION_CONTEXT_JSON" in captured["messages"][0]["content"]
    assert "2026-08-20" in captured["messages"][0]["content"]

    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("rate limited")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()

    monkeypatch.setattr("logic.GROQ_CLIENT_FACTORY", lambda api_key: FailingClient())
    failed = call_llm([{"role": "user", "content": "Explain"}], {"x": 1})
    assert failed.ok is False
    assert "rate limited" in failed.error


def test_call_llm_retries_once_on_model_not_found(monkeypatch):
    attempted_models = []

    class ModelNotFoundError(Exception):
        status_code = 404
        body = {"error": {"code": "model_not_found", "message": "model does not exist"}}

    class FakeCompletions:
        def create(self, **kwargs):
            attempted_models.append(kwargs["model"])
            if kwargs["model"] == GROQ_MODEL_PRIMARY:
                raise ModelNotFoundError("model_not_found")

            class Message:
                content = "Fallback answer."

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr("logic._load_project_dotenv", lambda: None)
    monkeypatch.setattr("logic.GROQ_CLIENT_FACTORY", lambda api_key: FakeClient())

    result = call_llm([{"role": "user", "content": "Explain"}], {"x": 1})

    assert result.ok is True
    assert result.content == "Fallback answer."
    assert attempted_models == [GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK]


def test_call_llm_does_not_retry_non_404_errors(monkeypatch):
    attempted_models = []

    class AuthError(Exception):
        status_code = 401
        body = {"error": {"code": "invalid_api_key", "message": "bad key"}}

    class FakeCompletions:
        def create(self, **kwargs):
            attempted_models.append(kwargs["model"])
            raise AuthError("bad key")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr("logic._load_project_dotenv", lambda: None)
    monkeypatch.setattr("logic.GROQ_CLIENT_FACTORY", lambda api_key: FakeClient())

    result = call_llm([{"role": "user", "content": "Explain"}], {"x": 1})

    assert result.ok is False
    assert "bad key" in result.error
    assert attempted_models == [GROQ_MODEL_PRIMARY]


def test_groq_model_availability_flags_missing_model(monkeypatch):
    class Model:
        def __init__(self, model_id):
            self.id = model_id

    class Models:
        def list(self):
            class Response:
                data = [Model(GROQ_MODEL_FALLBACK)]

            return Response()

    class FakeClient:
        models = Models()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr("logic._load_project_dotenv", lambda: None)
    monkeypatch.setattr("logic.GROQ_CLIENT_FACTORY", lambda api_key: FakeClient())

    available, warning = check_groq_model_availability()

    assert available is False
    assert warning is not None
    assert "WARNING" in warning
    assert GROQ_MODEL_PRIMARY in warning


def test_call_llm_missing_key_returns_inline_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("logic._load_project_dotenv", lambda: None)
    monkeypatch.setattr("logic.STREAMLIT_SECRETS_PATHS", ())
    monkeypatch.setattr("logic.GROQ_CLIENT_FACTORY", None)
    result = call_llm([{"role": "user", "content": "Explain"}], {"selected_date": "none"})
    assert result.ok is False
    assert "GROQ_API_KEY" in result.error


def test_groq_key_resolution_prefers_environment(monkeypatch):
    def forbidden_secrets_getter(key):
        raise AssertionError("st.secrets should not be checked when env already has the key")

    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setattr("logic._load_project_dotenv", lambda: None)
    key, source = resolve_groq_api_key(forbidden_secrets_getter)
    assert key == "env-key"
    assert source == ".env"


def test_groq_key_resolution_can_use_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("logic._load_project_dotenv", lambda: None)
    key, source = resolve_groq_api_key(lambda key: "secret-key")
    assert key == "secret-key"
    assert source == "st.secrets"


def test_rate_limiter_blocks_eleventh_request_within_window(tmp_path):
    db_path = tmp_path / "rate_limits.db"
    results = [
        check_rate_limit("ip:203.0.113.10", now=1_000_000 + idx, db_path=db_path)
        for idx in range(10)
    ]
    blocked = check_rate_limit("ip:203.0.113.10", now=1_000_100, db_path=db_path)
    other_identifier = check_rate_limit("ip:203.0.113.11", now=1_000_100, db_path=db_path)

    assert all(result.allowed for result in results)
    assert results[-1].remaining == 0
    assert blocked.allowed is False
    assert blocked.remaining == 0
    assert other_identifier.allowed is True


def test_rate_limiter_allows_after_sliding_window_expires(tmp_path):
    db_path = tmp_path / "rate_limits.db"
    for idx in range(10):
        check_rate_limit("session:abc", now=2_000_000 + idx, db_path=db_path)

    blocked = check_rate_limit("session:abc", now=2_000_100, db_path=db_path)
    allowed = check_rate_limit("session:abc", now=2_000_000 + 5 * 3600 + 10, db_path=db_path)

    assert blocked.allowed is False
    assert allowed.allowed is True
    assert allowed.remaining == 9
