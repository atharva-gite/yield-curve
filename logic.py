from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller

MATURITIES = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
SERIES_TO_MATURITY = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
SERIES_IDS = list(SERIES_TO_MATURITY.keys())
LLM_PROVIDER_ENV = "LLM_PROVIDER"
LLM_MODEL_ENV = "GROQ_MODEL"
DEFAULT_LLM_PROVIDER = "groq"
GROQ_MODEL_PRIMARY = "openai/gpt-oss-20b"
GROQ_MODEL_FALLBACK = "openai/gpt-oss-120b"
DEFAULT_GROQ_MODEL = GROQ_MODEL_PRIMARY
GROQ_CLIENT_FACTORY = None
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_HOURS = 5
RATE_LIMIT_DB_PATH = Path("data/rate_limits.db")
STREAMLIT_SECRETS_PATHS = (
    Path.cwd() / ".streamlit" / "secrets.toml",
    Path.home() / ".streamlit" / "secrets.toml",
)


@dataclass(frozen=True)
class YieldDataResult:
    data: pd.DataFrame
    source: str
    warning: str | None = None


@dataclass(frozen=True)
class FitResult:
    params: dict[str, float]
    fitted: np.ndarray
    residuals: np.ndarray
    rmse: float
    max_error: float
    success: bool
    message: str


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    content: str
    error: str | None = None


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: float
    limit: int
    window_seconds: int
    identifier: str


def _as_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _load_project_dotenv() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def _safe_loading(lambda_value: float, tau: np.ndarray) -> np.ndarray:
    x = lambda_value * tau
    out = np.empty_like(x, dtype=float)
    near_zero = np.abs(x) < 1e-8
    out[near_zero] = 1.0
    out[~near_zero] = (1.0 - np.exp(-x[~near_zero])) / x[~near_zero]
    return out


def ns_curve(
    maturities: Iterable[float] | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    lambda_: float,
) -> np.ndarray:
    tau = _as_array(maturities)
    l1 = _safe_loading(lambda_, tau)
    return beta0 + beta1 * l1 + beta2 * (l1 - np.exp(-lambda_ * tau))


def nss_curve(
    maturities: Iterable[float] | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda1: float,
    lambda2: float,
) -> np.ndarray:
    tau = _as_array(maturities)
    l1 = _safe_loading(lambda1, tau)
    l2 = _safe_loading(lambda2, tau)
    return (
        beta0
        + beta1 * l1
        + beta2 * (l1 - np.exp(-lambda1 * tau))
        + beta3 * (l2 - np.exp(-lambda2 * tau))
    )


def ns_forward(
    maturities: Iterable[float] | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    lambda_: float,
) -> np.ndarray:
    """Instantaneous forward of an NS curve treated as a continuous zero rate."""
    tau = _as_array(maturities)
    decay = np.exp(-lambda_ * tau)
    return beta0 + beta1 * decay + beta2 * (lambda_ * tau) * decay


def nss_forward(
    maturities: Iterable[float] | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda1: float,
    lambda2: float,
) -> np.ndarray:
    """Instantaneous forward of an NSS curve treated as a continuous zero rate."""
    tau = _as_array(maturities)
    decay1 = np.exp(-lambda1 * tau)
    decay2 = np.exp(-lambda2 * tau)
    return beta0 + beta1 * decay1 + beta2 * (lambda1 * tau) * decay1 + beta3 * (lambda2 * tau) * decay2


def ns_design_matrix(maturities: Iterable[float], lambda_: float) -> np.ndarray:
    tau = _as_array(maturities)
    l1 = _safe_loading(lambda_, tau)
    return np.column_stack([np.ones_like(tau), l1, l1 - np.exp(-lambda_ * tau)])


def nss_design_matrix(maturities: Iterable[float], lambda1: float, lambda2: float) -> np.ndarray:
    tau = _as_array(maturities)
    l1 = _safe_loading(lambda1, tau)
    l2 = _safe_loading(lambda2, tau)
    return np.column_stack(
        [np.ones_like(tau), l1, l1 - np.exp(-lambda1 * tau), l2 - np.exp(-lambda2 * tau)]
    )


def _ols_beta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def fit_ns(maturities: Iterable[float], yields: Iterable[float]) -> FitResult:
    tau = _as_array(maturities)
    y = _as_array(yields)

    def objective(params: np.ndarray) -> float:
        fitted = ns_curve(tau, params[0], params[1], params[2], params[3])
        return float(np.sum((y - fitted) ** 2))

    best_lambda = 0.6
    best_beta = _ols_beta(ns_design_matrix(tau, best_lambda), y)
    best_ssr = float(np.sum((y - ns_curve(tau, *best_beta, best_lambda)) ** 2))
    for lambda_ in np.arange(0.05, 3.0001, 0.05):
        beta = _ols_beta(ns_design_matrix(tau, lambda_), y)
        ssr = float(np.sum((y - ns_curve(tau, *beta, lambda_)) ** 2))
        if ssr < best_ssr:
            best_lambda = float(lambda_)
            best_beta = beta
            best_ssr = ssr

    result = minimize(
        objective,
        np.array([best_beta[0], best_beta[1], best_beta[2], best_lambda], dtype=float),
        method="L-BFGS-B",
        bounds=[(1e-6, 0.25), (-0.25, 0.25), (-0.25, 0.25), (0.01, 5.0)],
    )
    params = result.x if result.success else np.array([best_beta[0], best_beta[1], best_beta[2], best_lambda])
    fitted = ns_curve(tau, params[0], params[1], params[2], params[3])
    residuals = y - fitted
    return FitResult(
        params={"beta0": params[0], "beta1": params[1], "beta2": params[2], "lambda": params[3]},
        fitted=fitted,
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        max_error=float(np.max(np.abs(residuals))),
        success=bool(result.success),
        message=str(result.message),
    )


def rolling_ns_fit(yield_panel: pd.DataFrame, maturities: Iterable[float] | None = None) -> pd.DataFrame:
    """Fit Nelson-Siegel on month-end rows and return RMSE, lambda, and beta0."""
    tau = MATURITIES if maturities is None else _as_array(maturities)
    columns = [column for column in SERIES_IDS if column in yield_panel.columns]
    if len(columns) != len(tau):
        raise ValueError("Yield panel columns do not match the maturity grid.")
    panel = yield_panel[columns].dropna().copy()
    panel.index = pd.to_datetime(panel.index)
    month_end = panel.resample("ME").last().dropna()
    records = []
    for date, row in month_end.iterrows():
        fit = fit_ns(tau, row.to_numpy(dtype=float))
        records.append(
            {
                "date": date,
                "rmse": fit.rmse,
                "lambda": float(fit.params["lambda"]),
                "beta0": float(fit.params["beta0"]),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["rmse", "lambda", "beta0"])
    return frame.set_index("date")


def fit_nss(maturities: Iterable[float], yields: Iterable[float]) -> FitResult:
    tau = _as_array(maturities)
    y = _as_array(yields)
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    for lambda1 in np.arange(0.05, 3.0001, 0.15):
        for lambda2 in np.arange(0.05, 3.0001, 0.15):
            if abs(lambda1 - lambda2) < 0.05:
                continue
            beta = _ols_beta(nss_design_matrix(tau, lambda1, lambda2), y)
            fitted = nss_curve(tau, *beta, lambda1, lambda2)
            ssr = float(np.sum((y - fitted) ** 2))
            candidates.append((ssr, beta, float(lambda1), float(lambda2)))
    if not candidates:
        raise ValueError("No valid NSS lambda seed found.")

    def objective(lambdas: np.ndarray) -> float:
        lambda1, lambda2 = float(lambdas[0]), float(lambdas[1])
        lambda_gap = abs(lambda1 - lambda2)
        penalty = 1e3 * max(0.0, 0.025 - lambda_gap) ** 2
        x = nss_design_matrix(tau, lambda1, lambda2)
        beta = _ols_beta(x, y)
        beta0_penalty = 1e3 * max(0.0, 1e-6 - float(beta[0])) ** 2
        fitted = x @ beta
        return float((np.sum((y - fitted) ** 2) + penalty + beta0_penalty) * 1e8)

    best_refined: tuple[float, np.ndarray, object] | None = None
    for _, _, lambda1_seed, lambda2_seed in sorted(candidates, key=lambda item: item[0])[:12]:
        result = minimize(
            objective,
            np.array([lambda1_seed, lambda2_seed], dtype=float),
            method="L-BFGS-B",
            bounds=[(0.01, 5.0), (0.01, 5.0)],
            options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-12},
        )
        lambdas_candidate = result.x if result.success else np.array([lambda1_seed, lambda2_seed], dtype=float)
        x_candidate = nss_design_matrix(tau, lambdas_candidate[0], lambdas_candidate[1])
        beta_candidate = _ols_beta(x_candidate, y)
        ssr_candidate = float(np.sum((y - x_candidate @ beta_candidate) ** 2))
        if best_refined is None or ssr_candidate < best_refined[0]:
            best_refined = (ssr_candidate, lambdas_candidate, result)
    if best_refined is None:
        raise ValueError("NSS refinement failed to produce a candidate.")

    _, lambdas, result = best_refined
    beta = _ols_beta(nss_design_matrix(tau, lambdas[0], lambdas[1]), y)
    params = np.array([*beta, lambdas[0], lambdas[1]], dtype=float)
    if params[4] > params[5]:
        params = np.array([params[0], params[1], params[3], params[2], params[5], params[4]], dtype=float)
    fitted = nss_curve(tau, params[0], params[1], params[2], params[3], params[4], params[5])
    residuals = y - fitted
    return FitResult(
        params={
            "beta0": params[0],
            "beta1": params[1],
            "beta2": params[2],
            "beta3": params[3],
            "lambda1": params[4],
            "lambda2": params[5],
        },
        fitted=fitted,
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        max_error=float(np.max(np.abs(residuals))),
        success=bool(result.success),
        message=str(result.message),
    )


def fit_smoothing_spline(
    maturities: Iterable[float],
    yields: Iterable[float],
    smoothing_factor: float | None = None,
) -> FitResult:
    tau = _as_array(maturities)
    y = _as_array(yields)
    order = np.argsort(tau)
    tau_sorted = tau[order]
    y_sorted = y[order]
    s = smoothing_factor if smoothing_factor is not None else max(len(tau) * 1e-8, 1e-10)
    spline = UnivariateSpline(tau_sorted, y_sorted, s=s, k=min(3, len(tau_sorted) - 1))
    fitted_sorted = spline(tau_sorted)
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    residuals = y - fitted
    return FitResult(
        params={"smoothing_factor": float(s)},
        fitted=fitted,
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        max_error=float(np.max(np.abs(residuals))),
        success=True,
        message="Smoothing spline fit completed.",
    )


def predict_smoothing_spline(
    train_maturities: Iterable[float],
    train_yields: Iterable[float],
    target_maturities: Iterable[float],
    smoothing_factor: float | None = None,
) -> np.ndarray:
    tau = _as_array(train_maturities)
    y = _as_array(train_yields)
    order = np.argsort(tau)
    s = smoothing_factor if smoothing_factor is not None else max(len(tau) * 1e-8, 1e-10)
    spline = UnivariateSpline(tau[order], y[order], s=s, k=min(3, len(tau) - 1))
    return np.asarray(spline(_as_array(target_maturities)), dtype=float)


def leave_one_maturity_cv(
    maturities: Iterable[float],
    yields: Iterable[float],
    model: str,
) -> dict[str, object]:
    tau = _as_array(maturities)
    y = _as_array(yields)
    errors = []
    predictions = []
    for idx in range(len(tau)):
        train_mask = np.ones(len(tau), dtype=bool)
        train_mask[idx] = False
        train_tau = tau[train_mask]
        train_y = y[train_mask]
        target = np.array([tau[idx]])
        if model == "ns":
            fit = fit_ns(train_tau, train_y)
            pred = ns_curve(target, fit.params["beta0"], fit.params["beta1"], fit.params["beta2"], fit.params["lambda"])[0]
        elif model == "nss":
            fit = fit_nss(train_tau, train_y)
            pred = nss_curve(target, **fit.params)[0]
        elif model == "spline":
            pred = predict_smoothing_spline(train_tau, train_y, target)[0]
        else:
            raise ValueError(f"Unsupported curve model for CV: {model}")
        predictions.append(float(pred))
        errors.append(float(y[idx] - pred))
    errors_array = np.asarray(errors, dtype=float)
    return {
        "model": model,
        "predictions": np.asarray(predictions, dtype=float),
        "errors": errors_array,
        "rmse": float(np.sqrt(np.mean(errors_array**2))),
        "max_error": float(np.max(np.abs(errors_array))),
    }


def _fetch_with_fredapi() -> pd.DataFrame:
    _load_project_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is unset.")
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames = []
    for series_id in SERIES_IDS:
        series = fred.get_series(series_id).rename(series_id)
        frames.append(series)
    data = pd.concat(frames, axis=1)
    data.index.name = "date"
    return data


def _fetch_with_csv_endpoint(timeout: float = 15.0) -> pd.DataFrame:
    frames = []
    for series_id in SERIES_IDS:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text), parse_dates=["observation_date"])
        frame = frame.rename(columns={"observation_date": "date", series_id: series_id})
        frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
        frames.append(frame.set_index("date")[[series_id]])
    data = pd.concat(frames, axis=1)
    data.index.name = "date"
    return data


def _clean_yield_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned.replace(".", np.nan)
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(how="all").ffill().dropna()
    cleaned = cleaned[SERIES_IDS]
    finite_values = cleaned.to_numpy(dtype=float)
    median_level = float(np.nanmedian(finite_values))
    return cleaned / 100.0 if median_level > 0.5 else cleaned


def get_yield_data(cache_path: str | Path = "data/treasury_yields_cache.csv") -> YieldDataResult:
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for source, fetcher in (("fredapi", _fetch_with_fredapi), ("fred_csv", _fetch_with_csv_endpoint)):
        try:
            data = _clean_yield_data(fetcher())
            data.to_csv(cache, index_label="date")
            return YieldDataResult(data=data, source=source)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    if cache.exists():
        data = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        return YieldDataResult(
            data=_clean_yield_data(data),
            source="cache",
            warning="Live FRED downloads failed; loaded cached Treasury yields. " + " | ".join(errors),
        )
    raise RuntimeError("Unable to load Treasury yields from FRED or cache. " + " | ".join(errors))


def compute_pca(yield_panel: pd.DataFrame, n_components: int = 3) -> dict[str, object]:
    changes = yield_panel[SERIES_IDS].diff().dropna()
    centered = changes - changes.mean()
    covariance = np.cov(centered.values, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    scores = centered.values @ eigenvectors[:, :n_components]
    components = eigenvectors[:, :n_components].copy()
    labels = _label_pca_components(components)
    score_frame = pd.DataFrame(scores, index=changes.index, columns=[labels[i] for i in range(n_components)])
    return {
        "changes": changes,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "loadings": pd.DataFrame(components, index=SERIES_IDS, columns=[labels[i] for i in range(n_components)]),
        "explained_variance": explained[:n_components],
        "cumulative_explained_variance": np.cumsum(explained[:n_components]),
        "scores": score_frame,
        "diagnostic": None
        if float(np.sum(explained[:3])) >= 0.98
        else f"First 3 PCs explain {np.sum(explained[:3]):.2%}, below the 98% diagnostic threshold.",
    }


def _label_pca_components(loadings: np.ndarray) -> list[str]:
    names = [""] * loadings.shape[1]
    metrics = []
    for i in range(loadings.shape[1]):
        vector = loadings[:, i]
        if np.mean(vector) < 0:
            vector = -vector
            loadings[:, i] = vector
        same_sign_ratio = max(np.mean(vector >= 0), np.mean(vector <= 0))
        slope_score = abs(vector[-1] - vector[0])
        curvature_score = abs(vector[len(vector) // 2] - 0.5 * (vector[0] + vector[-1]))
        level_score = same_sign_ratio * abs(np.mean(vector)) / (np.std(vector) + 1e-12)
        metrics.append({"pc": i, "Level": level_score, "Slope": slope_score, "Curvature": curvature_score})
    remaining = set(range(loadings.shape[1]))
    for label in ("Level", "Slope", "Curvature"):
        chosen = max(remaining, key=lambda idx: metrics[idx][label])
        names[chosen] = label
        remaining.remove(chosen)
        if not remaining:
            break
    for idx in remaining:
        names[idx] = f"PC{idx + 1}"
    return names


def kalman_filter_diebold_li(
    yield_panel: pd.DataFrame,
    maturities: Iterable[float] = MATURITIES,
    lambda_: float = 0.6,
    transition: np.ndarray | None = None,
    process_cov: np.ndarray | None = None,
    measurement_cov: np.ndarray | None = None,
) -> pd.DataFrame:
    y = yield_panel.dropna().values.astype(float)
    tau = _as_array(maturities)
    h = ns_design_matrix(tau, lambda_)
    state_dim = 3
    phi = transition if transition is not None else np.eye(state_dim)
    q = process_cov if process_cov is not None else np.eye(state_dim) * 1e-6
    r = measurement_cov if measurement_cov is not None else np.eye(len(tau)) * 1e-5
    x = _ols_beta(h, y[0])
    p = np.eye(state_dim) * 1e-3
    estimates = []
    for row in y:
        x_pred = phi @ x
        p_pred = phi @ p @ phi.T + q
        innovation = row - h @ x_pred
        s = h @ p_pred @ h.T + r
        k = p_pred @ h.T @ np.linalg.pinv(s)
        x = x_pred + k @ innovation
        p = (np.eye(state_dim) - k @ h) @ p_pred
        estimates.append(x.copy())
    return pd.DataFrame(estimates, index=yield_panel.dropna().index, columns=["beta0", "beta1", "beta2"])


def discount_factors(maturities: Iterable[float], curve_yields: Iterable[float]) -> np.ndarray:
    tau = _as_array(maturities)
    y = _as_array(curve_yields)
    return np.exp(-y * tau)


KEY_RATE_TENORS = (2.0, 5.0, 10.0, 30.0)


def _key_rate_weight(tenor: float, times: np.ndarray) -> np.ndarray:
    """Tent weight: 1 at the key tenor and 0 at the neighboring key tenors."""
    nodes = np.array(KEY_RATE_TENORS, dtype=float)
    index = int(np.where(np.isclose(nodes, tenor))[0][0])
    left = 0.0 if index == 0 else float(nodes[index - 1])
    right = None if index == len(nodes) - 1 else float(nodes[index + 1])
    weight = np.zeros(len(times), dtype=float)
    if index == 0:
        weight[times <= tenor] = 1.0
    else:
        on_left = (times > left) & (times <= tenor)
        weight[on_left] = (times[on_left] - left) / (tenor - left)
    if right is None:
        weight[times >= tenor] = 1.0
    else:
        on_right = (times >= tenor) & (times < right)
        weight[on_right] = (right - times[on_right]) / (right - tenor)
    return weight


def bond_risk(
    maturity: float,
    coupon_rate: float,
    face_value: float,
    payment_frequency: int,
    curve_function: Callable[[np.ndarray], np.ndarray],
    bump: float = 0.0001,
) -> dict[str, object]:
    """Parallel DV01 and key-rate DV01 from a 1bp bump of curve_function."""
    price = price_coupon_bond(maturity, coupon_rate, face_value, payment_frequency, curve_function)

    def parallel_bumped(times: np.ndarray) -> np.ndarray:
        return curve_function(times) + bump

    price_up = price_coupon_bond(maturity, coupon_rate, face_value, payment_frequency, parallel_bumped)
    dv01 = -(price_up - price)
    modified_duration = dv01 / (price * bump) if price else float("nan")
    key_rate_dv01: dict[float, float] = {}
    for tenor in KEY_RATE_TENORS:

        def key_bumped(times: np.ndarray, tenor: float = tenor) -> np.ndarray:
            return curve_function(times) + bump * _key_rate_weight(tenor, _as_array(times))

        price_key = price_coupon_bond(maturity, coupon_rate, face_value, payment_frequency, key_bumped)
        key_rate_dv01[tenor] = -(price_key - price)
    return {
        "price": price,
        "dv01": dv01,
        "modified_duration": modified_duration,
        "key_rate_dv01": key_rate_dv01,
    }


def price_coupon_bond(
    maturity: float,
    coupon_rate: float,
    face_value: float,
    payment_frequency: int,
    curve_function: Callable[[np.ndarray], np.ndarray],
) -> float:
    periods = round(maturity * payment_frequency)
    cashflow_times = np.arange(1, periods + 1, dtype=float) / payment_frequency
    coupon = face_value * coupon_rate / payment_frequency
    cashflows = np.full(periods, coupon, dtype=float)
    cashflows[-1] += face_value
    yields = curve_function(cashflow_times)
    return float(np.sum(cashflows * discount_factors(cashflow_times, yields)))


def adf_stationarity_check(residuals: Iterable[float], alpha: float = 0.05) -> dict[str, float | bool | str]:
    values = _as_array(residuals)
    values = values[np.isfinite(values)]
    if len(values) < 4 or np.std(values) < 1e-12:
        return {"passed": False, "p_value": 1.0, "message": "Residual series is too short or constant for ADF."}
    try:
        _stat, p_value, *_ = adfuller(values, autolag="AIC")
    except Exception as exc:
        return {"passed": False, "p_value": 1.0, "message": f"ADF failed: {exc}"}
    return {
        "passed": bool(p_value < alpha),
        "p_value": float(p_value),
        "message": "Validated stationary residuals" if p_value < alpha else "Residuals failed stationarity guardrail",
    }


def run_curve_spread_backtest(
    yield_panel: pd.DataFrame,
    transaction_cost_bp: float = 1.0,
    slippage_multiplier: float = 1.0,
    lookback: int = 60,
    strategy: str = "slope_mean_reversion",
    train_window: int = 252,
    test_window: int = 21,
) -> dict[str, object]:
    panel = yield_panel[SERIES_IDS].dropna()
    series = _strategy_series(panel, strategy)
    signal = _walk_forward_signal(series, strategy, train_window=train_window, test_window=test_window)
    position = signal.shift(1).fillna(0.0)
    spread_change = series.diff().fillna(0.0)
    cost = abs(position.diff().fillna(position)) * transaction_cost_bp * 1e-4 * slippage_multiplier
    pnl = position * spread_change - cost
    pnl = pnl.dropna()
    sharpe = 0.0 if pnl.std() == 0 else float(np.sqrt(252) * pnl.mean() / pnl.std())
    cumulative_pnl = pnl.cumsum()
    drawdown = calculate_drawdown(cumulative_pnl)
    trades = pd.DataFrame(
        {
            "date": panel.index,
            "strategy_value": series,
            "signal": signal,
            "position": position,
            "pnl": pnl.reindex(panel.index).fillna(0.0),
        }
    )
    trade_log = trades[trades["position"].diff().fillna(trades["position"]).abs() > 0].copy()
    return {
        "strategy": strategy,
        "pnl": pnl,
        "cumulative_pnl": cumulative_pnl,
        "drawdown": drawdown["drawdown"],
        "max_drawdown": drawdown["max_drawdown"],
        "max_drawdown_duration": drawdown["max_duration"],
        "sharpe": sharpe,
        "trade_log": trade_log,
    }


def _strategy_series(panel: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy in {"slope_mean_reversion", "slope_momentum"}:
        return (panel["DGS10"] - panel["DGS2"]).rename("10Y-2Y slope")
    if strategy == "butterfly_mean_reversion":
        return (2.0 * panel["DGS5"] - panel["DGS2"] - panel["DGS10"]).rename("2s5s10s butterfly")
    raise ValueError(f"Unsupported strategy: {strategy}")


def _walk_forward_signal(
    series: pd.Series,
    strategy: str,
    train_window: int = 252,
    test_window: int = 21,
) -> pd.Series:
    signal = pd.Series(0.0, index=series.index)
    start = max(train_window, 2)
    for test_start in range(start, len(series), test_window):
        train = series.iloc[test_start - train_window : test_start].dropna()
        if len(train) < max(30, train_window // 4):
            continue
        mean = float(train.mean())
        std = float(train.std())
        if std == 0 or not np.isfinite(std):
            continue
        test_slice = series.iloc[test_start : min(test_start + test_window, len(series))]
        if strategy == "slope_momentum":
            window_signal = np.where(test_slice > mean + std, 1.0, np.where(test_slice < mean - std, -1.0, 0.0))
        else:
            window_signal = np.where(test_slice < mean - 2.0 * std, 1.0, np.where(test_slice > mean + 2.0 * std, -1.0, 0.0))
        signal.loc[test_slice.index] = window_signal
    return signal


def calculate_drawdown(cumulative_pnl: pd.Series) -> dict[str, object]:
    if cumulative_pnl.empty:
        empty = pd.Series(dtype=float)
        return {"drawdown": empty, "max_drawdown": 0.0, "max_duration": 0}
    running_peak = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_peak
    in_drawdown = drawdown < 0
    durations = []
    current = 0
    for flag in in_drawdown:
        current = current + 1 if flag else 0
        durations.append(current)
    return {
        "drawdown": drawdown,
        "max_drawdown": float(drawdown.min()),
        "max_duration": int(max(durations) if durations else 0),
    }


def serialize_session_context(context: dict[str, object]) -> str:
    return json.dumps(context, default=str, indent=2)


def resolve_groq_api_key(
    secrets_getter: Callable[[str], object | None] | None = None,
) -> tuple[str | None, str]:
    _load_project_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key, ".env"

    if secrets_getter is None and any(path.exists() for path in STREAMLIT_SECRETS_PATHS):
        try:
            import streamlit as st

            secrets_getter = st.secrets.get
        except Exception:
            secrets_getter = None

    if secrets_getter is not None:
        try:
            secret_value = secrets_getter("GROQ_API_KEY")
        except Exception:
            secret_value = None
        if secret_value:
            return str(secret_value), "st.secrets"

    return None, "not found"


def check_rate_limit(
    identifier: str,
    now: float | None = None,
    db_path: str | Path = RATE_LIMIT_DB_PATH,
    max_requests: int = RATE_LIMIT_MAX_REQUESTS,
    window_hours: float = RATE_LIMIT_WINDOW_HOURS,
    consume: bool = True,
) -> RateLimitResult:
    checked_at = time.time() if now is None else float(now)
    window_seconds = int(window_hours * 3600)
    window_start = checked_at - window_seconds
    clean_identifier = identifier.strip() if identifier and identifier.strip() else "global:shared"
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rate_limit_events (
                identifier TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_rate_limit_identifier_ts ON rate_limit_events(identifier, timestamp)")
        connection.execute("DELETE FROM rate_limit_events WHERE timestamp <= ?", (window_start,))
        rows = connection.execute(
            "SELECT timestamp FROM rate_limit_events WHERE identifier = ? ORDER BY timestamp ASC",
            (clean_identifier,),
        ).fetchall()
        timestamps = [float(row[0]) for row in rows]
        if len(timestamps) >= max_requests:
            reset_at = timestamps[0] + window_seconds
            return RateLimitResult(False, 0, reset_at, max_requests, window_seconds, clean_identifier)

        if consume:
            connection.execute(
                "INSERT INTO rate_limit_events(identifier, timestamp) VALUES (?, ?)",
                (clean_identifier, checked_at),
            )
            timestamps.append(checked_at)

        remaining = max(0, max_requests - len(timestamps))
        reset_at = (timestamps[0] + window_seconds) if timestamps else checked_at + window_seconds
        return RateLimitResult(True, remaining, reset_at, max_requests, window_seconds, clean_identifier)


def call_llm(messages: list[dict[str, str]], context_json: str | dict[str, object]) -> LLMResult:
    _load_project_dotenv()
    provider = os.getenv(LLM_PROVIDER_ENV, DEFAULT_LLM_PROVIDER).lower()
    if provider != "groq":
        return LLMResult(False, "", f"Unsupported LLM_PROVIDER '{provider}'. Set LLM_PROVIDER=groq.")
    api_key, key_source = resolve_groq_api_key()
    if not api_key:
        return LLMResult(False, "", "GROQ_API_KEY is missing. Add it to .env or Streamlit secrets.")
    if key_source == "st.secrets":
        os.environ.setdefault("GROQ_API_KEY", api_key)
    model = os.getenv(LLM_MODEL_ENV, GROQ_MODEL_PRIMARY)
    context_text = context_json if isinstance(context_json, str) else serialize_session_context(context_json)
    system_message = (
        "You are an institutional fixed-income quant analyst. Answer using the active session context. "
        "Do not invent data beyond this JSON context; call out missing fields clearly. "
        "When asked for an exhibit note, follow the requested headings exactly, quote figures with units, "
        "and write no forecast.\n\n"
        f"ACTIVE_SESSION_CONTEXT_JSON:\n{context_text}"
    )
    input_messages = [{"role": "system", "content": system_message}, *messages]
    try:
        client = _get_groq_client(api_key)
        response = _create_groq_chat_completion(client, model, input_messages)
        content = response.choices[0].message.content
        if not content:
            return LLMResult(False, "", "Groq returned an empty response.")
        return LLMResult(True, str(content))
    except Exception as exc:
        if _is_groq_model_not_found(exc) and model != GROQ_MODEL_FALLBACK:
            try:
                response = _create_groq_chat_completion(client, GROQ_MODEL_FALLBACK, input_messages)
                content = response.choices[0].message.content
                if not content:
                    return LLMResult(False, "", "Groq returned an empty response.")
                return LLMResult(True, str(content))
            except Exception as fallback_exc:
                return LLMResult(False, "", f"Groq request failed: {fallback_exc}")
        return LLMResult(False, "", f"Groq request failed: {exc}")


REPORT_SHARED_RULES = (
    "You are writing a rates-desk exhibit note, not a conversation and not a forecast. "
    "Use only JSON fields in ACTIVE_SESSION_CONTEXT_JSON.facts. "
    "Quote each number exactly as stored, and keep the unit that the field name implies "
    "(percent, bp, days, Sharpe). "
    "Every sentence in Reading and Decomposition must contain at least one quoted figure. "
    "Do not invent tenors, dates, p-values, or strategy names. "
    "Do not mention news, the Fed, recession, or a future path. "
    "If a field is absent, write that it is not in the brief. "
    "Output exactly three headings on their own lines: Reading, Decomposition, Caveats. "
    "Each heading is followed by one paragraph of 2-4 sentences. No bullets, no preamble, no closing sign-off."
)

REPORT_TOPIC_INSTRUCTIONS = {
    "curve": (
        "This exhibit is one U.S. Treasury constant-maturity curve. "
        "Reading: name the date; quote the 3-month, 2-year, 10-year, and 30-year yields in percent; "
        "quote the 2s10s and 10s30s gaps in basis points; say whether the 2s10s gap is positive (upward) "
        "or negative (inverted). "
        "Decomposition: compare Nelson-Siegel, Svensson, and spline in-sample RMSE in bp; "
        "note which fit is tightest and whether leave-one-maturity cross-val RMSE is larger; "
        "quote NS long-run (beta0) in percent and NS lambda; quote the NSS residual ADF p-value and "
        "whether those residuals are labelled stationary. "
        "Caveats: restate that these are constant-maturity yields and that any forward belongs to the fitted "
        "curve, not a stripped discount curve."
    ),
    "factors": (
        "This exhibit is a principal-component split of daily yield *changes*, not of yield levels. "
        "Reading: list each component by its supplied label and quote its share of variance in percent; "
        "quote the combined share of the first three. "
        "Decomposition: map the labels to motion — a level factor moves most tenors together, a slope factor "
        "moves the long end versus the front, curvature bends the belly — using only the labels in the brief; "
        "quote the diagnostic string if it is present. "
        "Caveats: this describes day-to-day changes over the sample, not the height of today's curve, "
        "and the leftover variance is the share not captured by the listed components."
    ),
    "backtest": (
        "This exhibit is a walk-forward test of one published-yield spread rule. "
        "Reading: name the strategy from strategy_label; quote the annualized Sharpe; quote peak-to-trough "
        "drawdown in bp as a positive loss using max_drawdown_bp_abs; quote underwater days. "
        "Decomposition: state the train and test windows in days; quote transaction cost in bp and the "
        "slippage multiplier; say that thresholds come only from the prior train window and that the position "
        "waits one day. Interpret Sharpe near zero as gains and setbacks cancelling, and a negative Sharpe as "
        "the rule losing money after costs on this path. "
        "Caveats: this is a research path on a published spread, not an account, not a live bid/offer, "
        "and not a statement about the next period."
    ),
}


def report_prompt(topic: str) -> str:
    """User message that asks for a figure-grounded exhibit note."""
    specific = REPORT_TOPIC_INSTRUCTIONS.get(
        topic,
        "Describe only the figures in the brief. Quote each number with its unit.",
    )
    return f"{REPORT_SHARED_RULES} Topic: {topic}. {specific}"


def summarize_facts(topic: str, facts: dict[str, object]) -> LLMResult:
    """Ask for a formal report that may use only the supplied figures."""
    return call_llm([{"role": "user", "content": report_prompt(topic)}], {"topic": topic, "facts": facts})


def check_groq_model_availability() -> tuple[bool, str | None]:
    _load_project_dotenv()
    api_key, key_source = resolve_groq_api_key()
    if not api_key:
        return True, None
    if key_source == "st.secrets":
        os.environ.setdefault("GROQ_API_KEY", api_key)

    model = os.getenv(LLM_MODEL_ENV, GROQ_MODEL_PRIMARY)
    try:
        client = _get_groq_client(api_key)
        models = client.models.list()
        model_ids = {getattr(item, "id", None) for item in getattr(models, "data", [])}
    except Exception as exc:
        return False, f"[startup] Could not verify Groq model availability: {exc}"

    if model not in model_ids:
        return False, (
            f"[startup] WARNING: configured Groq model '{model}' was not returned by "
            f"Groq models.list(); fallback model is '{GROQ_MODEL_FALLBACK}'."
        )
    return True, None


def _create_groq_chat_completion(client, model: str, input_messages: list[dict[str, str]]):
    return client.chat.completions.create(model=model, messages=input_messages)


def _is_groq_model_not_found(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code != 404:
        return False

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", {})
        if error.get("code") == "model_not_found":
            return True
        message = str(error.get("message", ""))
        if "model" in message.lower() and ("not found" in message.lower() or "does not exist" in message.lower()):
            return True

    message = str(exc).lower()
    return "model_not_found" in message or ("model" in message and ("not found" in message or "does not exist" in message))


def _get_groq_client(api_key: str):
    if GROQ_CLIENT_FACTORY is not None:
        return GROQ_CLIENT_FACTORY(api_key=api_key)
    from groq import Groq

    return Groq(api_key=api_key)
