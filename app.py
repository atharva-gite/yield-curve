from __future__ import annotations

import secrets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from logic import (
    MATURITIES,
    SERIES_IDS,
    adf_stationarity_check,
    call_llm,
    check_groq_model_availability,
    check_rate_limit,
    compute_pca,
    fit_ns,
    fit_nss,
    fit_smoothing_spline,
    get_yield_data,
    leave_one_maturity_cv,
    ns_curve,
    nss_curve,
    predict_smoothing_spline,
    resolve_groq_api_key,
    run_curve_spread_backtest,
    serialize_session_context,
)


st.set_page_config(page_title="Yield Curve Engine", layout="wide", page_icon="YC")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0f172a; color: #e2e8f0; }
        [data-testid="stHeader"] { background: #0f172a; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #e2e8f0; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px;
        }
        .status-card {
            border-radius: 8px;
            padding: 16px;
            font-weight: 700;
            border: 1px solid #334155;
            background: #1e293b;
        }
        .status-ok { color: #10b981; }
        .status-bad { color: #ef4444; }
        .status-bar {
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 12px;
            background: #1e293b;
            margin-bottom: 14px;
        }
        .accent { color: #38bdf8; }
        button[kind="primary"] { background: #10b981; border-color: #10b981; }
        @media (max-width: 800px) {
            section.main div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_dark_theme() -> dict:
    return {
        "layout": {
            "paper_bgcolor": "#0f172a",
            "plot_bgcolor": "#0f172a",
            "font": {"color": "#e2e8f0"},
            "xaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569"},
            "yaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569"},
            "colorway": ["#38bdf8", "#10b981", "#f59e0b", "#ef4444", "#a78bfa"],
        }
    }


@st.cache_data(show_spinner=False)
def load_data():
    result = get_yield_data()
    return result.data, result.source, result.warning


def _streamlit_secrets_available() -> bool:
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    return any(path.exists() for path in candidates)


def sync_streamlit_secrets_once() -> None:
    load_dotenv()
    if st.session_state.get("_secrets_checked"):
        return
    st.session_state["_secrets_checked"] = True

    if _streamlit_secrets_available():
        for key in ("FRED_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER", "GROQ_MODEL"):
            if os.getenv(key):
                continue
            try:
                value = st.secrets.get(key)
            except Exception:
                value = None
            if value:
                os.environ[key] = str(value)

    _, source = resolve_groq_api_key(lambda key: st.secrets.get(key) if _streamlit_secrets_available() else None)
    print(f"[startup] GROQ_API_KEY source: {source}", flush=True)
    _, model_warning = check_groq_model_availability()
    if model_warning:
        print(model_warning, flush=True)


def error_card(message: str) -> None:
    st.markdown(f"<div class='status-card status-bad'>{message}</div>", unsafe_allow_html=True)


def client_rate_limit_identifier() -> tuple[str, str]:
    try:
        headers = st.context.headers
        forwarded_for = headers.get("X-Forwarded-For")
        real_ip = headers.get("X-Real-IP")
        ip_address = (forwarded_for.split(",")[0].strip() if forwarded_for else real_ip)
        if ip_address:
            return f"ip:{ip_address}", "IP"
    except Exception:
        pass

    try:
        # Weaker fallback: this only limits a single active browser session, so a refresh or
        # new tab may reset it. It still prevents accidental rapid-fire usage in one session.
        session_id = st.session_state.setdefault("_rate_limit_session_id", secrets.token_hex(16))
        return f"session:{session_id}", "browser session"
    except Exception:
        return "global:shared", "shared fallback"


def format_reset_time(timestamp: float) -> str:
    return pd.Timestamp.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def closest_curve_date(data: pd.DataFrame, selected_date) -> pd.Timestamp:
    selected = pd.Timestamp(selected_date)
    return data.index[data.index.get_indexer([selected], method="nearest")[0]]


def fit_for_date(data: pd.DataFrame, selected_date) -> tuple[pd.Timestamp, np.ndarray, object, object, object, dict]:
    curve_date = closest_curve_date(data, selected_date)
    observed = data.loc[curve_date, SERIES_IDS].to_numpy(dtype=float)
    ns_fit = fit_ns(MATURITIES, observed)
    nss_fit = fit_nss(MATURITIES, observed)
    spline_fit = fit_smoothing_spline(MATURITIES, observed)
    cv = {
        "ns": leave_one_maturity_cv(MATURITIES, observed, "ns"),
        "nss": leave_one_maturity_cv(MATURITIES, observed, "nss"),
        "spline": leave_one_maturity_cv(MATURITIES, observed, "spline"),
    }
    return curve_date, observed, ns_fit, nss_fit, spline_fit, cv


def metric_row(ns_fit, nss_fit, spline_fit, cv) -> None:
    cols = st.columns(6)
    cols[0].metric("NS RMSE", f"{ns_fit.rmse * 10000:.2f} bp", help="Root mean squared fitting error.")
    cols[1].metric("NS CV RMSE", f"{cv['ns']['rmse'] * 10000:.2f} bp", help="Leave-one-maturity-out prediction error.")
    cols[2].metric("NSS RMSE", f"{nss_fit.rmse * 10000:.2f} bp")
    cols[3].metric("NSS CV RMSE", f"{cv['nss']['rmse'] * 10000:.2f} bp")
    cols[4].metric("Spline RMSE", f"{spline_fit.rmse * 10000:.2f} bp")
    cols[5].metric("Spline CV RMSE", f"{cv['spline']['rmse'] * 10000:.2f} bp")


def render_fitting_console(data: pd.DataFrame, session: dict) -> None:
    presets = {
        "Latest": data.index.max().date(),
        "2019-08 inversion": pd.Timestamp("2019-08-14").date(),
        "2020-03 COVID shock": pd.Timestamp("2020-03-18").date(),
        "2023-07": pd.Timestamp("2023-07-03").date(),
    }
    controls = st.columns([1, 1, 2])
    preset = controls[0].selectbox("Milestone", list(presets), help="Jump to a historically important curve date.")
    selected_date = controls[1].date_input("Curve date", value=presets[preset], min_value=data.index.min().date(), max_value=data.index.max().date())
    try:
        with st.spinner("Fitting NS, NSS, spline, and cross-validation diagnostics..."):
            curve_date, observed, ns_fit, nss_fit, spline_fit, cv = fit_for_date(data, selected_date)
    except Exception as exc:
        error_card(f"Curve fitting failed: {exc}")
        return
    session["selected_date"] = str(curve_date.date())
    session["ns_params"] = ns_fit.params
    session["nss_params"] = nss_fit.params
    session["spline_params"] = spline_fit.params
    session["curve_cv_rmse"] = {name: float(result["rmse"]) for name, result in cv.items()}

    controls[2].markdown(f"<span class='accent'>Using nearest available FRED date:</span> {curve_date.date()}", unsafe_allow_html=True)
    metric_row(ns_fit, nss_fit, spline_fit, cv)

    dense = np.linspace(MATURITIES.min(), MATURITIES.max(), 240)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MATURITIES, y=observed * 100, mode="markers", name="Observed"))
    fig.add_trace(
        go.Scatter(
            x=dense,
            y=ns_curve(
                dense,
                ns_fit.params["beta0"],
                ns_fit.params["beta1"],
                ns_fit.params["beta2"],
                ns_fit.params["lambda"],
            )
            * 100,
            name="NS",
        )
    )
    fig.add_trace(go.Scatter(x=dense, y=nss_curve(dense, **nss_fit.params) * 100, name="NSS"))
    fig.add_trace(
        go.Scatter(
            x=dense,
            y=predict_smoothing_spline(MATURITIES, observed, dense, spline_fit.params["smoothing_factor"]) * 100,
            name="Smoothing Spline",
        )
    )
    fig.update_layout(template=plotly_dark_theme(), title="Treasury Curve Fit", xaxis_title="Maturity (years)", yaxis_title="Yield (%)")
    st.plotly_chart(fig, use_container_width=True)

    adf = adf_stationarity_check(nss_fit.residuals)
    css = "status-ok" if adf["passed"] else "status-bad"
    label = "Validated" if adf["passed"] else "Invalid Fit"
    st.markdown(
        f"<div class='status-card {css}'>{label}: {adf['message']} (ADF p-value {adf['p_value']:.4f})</div>",
        unsafe_allow_html=True,
    )


def render_pca(data: pd.DataFrame, session: dict) -> None:
    with st.spinner("Computing PCA on daily yield changes..."):
        pca = compute_pca(data)
    session["pca_explained_variance"] = [float(x) for x in pca["explained_variance"]]
    if pca["diagnostic"]:
        st.warning(pca["diagnostic"])
    left, right = st.columns(2)
    loadings = pca["loadings"]
    fig_load = go.Figure()
    for col in loadings.columns:
        fig_load.add_trace(go.Scatter(x=MATURITIES, y=loadings[col], mode="lines+markers", name=col))
    fig_load.update_layout(template=plotly_dark_theme(), title="PCA Loadings on Yield Changes", xaxis_title="Maturity (years)")
    left.plotly_chart(fig_load, use_container_width=True)

    fig_var = go.Figure(go.Bar(x=list(loadings.columns), y=np.asarray(pca["cumulative_explained_variance"]) * 100))
    fig_var.update_layout(template=plotly_dark_theme(), title="Cumulative Explained Variance", yaxis_title="%")
    right.plotly_chart(fig_var, use_container_width=True)

    scores = pca["scores"]
    fig_scores = go.Figure()
    for col in scores.columns:
        fig_scores.add_trace(go.Scatter(x=scores.index, y=scores[col], name=col))
    fig_scores.update_layout(template=plotly_dark_theme(), title="Historical Level / Slope / Curvature Scores")
    st.plotly_chart(fig_scores, use_container_width=True)


def render_backtester(data: pd.DataFrame, session: dict) -> None:
    st.caption("Signals are generated with rolling walk-forward train/test windows so test thresholds use only prior in-sample data.")
    controls = st.columns(5)
    strategy = controls[0].selectbox(
        "Strategy",
        ["slope_mean_reversion", "slope_momentum", "butterfly_mean_reversion"],
        format_func=lambda value: {
            "slope_mean_reversion": "10Y/2Y mean reversion",
            "slope_momentum": "10Y/2Y momentum",
            "butterfly_mean_reversion": "2s5s10s fly mean reversion",
        }[value],
        help="All strategies use a one-day execution lag and identical cost model.",
    )
    cost_bp = controls[1].number_input("Transaction cost (bp)", value=1.0, min_value=0.0, step=0.25, help="Applied on position changes.")
    slippage = controls[2].slider("Slippage multiplier", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
    train_window = controls[3].number_input("Train window", value=252, min_value=80, max_value=1260, step=21)
    test_window = controls[4].number_input("Test window", value=21, min_value=5, max_value=126, step=1)
    with st.spinner("Running walk-forward backtest..."):
        result = run_curve_spread_backtest(
            data,
            transaction_cost_bp=cost_bp,
            slippage_multiplier=slippage,
            strategy=strategy,
            train_window=int(train_window),
            test_window=int(test_window),
        )
    session["backtest_sharpe"] = float(result["sharpe"])
    cols = st.columns(3)
    cols[0].metric("Annualized Sharpe", f"{result['sharpe']:.2f}")
    cols[1].metric("Max Drawdown", f"{result['max_drawdown'] * 10000:.2f} bp")
    cols[2].metric("Drawdown Duration", f"{result['max_drawdown_duration']} days")
    fig = go.Figure(go.Scatter(x=result["cumulative_pnl"].index, y=result["cumulative_pnl"], name="Cumulative PnL"))
    fig.update_layout(template=plotly_dark_theme(), title="Walk-Forward Strategy PnL", yaxis_title="PnL")
    st.plotly_chart(fig, use_container_width=True)
    dd_fig = go.Figure(go.Scatter(x=result["drawdown"].index, y=result["drawdown"], name="Drawdown", fill="tozeroy"))
    dd_fig.update_layout(template=plotly_dark_theme(), title="Drawdown", yaxis_title="PnL below peak")
    st.plotly_chart(dd_fig, use_container_width=True)
    st.dataframe(result["trade_log"].tail(50), use_container_width=True)


def render_ai_panel(session: dict) -> None:
    context_json = serialize_session_context(session)
    st.code(context_json, language="json")
    st.session_state.setdefault("chat_messages", [])
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    prompt = st.chat_input("Ask the quant analyst")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        identifier, identifier_mode = client_rate_limit_identifier()
        rate_limit = check_rate_limit(identifier)
        if not rate_limit.allowed:
            error_card(
                "Chat request limit reached. "
                f"Limit: {rate_limit.limit} requests per {rate_limit.window_seconds // 3600} hours per {identifier_mode}. "
                f"Try again after {format_reset_time(rate_limit.reset_at)}."
            )
            return
        st.caption(
            f"{rate_limit.remaining} chat requests remaining in this rolling "
            f"{rate_limit.window_seconds // 3600}-hour window for this {identifier_mode}."
        )
        with st.spinner("Calling the configured LLM provider with the active curve session context..."):
            result = call_llm(st.session_state["chat_messages"], context_json)
        if result.ok:
            st.session_state["chat_messages"].append({"role": "assistant", "content": result.content})
            with st.chat_message("assistant"):
                st.write(result.content)
        else:
            error_card(result.error or "LLM request failed.")


def main() -> None:
    inject_css()
    sync_streamlit_secrets_once()
    st.title("Yield Curve Construction & Term Structure Modeling Engine")
    try:
        with st.spinner("Loading Treasury yields from FRED or local cache..."):
            data, source, warning = load_data()
    except Exception as exc:
        error_card(f"Data load failed: {exc}")
        return
    st.markdown(
        f"<div class='status-bar'><span class='accent'>Data freshness:</span> {data.index.max().date()} | "
        f"<span class='accent'>Source:</span> {source} | Observations: {len(data):,}</div>",
        unsafe_allow_html=True,
    )
    if warning:
        st.warning(warning)
    session: dict = {}
    tabs = st.tabs(["Dynamic Fitting Console", "PCA Regime Analysis", "Operational Backtester", "AI Quant Analyst"])
    with tabs[0]:
        render_fitting_console(data, session)
    with tabs[1]:
        render_pca(data, session)
    with tabs[2]:
        render_backtester(data, session)
    with tabs[3]:
        render_ai_panel(session)


if __name__ == "__main__":
    main()
