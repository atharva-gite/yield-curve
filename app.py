from __future__ import annotations

import secrets
from html import escape
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
    ns_forward,
    nss_curve,
    nss_forward,
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
        /* Fonts: display + numeric/machine face */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

        :root {
            --color-bg: #0f172a;
            --color-text: #e2e8f0;
            --color-slate-600: #1e293b;
            --color-slate-700: #111827;
            --color-slate-500: #334155;
            --color-accent-sky: #38bdf8;
            --color-accent-emerald: #10b981;
            --color-amber: #f59e0b;
            --color-danger: #ef4444;
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 12px;
            --space-4: 16px;
            --space-5: 24px;
        }

        html, body, .stApp { background: var(--color-bg); color: var(--color-text); font-family: 'Space Grotesk', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
        [data-testid="stHeader"] { background: #0f172a; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #e2e8f0; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.18);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] h3 {
            margin-top: 0;
            font-size: 1rem;
            color: #f8fafc;
        }
        .status-card {
            border-radius: 8px;
            padding: var(--space-3);
            font-weight: 700;
            border: 1px solid var(--color-slate-500);
            background: var(--color-slate-600);
        }
        .guardrail-card {
            border-width: 2px;
            font-size: 1.02rem;
            margin-top: var(--space-3);
        }
        .status-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            color: var(--color-bg);
            background: var(--color-accent-emerald);
            margin-right: var(--space-2);
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
        .section-kicker {
            color: #94a3b8;
            font-size: 0.88rem;
            margin: -0.25rem 0 0.75rem;
        }
        .metric-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: var(--space-2);
            margin: var(--space-2) 0 var(--space-3);
        }
        .metric-card {
            background: var(--color-slate-700);
            border: 1px solid var(--color-slate-500);
            border-radius: 8px;
            padding: var(--space-3);
            min-height: 96px;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 600;
        }
        .metric-value {
            color: var(--color-text);
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.18;
            margin-top: 6px;
            overflow-wrap: anywhere;
            font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
            font-variant-numeric: tabular-nums;
        }
        .metric-sub {
            color: #cbd5e1;
            font-size: 0.82rem;
            margin-top: 6px;
        }
        .metric-emerald .metric-value { color: #10b981; }
        .metric-red .metric-value { color: #ef4444; }
        .metric-sky .metric-value { color: #38bdf8; }
        .metric-amber .metric-value { color: #f59e0b; }
        .mini-table {
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0 14px;
            font-size: 0.9rem;
        }
        .mini-table th, .mini-table td {
            border-bottom: 1px solid #334155;
            padding: 8px 6px;
            text-align: left;
        }
        .mini-table th { color: #94a3b8; font-weight: 700; }
        .mini-table td { color: #e2e8f0; }
        .spark-row {
            display: grid;
            grid-template-columns: 70px 1fr 52px;
            align-items: center;
            gap: 10px;
            margin: 8px 0;
            color: #cbd5e1;
            font-size: 0.86rem;
        }
        .guardrail-bar { width: 100%; height: 12px; background: var(--color-slate-700); border-radius: 999px; border: 1px solid var(--color-slate-500); position: relative; overflow: visible; }
        .guardrail-fill { height: 100%; background: linear-gradient(90deg, var(--color-accent-sky), var(--color-accent-emerald)); border-radius: 999px; }
        .guardrail-cutoff { position: absolute; top: -6px; left: 0; width: 2px; height: 24px; background: var(--color-amber); box-shadow: 0 0 6px rgba(245,158,11,0.5); }
        .spark-track {
            height: 8px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 999px;
            overflow: hidden;
        }
        .spark-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #10b981); }
        .fit-card-title {
            color: #f8fafc;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .chat-toolbar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .chat-shell {
            border: 1px solid #334155;
            border-radius: 8px;
            background: #1e293b;
            padding: 16px;
            margin-top: 8px;
        }
        .chat-empty {
            color: #94a3b8;
            border: 1px dashed #475569;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            background: #111827;
        }
        .bubble-row {
            display: flex;
            gap: 10px;
            margin: 12px 0;
            align-items: flex-start;
        }
        .bubble-row.user { justify-content: flex-end; }
        .bubble-row.user .bubble-avatar { order: 2; background: #475569; color: #f8fafc; }
        .bubble-avatar {
            width: 30px;
            height: 30px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 30px;
            background: #10b981;
            color: #052e2b;
            font-weight: 900;
            font-size: 0.82rem;
        }
        .bubble {
            max-width: min(760px, 82%);
            border-radius: 8px;
            padding: 11px 13px;
            line-height: 1.48;
            color: #e2e8f0;
            background: #0f172a;
            border: 1px solid #334155;
            box-shadow: 0 8px 22px rgba(2, 6, 23, 0.24);
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .bubble-row.user .bubble { background: #134e4a; border-color: #0f766e; color: #ecfeff; }
        .terminal-log {
            background: #020617;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px;
            color: #cbd5e1;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.86rem;
            line-height: 1.42;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .terminal-user { color: #38bdf8; }
        .terminal-assistant { color: #10b981; }
        div[role="radiogroup"] {
            gap: 6px;
        }
        div[role="radiogroup"] label {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 6px 10px;
            min-height: 36px;
        }
        button[kind="primary"] { background: #10b981; border-color: #10b981; }
        @media (max-width: 900px) {
            section.main div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
            .metric-card-grid { grid-template-columns: 1fr; }
            .spark-row { grid-template-columns: 58px 1fr 48px; }
            .bubble { max-width: calc(100% - 44px); }
        }
        .block-container { padding: var(--space-3) !important; }
        :focus-visible { outline: 3px solid rgba(56,189,248,0.18); outline-offset: 2px; border-radius: 6px; }
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


def bp(value: float) -> str:
    return f"{value * 10000:.2f} bp"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_cards(items: list[tuple[str, str, str, str]]) -> None:
    cards = []
    for label, value, sub, tone in items:
        tone_class = f" metric-{tone}" if tone else ""
        cards.append(
            "<div class='metric-card{tone_class}'>"
            "<div class='metric-label'>{label}</div>"
            "<div class='metric-value'>{value}</div>"
            "<div class='metric-sub'>{sub}</div>"
            "</div>".format(
                tone_class=tone_class,
                label=escape(label),
                value=escape(value),
                sub=escape(sub),
            )
        )
    st.markdown(f"<div class='metric-card-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def mini_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table class='mini-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def section_intro(title: str, detail: str | None = None) -> None:
    st.markdown(f"### {title}")
    if detail:
        st.markdown(f"<div class='section-kicker'>{escape(detail)}</div>", unsafe_allow_html=True)


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
    metric_cards(
        [
            ("NS Fit RMSE", bp(ns_fit.rmse), f"CV {bp(cv['ns']['rmse'])}", "sky"),
            ("NSS Fit RMSE", bp(nss_fit.rmse), f"CV {bp(cv['nss']['rmse'])}", "emerald"),
            ("Spline Fit RMSE", bp(spline_fit.rmse), f"CV {bp(cv['spline']['rmse'])}", "amber"),
        ]
    )


def fit_comparison_cards(ns_fit, nss_fit, spline_fit, cv) -> None:
    cols = st.columns(3)
    fits = [
        ("Nelson-Siegel", ns_fit, cv["ns"], "sky"),
        ("NSS", nss_fit, cv["nss"], "emerald"),
        ("Smoothing Spline", spline_fit, cv["spline"], "amber"),
    ]

    def make_sparkline(resids: np.ndarray) -> str:
        try:
            arr = np.asarray(resids, dtype=float)
            if arr.size < 2:
                return ""
            # normalize to [0,1]
            x = np.linspace(0, 100, arr.size)
            mn, mx = arr.min(), arr.max()
            denom = mx - mn if mx > mn else 1.0
            y = (arr - mn) / denom
            points = " ".join(f"{xp:.1f},{(100 - yp*80 + 10):.1f}" for xp, yp in zip(x, y))
            svg = f"<svg width='100' height='28' viewBox='0 0 100 28' preserveAspectRatio='none'><polyline points='{points}' fill='none' stroke='rgba(255,255,255,0.85)' stroke-width='1.4' stroke-linejoin='round' stroke-linecap='round'/></svg>"
            return svg
        except Exception:
            return ""

    for col, (name, fit, cv_result, tone) in zip(cols, fits):
        with col.container(border=True):
            st.markdown(f"<div class='fit-card-title'>{escape(name)}</div>", unsafe_allow_html=True)
            # Build a single card per model showing in-sample RMSE, CV RMSE and a tiny residual sparkline
            res_svg = make_sparkline(getattr(fit, 'residuals', np.array([])))
            card_html = (
                "<div class='metric-card'>"
                f"<div class='metric-label'>In-sample / Cross-val</div>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div style='flex:1'>"
                f"<div class='metric-value'>{escape(bp(fit.rmse))}</div>"
                f"<div class='metric-sub'>Cross-val {escape(bp(cv_result['rmse']))}</div>"
                f"</div>"
                f"<div style='width:108px;margin-left:12px'>{res_svg}</div>"
                f"</div>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)


def render_fitting_console(data: pd.DataFrame, session: dict) -> None:
    section_intro(
        "Dynamic Fitting Console",
        "Fit Nelson-Siegel, NSS, and smoothing spline curves against the selected Treasury date.",
    )
    presets = {
        "Latest": data.index.max().date(),
        "2019-08 inversion": pd.Timestamp("2019-08-14").date(),
        "2020-03 COVID shock": pd.Timestamp("2020-03-18").date(),
        "2023-07": pd.Timestamp("2023-07-03").date(),
    }
    with st.container(border=True):
        controls = st.columns([1, 1, 2])
        preset = controls[0].selectbox("Milestone", list(presets), help="Jump to a historically important curve date.")
        selected_date = controls[1].date_input("Curve date", value=presets[preset], min_value=data.index.min().date(), max_value=data.index.max().date())
        controls[2].markdown("<div class='section-kicker'>Nearest available FRED observation will be used for fitting.</div>", unsafe_allow_html=True)
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

    st.markdown(f"<span class='accent'>Using nearest available FRED date:</span> {curve_date.date()}", unsafe_allow_html=True)
    # Consolidated per-model cards (in-sample, cross-val, residual sparkline)
    fit_comparison_cards(ns_fit, nss_fit, spline_fit, cv)

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
    fig.add_trace(
        go.Scatter(
            x=dense,
            y=ns_forward(
                dense,
                ns_fit.params["beta0"],
                ns_fit.params["beta1"],
                ns_fit.params["beta2"],
                ns_fit.params["lambda"],
            )
            * 100,
            name="NS forward",
            line={"dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dense,
            y=nss_forward(dense, **nss_fit.params) * 100,
            name="NSS forward",
            line={"dash": "dot"},
        )
    )
    fig.update_layout(
        template=plotly_dark_theme(),
        title="Fitted yields and instantaneous forwards",
        xaxis_title="Maturity (years)",
        yaxis_title="Rate (%)",
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Forwards are the instantaneous forwards of the fitted constant-maturity curve, "
            "not a bootstrapped Treasury discount curve."
        )

    adf = adf_stationarity_check(nss_fit.residuals)
    css = "status-ok" if adf["passed"] else "status-bad"
    label = "Validated" if adf["passed"] else "Invalid Fit"
    # Guardrail: visual threshold showing ADF p-value relative to 0.05 cutoff
    pval = float(adf.get("p_value", 1.0))
    # Visual range 0 - 0.10 (10%) maps to 0-100%
    fill_pct = max(0.0, min(100.0, (pval / 0.10) * 100.0))
    cutoff_pct = (0.05 / 0.10) * 100.0
    guard_html = (
        f"<div class='status-card guardrail-card {css}'>"
        f"<span class='status-pill'>{escape(label)}</span>"
        f"<div style='display:flex;gap:12px;align-items:center;margin-top:8px'>"
        f"<div style='flex:1'>"
        f"<div class='guardrail-bar'>"
        f"<div class='guardrail-fill' style='width:{fill_pct:.1f}%'></div>"
        f"<div class='guardrail-cutoff' style='left:{cutoff_pct:.1f}%' title='ADF cutoff 0.05'></div>"
        f"</div>"
        f"</div>"
        f"<div style='min-width:120px;font-family:JetBrains Mono, monospace;color:var(--color-text);'>p = {pval:.4f}<div style='font-size:0.78rem;color:#94a3b8;margin-top:6px;'>cutoff 0.05</div></div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(guard_html, unsafe_allow_html=True)


def render_pca(data: pd.DataFrame, session: dict) -> None:
    section_intro(
        "PCA Regime Analysis",
        "Decompose daily yield changes into level, slope, and curvature factors.",
    )
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
    with left.container(border=True):
        st.markdown("### Loadings")
        st.plotly_chart(fig_load, use_container_width=True)

    fig_var = go.Figure(go.Bar(x=list(loadings.columns), y=np.asarray(pca["cumulative_explained_variance"]) * 100))
    fig_var.update_layout(template=plotly_dark_theme(), title="Cumulative Explained Variance", yaxis_title="%")
    with right.container(border=True):
        st.markdown("### Variance")
        st.plotly_chart(fig_var, use_container_width=True)

    scores = pca["scores"]
    fig_scores = go.Figure()
    for col in scores.columns:
        fig_scores.add_trace(go.Scatter(x=scores.index, y=scores[col], name=col))
    fig_scores.update_layout(template=plotly_dark_theme(), title="Historical Level / Slope / Curvature Scores")
    with st.container(border=True):
        st.markdown("### Historical Factor Scores")
        st.plotly_chart(fig_scores, use_container_width=True)


def render_backtester(data: pd.DataFrame, session: dict) -> None:
    section_intro(
        "Operational Backtester",
        "Walk-forward strategy testing with prior-window thresholds, transaction costs, and slippage.",
    )
    with st.container(border=True):
        st.markdown("### Strategy And Cost Model")
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
    metric_cards(
        [
            ("Annualized Sharpe", f"{result['sharpe']:.2f}", "Walk-forward risk-adjusted return", "emerald" if result["sharpe"] >= 0 else "red"),
            ("Max Drawdown", bp(result["max_drawdown"]), "Peak-to-trough PnL", "red"),
            ("Drawdown Duration", f"{result['max_drawdown_duration']} days", "Longest underwater stretch", "sky"),
        ]
    )
    fig = go.Figure(go.Scatter(x=result["cumulative_pnl"].index, y=result["cumulative_pnl"], name="Cumulative PnL"))
    fig.update_layout(template=plotly_dark_theme(), title="Walk-Forward Strategy PnL", yaxis_title="PnL")
    dd_fig = go.Figure(go.Scatter(x=result["drawdown"].index, y=result["drawdown"], name="Drawdown", fill="tozeroy"))
    dd_fig.update_layout(template=plotly_dark_theme(), title="Drawdown", yaxis_title="PnL below peak")
    chart_cols = st.columns(2)
    with chart_cols[0].container(border=True):
        st.markdown("### PnL")
        st.plotly_chart(fig, use_container_width=True)
    with chart_cols[1].container(border=True):
        st.markdown("### Drawdown")
        st.plotly_chart(dd_fig, use_container_width=True)
    with st.container(border=True):
        st.markdown("### Latest Trades")
        st.dataframe(result["trade_log"].tail(50), use_container_width=True)


def render_context_dashboard(session: dict) -> None:
    with st.expander("View session context", expanded=False):
        selected_date = str(session.get("selected_date", "Not fitted"))
        cv = session.get("curve_cv_rmse", {}) or {}
        pca_variance = session.get("pca_explained_variance", []) or []
        sharpe = session.get("backtest_sharpe")
        sharpe_value = "Not run" if sharpe is None else f"{float(sharpe):.2f}"
        sharpe_tone = "" if sharpe is None else ("emerald" if float(sharpe) >= 0 else "red")

        metric_cards(
            [
                ("Selected Date", selected_date, "Nearest available Treasury curve", "sky"),
                ("Backtest Sharpe", sharpe_value, "Latest walk-forward run", sharpe_tone),
            ]
        )

        rows = []
        for label, key in (("NS", "ns"), ("NSS", "nss"), ("Spline", "spline")):
            value = cv.get(key)
            rows.append([label, "Pending" if value is None else bp(float(value))])
        st.markdown(mini_table(["Model", "CV-RMSE"], rows), unsafe_allow_html=True)

        if pca_variance:
            spark_rows = []
            for idx, value in enumerate(pca_variance, start=1):
                width = max(0.0, min(100.0, float(value) * 100))
                spark_rows.append(
                    "<div class='spark-row'>"
                    f"<span>PC{idx}</span>"
                    "<span class='spark-track'>"
                    f"<span class='spark-fill' style='display:block;width:{width:.1f}%'></span>"
                    "</span>"
                    f"<span>{pct(float(value))}</span>"
                    "</div>"
                )
            st.markdown("".join(spark_rows), unsafe_allow_html=True)
        else:
            st.markdown("<div class='chat-empty'>PCA context will appear after the regime tab computes.</div>", unsafe_allow_html=True)


def render_modern_chat(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.markdown("<div class='chat-empty'>Ask a curve, PCA, or backtest question to start the analyst thread.</div>", unsafe_allow_html=True)
        return
    rows = []
    for message in messages:
        role = message["role"]
        avatar = "A" if role == "assistant" else "U"
        rows.append(
            f"<div class='bubble-row {escape(role)}'>"
            f"<div class='bubble-avatar'>{avatar}</div>"
            f"<div class='bubble'>{escape(message['content'])}</div>"
            "</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_terminal_chat(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.markdown("<div class='terminal-log'><span class='terminal-assistant'>assistant:</span> ready</div>", unsafe_allow_html=True)
        return
    lines = []
    for message in messages:
        role = message["role"]
        css = "terminal-assistant" if role == "assistant" else "terminal-user"
        prefix = f"&gt; {escape(role)}:"
        lines.append(f"<span class='{css}'>{prefix}</span> {escape(message['content'])}")
    st.markdown(f"<div class='terminal-log'>{'<br><br>'.join(lines)}</div>", unsafe_allow_html=True)


def render_ai_panel(session: dict) -> None:
    section_intro(
        "AI Quant Analyst",
        "Grounded answers using the active curve, PCA, and backtest session context.",
    )
    context_json = serialize_session_context(session)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("chat_style", "modern")
    st.session_state.setdefault("chat_style_icon", "▤" if st.session_state["chat_style"] == "terminal" else "▦")

    toolbar = st.columns([0.12, 0.88])
    selected_icon = toolbar[0].radio(
        "Chat style",
        ["▦", "▤"],
        horizontal=True,
        label_visibility="collapsed",
        key="chat_style_icon",
        help="Toggle modern or terminal chat rendering.",
    )
    st.session_state["chat_style"] = "terminal" if selected_icon == "▤" else "modern"
    current_style = st.session_state["chat_style"]
    toolbar[1].markdown(
        f"<div class='section-kicker'>Mode: {escape(current_style.title())} | "
        f"{len(st.session_state['chat_messages'])} messages in shared history</div>",
        unsafe_allow_html=True,
    )

    render_context_dashboard(session)

    with st.container(border=True):
        st.markdown("### Analyst Chat")
        if st.session_state["chat_style"] == "terminal":
            render_terminal_chat(st.session_state["chat_messages"])
        else:
            render_modern_chat(st.session_state["chat_messages"])

    prompt = st.chat_input("Ask the quant analyst")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
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
            st.rerun()
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
