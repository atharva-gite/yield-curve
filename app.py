from __future__ import annotations

import os
import secrets
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from logic import (
    MATURITIES,
    SERIES_IDS,
    adf_stationarity_check,
    bond_risk,
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
    rolling_ns_fit,
    run_curve_spread_backtest,
    serialize_session_context,
    summarize_facts,
)

st.set_page_config(page_title="Yield Curve", layout="wide", page_icon="YC")

THEMES = {
    "research": {
        "bg": "#f6f3ec",
        "panel": "#fffdf8",
        "ink": "#1c1917",
        "muted": "#6b6560",
        "rule": "#d6d1c7",
        "accent": "#1e3a5f",
        "positive": "#166534",
        "negative": "#991b1b",
        "grid": "#e6e1d6",
        "plot": "#fffdf8",
        "zero": "#d6d1c7",
        "fill": "rgba(153, 27, 27, 0.10)",
        "colorway": ["#1c1917", "#1e3a5f", "#3d5a40", "#a8a29e", "#5b7c99", "#6b8f71"],
    },
    "desk": {
        "bg": "#0c0e11",
        "panel": "#15181d",
        "ink": "#e7e2d6",
        "muted": "#9a9488",
        "rule": "#2a2e35",
        "accent": "#d4a017",
        "positive": "#8fbc8f",
        "negative": "#d4786a",
        "grid": "#2a2e35",
        "plot": "#101318",
        "zero": "#3a3f47",
        "fill": "rgba(212, 120, 106, 0.16)",
        "colorway": ["#e7e2d6", "#d4a017", "#c4b8a0", "#6f6a62", "#8a7340", "#9a8f7c"],
    },
}


def active_theme() -> str:
    theme = st.session_state.get("theme", "research")
    return theme if theme in THEMES else "research"


def inject_css(theme: str) -> None:
    tokens = THEMES[theme if theme in THEMES else "research"]
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {{
            --bg: {tokens["bg"]};
            --panel: {tokens["panel"]};
            --ink: {tokens["ink"]};
            --muted: {tokens["muted"]};
            --rule: {tokens["rule"]};
            --accent: {tokens["accent"]};
            --positive: {tokens["positive"]};
            --negative: {tokens["negative"]};
        }}

        html, body, .stApp, [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: var(--bg);
            color: var(--ink);
            font-family: "IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
        }}
        h1, h2, h3 {{ letter-spacing: 0; font-weight: 500; color: var(--ink); }}
        p, li, label, .stMarkdown {{ color: var(--ink); }}
        .kicker {{
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 0;
        }}
        .app-title {{
            color: var(--ink);
            font-size: 1.85rem;
            font-weight: 500;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin: 2px 0 6px;
        }}
        .meta-line {{
            color: var(--muted);
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 0.82rem;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel);
            border: 1px solid var(--rule);
            border-radius: 0;
            padding: 16px;
            box-shadow: none;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] h3 {{
            margin-top: 0;
            font-size: 1rem;
            font-weight: 500;
            color: var(--ink);
        }}
        .status-card, .status-line {{
            border: 1px solid var(--rule);
            background: var(--panel);
            padding: 10px 12px;
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 0.86rem;
        }}
        .status-ok {{ color: var(--positive); }}
        .status-bad {{ color: var(--negative); }}
        .section-kicker {{
            color: var(--muted);
            font-size: 0.9rem;
            margin: -0.2rem 0 0.85rem;
        }}
        .metric-card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 8px;
            margin: 8px 0 12px;
        }}
        .metric-card {{
            background: var(--panel);
            border: 1px solid var(--rule);
            padding: 12px;
            min-height: 88px;
        }}
        .metric-label {{
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .metric-value {{
            color: var(--ink);
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-variant-numeric: tabular-nums;
            font-size: 1.2rem;
            font-weight: 500;
            margin-top: 6px;
        }}
        [data-testid="stAppViewBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="column"] > [data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent;
            border: none;
            padding: 0;
            box-shadow: none;
        }}
        .metric-sub {{ color: var(--muted); font-size: 0.8rem; margin-top: 6px; white-space: nowrap; }}
        .metric-positive .metric-value {{ color: var(--positive); }}
        .metric-negative .metric-value {{ color: var(--negative); }}
        .mini-table {{ width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 0.9rem; }}
        .mini-table th, .mini-table td {{
            border-bottom: 1px solid var(--rule);
            padding: 8px 6px;
            text-align: left;
        }}
        .mini-table th {{ color: var(--muted); font-weight: 500; }}
        .mini-table td {{
            color: var(--ink);
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-variant-numeric: tabular-nums;
        }}
        .spark-row {{
            display: grid;
            grid-template-columns: 48px 1fr 52px;
            align-items: center;
            gap: 10px;
            margin: 8px 0;
            color: var(--muted);
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 0.8rem;
        }}
        .spark-track {{ height: 2px; background: var(--rule); }}
        .spark-fill {{ display: block; height: 2px; background: var(--accent); }}
        .fit-card-title {{ color: var(--ink); font-weight: 500; margin-bottom: 8px; }}
        .chat-empty {{
            color: var(--muted);
            border: 1px solid var(--rule);
            padding: 16px;
            text-align: left;
            background: var(--panel);
        }}
        .memo-note {{
            border-top: 1px solid var(--rule);
            padding: 12px 0;
        }}
        .memo-role {{
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .memo-body {{ color: var(--ink); line-height: 1.5; white-space: pre-wrap; }}
        .terminal-log {{
            background: var(--bg);
            border: 1px solid var(--rule);
            padding: 14px;
            color: var(--ink);
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 0.84rem;
            line-height: 1.45;
            white-space: pre-wrap;
        }}
        .terminal-user {{ color: var(--accent); }}
        .terminal-assistant {{ color: var(--ink); }}
        button[data-baseweb="tab"] {{
            background: transparent;
            color: var(--muted);
            border-radius: 0;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--ink);
            border-bottom: 1px solid var(--accent);
        }}
        div[role="radiogroup"] label {{
            background: transparent;
            border: 1px solid var(--rule);
            border-radius: 0;
            color: var(--ink);
            padding: 4px 10px;
        }}
        .stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea,
        [data-baseweb="select"] > div, [data-testid="stChatInput"] textarea {{
            background: var(--bg);
            color: var(--ink);
            border-color: var(--rule);
            border-radius: 0;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid var(--rule);
            background: var(--panel);
        }}
        button[kind="primary"], button[kind="secondary"] {{
            background: transparent;
            color: var(--ink);
            border: 1px solid var(--accent);
            border-radius: 0;
        }}
        @media (max-width: 900px) {{
            section.main div[data-testid="column"] {{ width: 100% !important; flex: 1 1 100% !important; }}
            .metric-card-grid {{ grid-template-columns: 1fr; }}
        }}
        .summary-note {{
            border-top: 1px solid var(--rule);
            margin-top: 10px;
            padding-top: 10px;
            color: var(--ink);
            line-height: 1.5;
        }}
        .block-container {{ padding-top: 1.2rem !important; max-width: 1180px; }}
        :focus-visible {{ outline: 1px solid var(--accent); outline-offset: 2px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, **updates) -> None:
    tokens = THEMES[active_theme()]
    axis = {
        "gridcolor": tokens["grid"],
        "zerolinecolor": tokens["zero"],
        "color": tokens["muted"],
        "title_font": {"color": tokens["muted"]},
    }
    fig.update_layout(
        template="none",
        paper_bgcolor=tokens["plot"],
        plot_bgcolor=tokens["plot"],
        font={"color": tokens["ink"], "family": "IBM Plex Sans, Helvetica, Arial, sans-serif"},
        xaxis=axis,
        yaxis=axis,
        colorway=tokens["colorway"],
        legend={"font": {"color": tokens["muted"], "size": 12}, "bgcolor": "rgba(0,0,0,0)"},
        margin={"t": 12, "r": 12, "b": 36, "l": 48},
        hoverlabel={"bgcolor": tokens["panel"], "bordercolor": tokens["rule"], "font": {"color": tokens["ink"]}},
    )
    if updates:
        fig.update_layout(**updates)


@st.cache_data(show_spinner="Fitting the month-end Nelson-Siegel path...")
def cached_rolling_ns(data: pd.DataFrame) -> pd.DataFrame:
    return rolling_ns_fit(data)


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
            f"<div class='metric-card{tone_class}'>"
            f"<div class='metric-label'>{escape(label)}</div>"
            f"<div class='metric-value'>{escape(value)}</div>"
            f"<div class='metric-sub'>{escape(sub)}</div>"
            "</div>"
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
            ("NS Fit RMSE", bp(ns_fit.rmse), f"CV {bp(cv['ns']['rmse'])}", ""),
            ("NSS Fit RMSE", bp(nss_fit.rmse), f"CV {bp(cv['nss']['rmse'])}", ""),
            ("Spline Fit RMSE", bp(spline_fit.rmse), f"CV {bp(cv['spline']['rmse'])}", ""),
        ]
    )


def fit_comparison_cards(ns_fit, nss_fit, spline_fit, cv) -> None:
    cols = st.columns(3)
    fits = [
        ("Nelson-Siegel", ns_fit, cv["ns"]),
        ("NSS", nss_fit, cv["nss"]),
        ("Smoothing Spline", spline_fit, cv["spline"]),
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
            svg = f"<svg width='100' height='28' viewBox='0 0 100 28' preserveAspectRatio='none'><polyline points='{points}' fill='none' stroke='currentColor' stroke-width='1.2' stroke-linejoin='round' stroke-linecap='round'/></svg>"
            return svg
        except Exception:
            return ""

    for col, (name, fit, cv_result) in zip(cols, fits):
        with col.container(border=True):
            st.markdown(f"<div class='fit-card-title'>{escape(name)}</div>", unsafe_allow_html=True)
            # Build a single card per model showing in-sample RMSE, CV RMSE and a tiny residual sparkline
            res_svg = make_sparkline(getattr(fit, 'residuals', np.array([])))
            card_html = (
                f"<div class='metric-label'>In-sample RMSE</div>"
                f"<div class='metric-value'>{escape(bp(fit.rmse))}</div>"
                f"<div class='metric-sub'>Cross-val {escape(bp(cv_result['rmse']))}</div>"
                f"<div style='margin-top:8px;color:var(--ink)'>{res_svg}</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)


def render_ns_stability(data: pd.DataFrame) -> None:
    path = cached_rolling_ns(data)
    if path.empty:
        return
    section_intro(
        "Nelson-Siegel stability",
        "Month-end refits. Lambda should move with the curve, not jump because the optimizer changed its mind.",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path.index, y=path["lambda"], name="λ"))
    fig.add_trace(go.Scatter(x=path.index, y=path["rmse"] * 1e4, name="RMSE (bp)", yaxis="y2"))
    tokens = THEMES[active_theme()]
    style_figure(
        fig,
        xaxis_title="Month end",
        yaxis_title="λ",
        yaxis2={
            "title": "RMSE (bp)",
            "overlaying": "y",
            "side": "right",
            "gridcolor": tokens["grid"],
            "zerolinecolor": tokens["zero"],
            "color": tokens["muted"],
        },
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)


def render_bond_risk(nss_params: dict) -> None:
    section_intro(
        "Bond risk",
        "Price one bond off the fitted NSS curve. DV01 is the price loss, per 100 face, for a 1bp rise.",
    )
    controls = st.columns(3)
    maturity = controls[0].number_input("Maturity (years)", min_value=0.25, max_value=30.0, value=10.0, step=0.25)
    coupon = controls[1].number_input("Coupon rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    frequency = controls[2].selectbox("Payments per year", [1, 2, 4], index=1)
    risk = bond_risk(
        float(maturity),
        float(coupon),
        100.0,
        int(frequency),
        lambda times: nss_curve(times, **nss_params),
    )
    rows = [
        ["Price", f"{risk['price']:.4f}"],
        ["Modified duration", f"{float(risk['modified_duration']):.3f}"],
        ["Parallel DV01", f"{risk['dv01']:.4f}"],
    ]
    for tenor, value in risk["key_rate_dv01"].items():
        rows.append([f"{tenor:.0f}y key-rate DV01", f"{float(value):.4f}"])
    st.markdown(mini_table(["Measure", "Per 100 face"], rows), unsafe_allow_html=True)


def render_fitting_console(data: pd.DataFrame, session: dict) -> None:
    section_intro(
        "Curve",
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

    st.markdown(f"<div class='meta-line'>Nearest FRED date {curve_date.date()}</div>", unsafe_allow_html=True)
    # Consolidated per-model cards (in-sample, cross-val, residual sparkline)
    fit_comparison_cards(ns_fit, nss_fit, spline_fit, cv)

    dense = np.linspace(MATURITIES.min(), MATURITIES.max(), 240)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MATURITIES, y=observed * 100, mode="markers", name="Observed", marker={"size": 7}))
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
            line={"width": 2},
        )
    )
    fig.add_trace(go.Scatter(x=dense, y=nss_curve(dense, **nss_fit.params) * 100, name="NSS", line={"width": 2}))
    fig.add_trace(
        go.Scatter(
            x=dense,
            y=predict_smoothing_spline(MATURITIES, observed, dense, spline_fit.params["smoothing_factor"]) * 100,
            name="Smoothing Spline",
            line={"width": 1},
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
            line={"dash": "dash", "width": 1.4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dense,
            y=nss_forward(dense, **nss_fit.params) * 100,
            name="NSS forward",
            line={"dash": "dash", "width": 1.4},
        )
    )
    style_figure(
        fig,
        xaxis_title="Maturity (years)",
        yaxis_title="Rate (%)",
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Forwards are the instantaneous forwards of the fitted constant-maturity curve, "
            "not a bootstrapped Treasury discount curve."
        )
    render_ns_stability(data)
    render_bond_risk(nss_fit.params)

    adf = adf_stationarity_check(nss_fit.residuals)
    pval = float(adf.get("p_value", 1.0))
    state = "stationary" if adf["passed"] else "not stationary"
    tone = "status-ok" if adf["passed"] else "status-bad"
    st.markdown(f"<div class='status-line {tone}'>p = {pval:.3f} · {state}</div>", unsafe_allow_html=True)
    render_summary(
        "curve",
        {
            "date": str(curve_date.date()),
            "yields_percent": {series: round(float(level) * 100, 3) for series, level in zip(SERIES_IDS, observed)},
            "ns_rmse_bp": round(ns_fit.rmse * 10000, 2),
            "nss_rmse_bp": round(nss_fit.rmse * 10000, 2),
            "spline_rmse_bp": round(spline_fit.rmse * 10000, 2),
            "ns_lambda": round(float(ns_fit.params["lambda"]), 3),
            "ns_long_run_percent": round(float(ns_fit.params["beta0"]) * 100, 3),
            "nss_residual_adf_p": round(pval, 4),
            "nss_residuals_stationary": bool(adf["passed"]),
            "caveat": "Constant-maturity yields. Forwards belong to the fitted curve, not a bootstrapped discount curve.",
        },
    )


def render_pca(data: pd.DataFrame, session: dict) -> None:
    section_intro(
        "Factors",
        "Decompose daily yield changes into level, slope, and curvature factors.",
    )
    with st.spinner("Computing PCA on daily yield changes..."):
        pca = compute_pca(data)
    session["pca_explained_variance"] = [float(x) for x in pca["explained_variance"]]
    if pca["diagnostic"]:
        st.warning(pca["diagnostic"])
    render_summary(
        "factors",
        {
            "explained_variance": [round(float(value), 4) for value in pca["explained_variance"]],
            "labels": [str(label) for label in pca["loadings"].columns],
            "diagnostic": str(pca["diagnostic"] or "The first three components cover the usual share of daily changes."),
        },
    )
    left, right = st.columns(2)
    loadings = pca["loadings"]
    fig_load = go.Figure()
    for col in loadings.columns:
        fig_load.add_trace(go.Scatter(x=MATURITIES, y=loadings[col], mode="lines+markers", name=col))
    style_figure(fig_load, xaxis_title="Maturity (years)")
    with left.container(border=True):
        st.markdown("### Loadings")
        st.plotly_chart(fig_load, use_container_width=True)

    fig_var = go.Figure(
        go.Bar(
            x=list(loadings.columns),
            y=np.asarray(pca["cumulative_explained_variance"]) * 100,
            marker_color=THEMES[active_theme()]["accent"],
        )
    )
    style_figure(fig_var, yaxis_title="%")
    with right.container(border=True):
        st.markdown("### Variance")
        st.plotly_chart(fig_var, use_container_width=True)

    scores = pca["scores"]
    fig_scores = go.Figure()
    for col in scores.columns:
        fig_scores.add_trace(go.Scatter(x=scores.index, y=scores[col], name=col))
    style_figure(fig_scores)
    with st.container(border=True):
        st.markdown("### Historical Factor Scores")
        st.plotly_chart(fig_scores, use_container_width=True)


def render_backtester(data: pd.DataFrame, session: dict) -> None:
    section_intro(
        "Backtest",
        "Walk-forward strategy testing with prior-window thresholds, transaction costs, and slippage.",
    )
    with st.container(border=True):
        st.markdown("### Strategy")
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
            ("Annualized Sharpe", f"{result['sharpe']:.2f}", "Walk-forward risk-adjusted return", "positive" if result["sharpe"] >= 0 else "negative"),
            ("Max Drawdown", bp(result["max_drawdown"]), "Peak-to-trough PnL", "negative"),
            ("Drawdown Duration", f"{result['max_drawdown_duration']} days", "Longest underwater stretch", ""),
        ]
    )
    render_summary(
        "backtest",
        {
            "strategy": strategy,
            "sharpe": round(float(result["sharpe"]), 2),
            "max_drawdown_bp": round(float(result["max_drawdown"]) * 10000, 2),
            "max_drawdown_duration_days": int(result["max_drawdown_duration"]),
            "train_window_days": int(train_window),
            "test_window_days": int(test_window),
            "transaction_cost_bp": float(cost_bp),
            "slippage_multiplier": float(slippage),
        },
    )
    fig = go.Figure(go.Scatter(x=result["cumulative_pnl"].index, y=result["cumulative_pnl"], name="Cumulative PnL"))
    style_figure(fig, yaxis_title="PnL")
    dd_fig = go.Figure(
        go.Scatter(
            x=result["drawdown"].index,
            y=result["drawdown"],
            name="Drawdown",
            fill="tozeroy",
            fillcolor=THEMES[active_theme()]["fill"],
        )
    )
    style_figure(dd_fig, yaxis_title="PnL below peak")
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
        sharpe_tone = "" if sharpe is None else ("positive" if float(sharpe) >= 0 else "negative")

        metric_cards(
            [
                ("Selected Date", selected_date, "Nearest available Treasury curve", ""),
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
            st.markdown("<div class='chat-empty'>Factor context appears after the Factors tab runs.</div>", unsafe_allow_html=True)


def render_memo_chat(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.markdown("<div class='chat-empty'>Ask about the curve, the factors, or the backtest.</div>", unsafe_allow_html=True)
        return
    rows = []
    for message in messages:
        role = "Analyst" if message["role"] == "assistant" else "You"
        rows.append(
            "<div class='memo-note'>"
            f"<div class='memo-role'>{role}</div>"
            f"<div class='memo-body'>{escape(message['content'])}</div>"
            "</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_terminal_chat(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.markdown("<div class='terminal-log'><span class='terminal-assistant'>analyst</span> ready</div>", unsafe_allow_html=True)
        return
    lines = []
    for message in messages:
        role = "analyst" if message["role"] == "assistant" else "you"
        css = "terminal-assistant" if message["role"] == "assistant" else "terminal-user"
        lines.append(f"<span class='{css}'>{role}</span>  {escape(message['content'])}")
    st.markdown(f"<div class='terminal-log'>{'<br><br>'.join(lines)}</div>", unsafe_allow_html=True)


def render_ai_panel(session: dict) -> None:
    section_intro(
        "Analyst",
        "Grounded answers using the active curve, PCA, and backtest session context.",
    )
    context_json = serialize_session_context(session)
    st.session_state.setdefault("chat_messages", [])
    render_context_dashboard(session)

    with st.container(border=True):
        if active_theme() == "desk":
            render_terminal_chat(st.session_state["chat_messages"])
        else:
            render_memo_chat(st.session_state["chat_messages"])

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


def render_guide() -> None:
    st.markdown(
        """
**What this is.** A desk view of the U.S. Treasury curve: a fit, the forwards implied by that fit, a factor split of daily changes, and a walk-forward spread backtest.

**The data.** FRED constant-maturity yields (`DGS1MO` through `DGS30`). They are par-style yields, not a discount curve built from bills and coupons. Prints are in percent on the charts and in decimal inside the models.

**Curve.** Nelson-Siegel, Nelson-Siegel-Svensson, and a smoothing spline are fit to one date. The dashed lines are instantaneous forwards of those fitted curves. RMSE is the in-sample miss. Cross-val drops one maturity and refits. The p-value is an ADF test on the NSS residuals.

**Factors.** Principal components of daily yield *changes*, not of the yield levels. The loadings are labeled level, slope, or curvature from their shape.

**Backtest.** A 10s/2s slope or a 2s5s10s butterfly. Each test window uses thresholds from the prior train window only, then waits one day to trade. Sharpe, drawdown, and time underwater are on that path.

**Risk.** The bond table prices one coupon bond off the NSS curve. DV01 is the price loss per 100 face for a 1bp rise. Key-rate DV01 bumps only the 2y, 5y, 10y, or 30y region.

**Desk note.** The note on each tab is written from the figures already on screen. It is not sent until you ask, and it does not see anything outside that tab.
        """
    )


def render_summary(topic: str, facts: dict[str, object]) -> None:
    st.markdown("<div class='metric-label'>Desk note</div>", unsafe_allow_html=True)
    api_key, _source = resolve_groq_api_key()
    if not api_key:
        st.markdown(
            "<div class='section-kicker'>Add GROQ_API_KEY to write a note from the figures on this tab. Nothing is sent until you ask.</div>",
            unsafe_allow_html=True,
        )
        return
    cache_key = f"summary::{topic}::{serialize_session_context(facts)}"
    if st.button("Write note", key=f"write-note-{topic}"):
        identifier, identifier_mode = client_rate_limit_identifier()
        rate_limit = check_rate_limit(identifier)
        if not rate_limit.allowed:
            error_card(
                "Note limit reached. "
                f"Limit: {rate_limit.limit} requests per {rate_limit.window_seconds // 3600} hours per {identifier_mode}. "
                f"Try again after {format_reset_time(rate_limit.reset_at)}."
            )
            return
        with st.spinner("Writing the note from the figures on this tab..."):
            st.session_state[cache_key] = summarize_facts(topic, facts)
    result = st.session_state.get(cache_key)
    if result is None:
        return
    if result.ok:
        st.markdown(f"<div class='summary-note'>{escape(result.content)}</div>", unsafe_allow_html=True)
    else:
        error_card(result.error or "The note could not be written.")


def main() -> None:
    st.session_state.setdefault("theme", "research")
    if st.session_state["theme"] not in THEMES:
        st.session_state["theme"] = "research"
    inject_css(st.session_state["theme"])
    sync_streamlit_secrets_once()
    try:
        with st.spinner("Loading Treasury yields from FRED or local cache..."):
            data, source, warning = load_data()
    except Exception as exc:
        error_card(f"Data load failed: {exc}")
        return
    title_col, switch_col = st.columns([4, 1.3])
    with title_col:
        st.markdown(
            "<div class='kicker'>U.S. Treasury</div>"
            "<div class='app-title'>Yield Curve</div>"
            f"<div class='meta-line'>As of {data.index.max().date()} · {escape(str(source))} · {len(data):,} observations</div>",
            unsafe_allow_html=True,
        )
    with switch_col:
        st.radio(
            "Theme",
            ["research", "desk"],
            horizontal=True,
            format_func=lambda value: "Research" if value == "research" else "Desk",
            label_visibility="visible",
            key="theme",
        )
        with st.popover("Help", help="What the data and each tab measure"):
            render_guide()
    if warning:
        st.warning(warning)
    session: dict = {}
    tabs = st.tabs(["Curve", "Factors", "Backtest", "Analyst"])
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
