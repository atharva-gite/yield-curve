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
        "bg": "#F6F4EF",
        "panel": "#FFFFFF",
        "ink": "#252C28",
        "muted": "#77766E",
        "rule": "#E5E1D8",
        "accent": "#526B58",
        "on_accent": "#FFFFFF",
        "hover": "#F5F3EE",
        "positive": "#526B58",
        "negative": "#991b1b",
        "grid": "#E5E1D8",
        "plot": "#FFFFFF",
        "zero": "#E5E1D8",
        "fill": "rgba(82, 107, 88, 0.10)",
        "colorway": ["#252C28", "#526B58", "#3d5a40", "#77766E", "#5b7c99", "#6b8f71"],
    },
    "desk": {
        "bg": "#0c0e11",
        "panel": "#15181d",
        "ink": "#e7e2d6",
        "muted": "#9a9488",
        "rule": "#2a2e35",
        "accent": "#8fbc8f",
        "on_accent": "#0c0e11",
        "hover": "#1c2026",
        "positive": "#8fbc8f",
        "negative": "#d4786a",
        "grid": "#2a2e35",
        "plot": "#101318",
        "zero": "#3a3f47",
        "fill": "rgba(212, 120, 106, 0.16)",
        "colorway": ["#e7e2d6", "#8fbc8f", "#c4b8a0", "#6f6a62", "#8a7340", "#9a8f7c"],
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
            --on-accent: {tokens["on_accent"]};
            --hover: {tokens["hover"]};
            --positive: {tokens["positive"]};
            --negative: {tokens["negative"]};
        }}
        *, *::before, *::after {{ box-sizing: border-box; }}
        .stApp {{ overflow-x: hidden; }}

        html, body, .stApp, [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: var(--bg);
            color: var(--ink);
            font-family: "IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
        }}
        h1, h2, h3 {{ letter-spacing: 0; font-weight: 500; color: var(--ink); }}
        p, li, label, .stMarkdown {{ color: var(--ink); }}
        .kicker {{
            color: var(--muted);
            font-size: 12px;
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
            border-radius: 8px;
            padding: 24px;
            box-shadow: none;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] h3 {{
            margin: 0 0 24px;
            font-size: 24px;
            font-weight: 600;
            line-height: 1.2;
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
        .stButton > button,
        button[kind="primary"],
        button[kind="secondary"] {{
            appearance: none !important;
            -webkit-appearance: none !important;
            background: transparent !important;
            color: var(--ink) !important;
            border: 1px solid var(--rule) !important;
            border-radius: 6px !important;
            box-shadow: none !important;
            padding: 8px !important;
            min-height: 0 !important;
            height: auto !important;
            line-height: 1.5 !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }}
        .stButton > button:hover {{
            background: var(--hover) !important;
            border-color: var(--rule) !important;
        }}
        .stButton > button[kind="primary"] {{
            background: var(--accent) !important;
            color: var(--on-accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 6px !important;
            padding: 10px 16px !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 12px !important;
            font-weight: 600 !important;
            min-height: 40px !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: var(--accent) !important;
            filter: brightness(0.92);
            border-color: var(--accent) !important;
        }}
        .element-container:has(.help-control) {{
            display: none;
        }}
        .element-container:has(.help-control) ~ .element-container .stButton {{
            display: flex;
            justify-content: flex-end;
            margin: 0;
        }}
        .element-container:has(.help-control) ~ .element-container .stButton > button {{
            width: auto !important;
            min-width: 0 !important;
            height: 32px !important;
            padding: 0 12px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.08em !important;
            line-height: 1 !important;
        }}
        .header-controls {{
            text-align: right;
            margin: 0 0 8px;
        }}
        div[data-testid="column"]:has(.header-controls) [data-testid="stHorizontalBlock"] {{
            justify-content: flex-end;
            align-items: center;
            flex-wrap: nowrap !important;
            gap: 8px !important;
        }}
        div[data-testid="column"]:has(.header-controls) [data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
            width: auto !important;
            flex: 0 0 auto !important;
            min-width: 0;
        }}
        div[data-testid="column"]:has(.header-controls) [data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {{
            min-width: 88px;
        }}
        .report-copy {{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
            margin: 4px 0 0;
            max-width: 52ch;
        }}
        .element-container:has(.report-card-mark) {{
            display: none;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-card-mark) > div[data-testid="stVerticalBlock"] {{
            gap: 0;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-card-mark) [data-testid="stHorizontalBlock"] {{
            align-items: center;
            gap: 16px !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-card-mark) h3 {{
            margin: 0;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-card-mark) [data-testid="column"]:last-child .stButton {{
            display: flex;
            justify-content: flex-end;
            margin: 0;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-card-mark) [data-testid="column"]:last-child .stButton > button {{
            min-width: 148px;
            margin: 0;
        }}
        .report {{
            border-top: 1px solid var(--rule);
            margin-top: 16px;
            padding-top: 16px;
        }}
        .stButton > button:focus,
        .stButton > button:focus-visible,
        button[kind="primary"]:focus-visible,
        button[kind="secondary"]:focus-visible {{
            outline: 2px solid var(--accent) !important;
            outline-offset: 2px !important;
            box-shadow: none !important;
        }}
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {{
            background: transparent !important;
            border: 0 !important;
            outline: none !important;
            box-shadow: none !important;
            color: inherit !important;
            margin: 0 !important;
            padding: 0 !important;
            font-size: inherit !important;
            line-height: inherit !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"]:has(.analyst-prompts) {{
            gap: 0 !important;
        }}
        .element-container:has(.analyst-prompts) {{
            display: none;
        }}
        .element-container:has(.analyst-prompts) ~ .element-container {{
            margin: 0;
        }}
        .element-container:has(.analyst-prompts) ~ .element-container .stButton {{
            width: 100%;
            margin: 0 0 14px;
        }}
        .element-container:has(.analyst-prompts) ~ .element-container:last-of-type .stButton {{
            margin-bottom: 0;
        }}
        .element-container:has(.analyst-prompts) ~ .element-container .stButton > button {{
            width: 100%;
            display: flex !important;
            align-items: flex-start;
            justify-content: flex-start;
            text-align: left !important;
            white-space: normal !important;
            overflow: visible;
            text-overflow: unset;
            background: var(--panel) !important;
            border: 1px solid var(--rule) !important;
            border-radius: 8px !important;
            padding: 16px !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            line-height: 1.5 !important;
        }}
        .element-container:has(.analyst-prompts) ~ .element-container .stButton > button:hover {{
            background: var(--hover) !important;
        }}
        @media (max-width: 900px) {{
            section.main div[data-testid="column"] {{ width: 100% !important; flex: 1 1 100% !important; }}
            .metric-card-grid {{ grid-template-columns: 1fr; }}
            .element-container:has(.analyst-prompts) ~ .element-container .stButton > button {{
                max-width: 100%;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-card-mark) [data-testid="column"]:last-child .stButton {{
                justify-content: flex-start;
                margin-top: 12px;
            }}
        }}
        .summary-note, .report-body {{
            color: var(--ink);
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        .guide-prose {{
            max-width: 74ch;
            color: var(--ink);
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 0 0 0.4rem;
        }}
        .guide-caption {{
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 8px 0 20px;
        }}
        .analyst-empty {{
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 8px;
            min-height: 0;
            margin: 0 0 24px;
        }}
        .analyst-empty-title {{
            color: var(--ink);
            font-size: 24px;
            font-weight: 600;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }}
        .analyst-empty p {{
            color: var(--muted);
            font-size: 15px;
            line-height: 1.5;
            margin: 0;
        }}
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
        .block-container {{ padding-top: 24px !important; max-width: 1180px; overflow-x: hidden; }}
        :focus {{ outline: none; }}
        button:focus-visible,
        [role="tab"]:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="radio"]:focus-visible {{
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }}
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
    session["yields_percent"] = {
        series: round(float(level) * 100, 3) for series, level in zip(SERIES_IDS, observed)
    }

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
    yields_percent = {series: round(float(level) * 100, 3) for series, level in zip(SERIES_IDS, observed)}
    y3m = yields_percent["DGS3MO"]
    y2 = yields_percent["DGS2"]
    y10 = yields_percent["DGS10"]
    y30 = yields_percent["DGS30"]
    render_summary(
        "curve",
        {
            "date": str(curve_date.date()),
            "three_month_percent": y3m,
            "two_year_percent": y2,
            "ten_year_percent": y10,
            "thirty_year_percent": y30,
            "slope_2s10s_bp": round((y10 - y2) * 100, 1),
            "slope_10s30s_bp": round((y30 - y10) * 100, 1),
            "yields_percent": yields_percent,
            "ns_rmse_bp": round(ns_fit.rmse * 10000, 2),
            "nss_rmse_bp": round(nss_fit.rmse * 10000, 2),
            "spline_rmse_bp": round(spline_fit.rmse * 10000, 2),
            "ns_cv_rmse_bp": round(float(cv["ns"]["rmse"]) * 10000, 2),
            "nss_cv_rmse_bp": round(float(cv["nss"]["rmse"]) * 10000, 2),
            "spline_cv_rmse_bp": round(float(cv["spline"]["rmse"]) * 10000, 2),
            "ns_lambda": round(float(ns_fit.params["lambda"]), 3),
            "ns_long_run_percent": round(float(ns_fit.params["beta0"]) * 100, 3),
            "nss_long_run_percent": round(float(nss_fit.params["beta0"]) * 100, 3),
            "nss_lambda1": round(float(nss_fit.params["lambda1"]), 3),
            "nss_residual_adf_p": round(pval, 4),
            "nss_residuals_stationary": bool(adf["passed"]),
            "units": {
                "yields": "percent",
                "rmse": "basis points",
                "slopes": "basis points",
            },
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
    explained = [float(value) for value in pca["explained_variance"]]
    render_summary(
        "factors",
        {
            "labels": [str(label) for label in pca["loadings"].columns],
            "explained_pct": [round(value * 100, 1) for value in explained],
            "explained_variance": [round(value, 4) for value in explained],
            "combined_first_three_pct": round(float(np.sum(explained[:3])) * 100, 1),
            "input": "daily yield changes, not yield levels",
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
    session["backtest_max_drawdown"] = float(result["max_drawdown"])
    session["backtest_strategy"] = strategy
    metric_cards(
        [
            ("Annualized Sharpe", f"{result['sharpe']:.2f}", "Walk-forward risk-adjusted return", "positive" if result["sharpe"] >= 0 else "negative"),
            ("Max Drawdown", bp(result["max_drawdown"]), "Peak-to-trough PnL", "negative"),
            ("Drawdown Duration", f"{result['max_drawdown_duration']} days", "Longest underwater stretch", ""),
        ]
    )
    strategy_labels = {
        "slope_mean_reversion": "10-year minus 2-year mean reversion",
        "slope_momentum": "10-year minus 2-year momentum",
        "butterfly_mean_reversion": "2s5s10s butterfly mean reversion",
    }
    drawdown_bp = round(float(result["max_drawdown"]) * 10000, 2)
    render_summary(
        "backtest",
        {
            "strategy": strategy,
            "strategy_label": strategy_labels.get(strategy, strategy),
            "sharpe": round(float(result["sharpe"]), 2),
            "max_drawdown_bp": drawdown_bp,
            "max_drawdown_bp_abs": abs(drawdown_bp),
            "max_drawdown_duration_days": int(result["max_drawdown_duration"]),
            "train_window_days": int(train_window),
            "test_window_days": int(test_window),
            "transaction_cost_bp": float(cost_bp),
            "slippage_multiplier": float(slippage),
            "execution": "one-day lag after the signal; costs charged on position changes",
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


ANALYST_PROMPTS = (
    "Read the selected curve: the level, the 2s10s slope, and which fit is tightest.",
    "Explain the factor shares, and what a level, slope, or curvature move means in this brief.",
    "Interpret the walk-forward Sharpe and the drawdown for the strategy in the brief.",
)


def render_analyst_brief(session: dict) -> None:
    yields = session.get("yields_percent") or {}
    cv = session.get("curve_cv_rmse") or {}
    pca_variance = session.get("pca_explained_variance") or []
    sharpe = session.get("backtest_sharpe")
    drawdown = session.get("backtest_max_drawdown")
    y2 = yields.get("DGS2")
    y10 = yields.get("DGS10")
    slope = None if y2 is None or y10 is None else (float(y10) - float(y2)) * 100
    strategy_labels = {
        "slope_mean_reversion": "10Y/2Y mean reversion",
        "slope_momentum": "10Y/2Y momentum",
        "butterfly_mean_reversion": "2s5s10s fly",
    }
    strategy_name = strategy_labels.get(str(session.get("backtest_strategy", "")), "Latest rule")
    sharpe_value = "Pending" if sharpe is None else f"{float(sharpe):.2f}"
    sharpe_tone = "" if sharpe is None else ("positive" if float(sharpe) >= 0 else "negative")
    slope_value = "Pending" if slope is None else f"{slope:.1f} bp"
    metric_cards(
        [
            ("Curve date", str(session.get("selected_date", "Pending")), "Date the fits use", ""),
            ("2s10s slope", slope_value, "10-year yield minus 2-year yield", ""),
            ("Walk-forward Sharpe", sharpe_value, strategy_name, sharpe_tone),
        ]
    )
    rows = []
    for label, key in (("Nelson-Siegel", "ns"), ("Svensson", "nss"), ("Spline", "spline")):
        value = cv.get(key)
        rows.append([label, "Pending" if value is None else bp(float(value))])
    st.markdown(mini_table(["Fit", "Cross-val error"], rows), unsafe_allow_html=True)
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
    drawdown_text = "pending" if drawdown is None else bp(abs(float(drawdown)))
    st.markdown(
        "<div class='guide-caption'>"
        "This brief is the entire source the analyst may use. "
        f"The latest peak-to-trough result on the backtest is {escape(drawdown_text)}. "
        "A reply that needs a figure missing from this brief should say so."
        "</div>",
        unsafe_allow_html=True,
    )


def render_memo_chat(messages: list[dict[str, str]]) -> None:
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
    lines = []
    for message in messages:
        role = "analyst" if message["role"] == "assistant" else "you"
        css = "terminal-assistant" if message["role"] == "assistant" else "terminal-user"
        lines.append(f"<span class='{css}'>{role}</span>  {escape(message['content'])}")
    st.markdown(f"<div class='terminal-log'>{'<br><br>'.join(lines)}</div>", unsafe_allow_html=True)


def _ask_analyst(prompt: str, context_json: str) -> None:
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    identifier, identifier_mode = client_rate_limit_identifier()
    rate_limit = check_rate_limit(identifier)
    if not rate_limit.allowed:
        error_card(
            "Question limit reached. "
            f"Limit: {rate_limit.limit} requests per {rate_limit.window_seconds // 3600} hours per {identifier_mode}. "
            f"Try again after {format_reset_time(rate_limit.reset_at)}."
        )
        return
    with st.spinner("Writing the reply from the brief..."):
        result = call_llm(st.session_state["chat_messages"], context_json)
    if result.ok:
        st.session_state["chat_messages"].append({"role": "assistant", "content": result.content})
        st.rerun()
    else:
        error_card(result.error or "The reply could not be written.")


def render_ai_panel(session: dict) -> None:
    section_intro(
        "Analyst",
        "A conversation limited to the brief on the left: the fitted curve, the factor shares, and the backtest.",
    )
    context_json = serialize_session_context(session)
    st.session_state.setdefault("chat_messages", [])
    messages = st.session_state["chat_messages"]
    brief_col, thread_col = st.columns([1, 1.25], gap="large")
    with brief_col, st.container(border=True):
        st.markdown("### Brief")
        render_analyst_brief(session)
    with thread_col, st.container(border=True):
        st.markdown("### Correspondence")
        if messages:
            if active_theme() == "desk":
                render_terminal_chat(messages)
            else:
                render_memo_chat(messages)
        else:
            st.markdown(
                "<div class='analyst-empty'>"
                "<div class='kicker'>Awaiting a question</div>"
                "<div class='analyst-empty-title'>The brief is ready.</div>"
                "<p>Ask about the slope, the factors, or the backtest. "
                "The reply stays inside those figures and will name a gap if a number is missing.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div class='analyst-prompts'></div>", unsafe_allow_html=True)
        for index, suggestion in enumerate(ANALYST_PROMPTS):
            if st.button(suggestion, key=f"analyst-prompt-{index}", use_container_width=True):
                _ask_analyst(suggestion, context_json)
    prompt = st.chat_input("Ask about the brief")
    if prompt:
        _ask_analyst(prompt, context_json)


def render_summary(topic: str, facts: dict[str, object]) -> None:
    api_key, _source = resolve_groq_api_key()
    copy = (
        "A formal note from the figures already on this tab. Nothing is sent until you ask."
        if api_key
        else "Add GROQ_API_KEY to generate an AI report from the figures on this tab."
    )
    with st.container(border=True):
        st.markdown("<div class='report-card-mark'></div>", unsafe_allow_html=True)
        copy_col, btn_col = st.columns([3.4, 1], vertical_alignment="center", gap="medium")
        with copy_col:
            st.markdown("### AI Report")
            st.markdown(f"<p class='report-copy'>{escape(copy)}</p>", unsafe_allow_html=True)
        with btn_col:
            asked = st.button(
                "Generate",
                key=f"ai-report-{topic}",
                type="primary",
                disabled=not bool(api_key),
            )
        if not api_key:
            return
        cache_key = f"summary::{topic}::{serialize_session_context(facts)}"
        if asked:
            identifier, identifier_mode = client_rate_limit_identifier()
            rate_limit = check_rate_limit(identifier)
            if not rate_limit.allowed:
                error_card(
                    "AI Report limit reached. "
                    f"Limit: {rate_limit.limit} requests per {rate_limit.window_seconds // 3600} hours per {identifier_mode}. "
                    f"Try again after {format_reset_time(rate_limit.reset_at)}."
                )
                return
            with st.spinner("Preparing the AI report from the figures on this tab..."):
                st.session_state[cache_key] = summarize_facts(topic, facts)
        result = st.session_state.get(cache_key)
        if result is None:
            return
        if result.ok:
            st.markdown(
                f"<div class='report'><div class='report-body'>{escape(result.content)}</div></div>",
                unsafe_allow_html=True,
            )
        else:
            error_card(result.error or "The AI report could not be written.")


TENOR_LABELS = (
    "1 month",
    "3 months",
    "6 months",
    "1 year",
    "2 years",
    "3 years",
    "5 years",
    "7 years",
    "10 years",
    "20 years",
    "30 years",
)


@st.cache_data(show_spinner="Preparing the walkthrough from the latest Treasury curve...")
def guide_snapshot(data: pd.DataFrame) -> dict:
    curve_date = data.index.max()
    observed = data.loc[curve_date, SERIES_IDS].to_numpy(dtype=float)
    ns_fit = fit_ns(MATURITIES, observed)
    nss_fit = fit_nss(MATURITIES, observed)
    spline_fit = fit_smoothing_spline(MATURITIES, observed)
    grid = np.linspace(float(MATURITIES[0]), 30.0, 90)
    ns_params = ns_fit.params
    nss_params = nss_fit.params
    ns_line = ns_curve(grid, ns_params["beta0"], ns_params["beta1"], ns_params["beta2"], ns_params["lambda"])
    nss_line = nss_curve(
        grid,
        nss_params["beta0"],
        nss_params["beta1"],
        nss_params["beta2"],
        nss_params["beta3"],
        nss_params["lambda1"],
        nss_params["lambda2"],
    )
    spline_line = predict_smoothing_spline(MATURITIES, observed, grid, spline_fit.params["smoothing_factor"])
    forward = nss_forward(
        grid,
        nss_params["beta0"],
        nss_params["beta1"],
        nss_params["beta2"],
        nss_params["beta3"],
        nss_params["lambda1"],
        nss_params["lambda2"],
    )
    pca = compute_pca(data)
    loadings = pca["loadings"]
    backtest = run_curve_spread_backtest(data)

    def nss_at(times: np.ndarray) -> np.ndarray:
        return nss_curve(
            times,
            nss_params["beta0"],
            nss_params["beta1"],
            nss_params["beta2"],
            nss_params["beta3"],
            nss_params["lambda1"],
            nss_params["lambda2"],
        )

    risk = bond_risk(10.0, 0.04, 100.0, 2, nss_at)
    y2 = float(observed[SERIES_IDS.index("DGS2")]) * 100
    y10 = float(observed[SERIES_IDS.index("DGS10")]) * 100
    pnl = backtest["cumulative_pnl"]
    return {
        "date": str(pd.Timestamp(curve_date).date()),
        "observations": len(data),
        "yields_percent": {
            label: round(float(level) * 100, 3) for label, level in zip(TENOR_LABELS, observed)
        },
        "maturities": [float(value) for value in MATURITIES],
        "observed_percent": [float(value) * 100 for value in observed],
        "grid": [float(value) for value in grid],
        "ns_percent": [float(value) * 100 for value in ns_line],
        "nss_percent": [float(value) * 100 for value in nss_line],
        "spline_percent": [float(value) * 100 for value in spline_line],
        "forward_percent": [float(value) * 100 for value in forward],
        "ns_rmse_bp": round(float(ns_fit.rmse) * 10000, 2),
        "nss_rmse_bp": round(float(nss_fit.rmse) * 10000, 2),
        "spline_rmse_bp": round(float(spline_fit.rmse) * 10000, 2),
        "two_year": round(y2, 3),
        "ten_year": round(y10, 3),
        "slope_bp": round((y10 - y2) * 100, 1),
        "loadings": {str(column): [float(value) for value in loadings[column].tolist()] for column in loadings.columns},
        "explained": [float(value) for value in pca["explained_variance"]],
        "explained_total": float(np.sum(pca["explained_variance"])),
        "sharpe": round(float(backtest["sharpe"]), 2),
        "max_drawdown_bp": round(abs(float(backtest["max_drawdown"])) * 10000, 1),
        "underwater_days": int(backtest["max_drawdown_duration"]),
        "pnl_dates": [str(pd.Timestamp(stamp).date()) for stamp in pnl.index],
        "pnl": [float(value) * 10000 for value in pnl.to_numpy()],
        "bond_price": round(float(risk["price"]), 2),
        "bond_duration": round(float(risk["modified_duration"]), 2),
        "bond_dv01": round(float(risk["dv01"]), 3),
        "key_dv01": {str(int(tenor)): round(float(value), 3) for tenor, value in risk["key_rate_dv01"].items()},
    }


def _guide_text(text: str) -> None:
    st.markdown(f"<div class='guide-prose'><p>{text}</p></div>", unsafe_allow_html=True)


def _guide_caption(text: str) -> None:
    st.markdown(f"<div class='guide-caption'>{text}</div>", unsafe_allow_html=True)


def render_guide_page() -> None:
    st.session_state.setdefault("theme", "research")
    if st.session_state["theme"] not in THEMES:
        st.session_state["theme"] = "research"
    inject_css(st.session_state["theme"])
    if st.button("Back to the desk"):
        st.switch_page("app.py")
    try:
        data, source, warning = load_data()
    except Exception as exc:
        error_card(f"Data load failed: {exc}")
        return
    if warning:
        st.warning(warning)
    snap = guide_snapshot(data)
    st.markdown(
        "<div class='kicker'>How to read this</div>"
        "<div class='app-title'>A walkthrough of the Treasury curve</div>"
        f"<div class='meta-line'>Figures as of {escape(snap['date'])} · {escape(str(source))} · {snap['observations']:,} days</div>",
        unsafe_allow_html=True,
    )
    st.markdown("## What you are looking at")
    _guide_text(
        "The U.S. Treasury borrows for one month, for two years, for thirty years. "
        "Each maturity has an interest rate. Place those rates on a chart, short loans on the left and long loans on the right, "
        "and the line you get is the yield curve. It is the market's price of time."
    )
    _guide_text(
        "This desk draws that line, measures how well a formula can reproduce it, "
        "splits the day-to-day movement into a few repeating patterns, and then asks whether a simple trading rule would have worked in the past. "
        "The pages below use the same numbers as the screen you just left."
    )

    st.markdown("## The eleven rates")
    _guide_text(
        "The dots come from FRED, the public database of the Federal Reserve Bank of St. Louis. "
        "They are constant-maturity Treasury yields: the rate on a Treasury loan of that exact length, published each business day. "
        "A higher dot means it costs the government more to borrow for that long."
    )
    if snap["slope_bp"] >= 0:
        slope_sentence = (
            f"On {snap['date']} the 2-year yield is {snap['two_year']:.2f}% and the 10-year yield is {snap['ten_year']:.2f}%. "
            f"The 10-year sits {snap['slope_bp']:.0f} basis points above the 2-year. "
            "One basis point is one hundredth of a percent. That gap is the slope: lenders ask to be paid more to lock money away for longer."
        )
    else:
        slope_sentence = (
            f"On {snap['date']} the 2-year yield is {snap['two_year']:.2f}% and the 10-year yield is {snap['ten_year']:.2f}%. "
            f"The 10-year sits {abs(snap['slope_bp']):.0f} basis points below the 2-year. "
            "The curve slopes downward. That is unusual, and readers often take it as concern about growth ahead."
        )
    _guide_text(slope_sentence)
    observed = go.Figure()
    observed.add_trace(
        go.Scatter(
            x=snap["maturities"],
            y=snap["observed_percent"],
            mode="lines+markers",
            name="Published yield",
            marker={"size": 8},
        )
    )
    style_figure(observed, xaxis_title="Years until the loan matures", yaxis_title="Yield (%)")
    with st.container(border=True):
        st.plotly_chart(observed, use_container_width=True)
        _guide_caption(
            "Each dot is one published Treasury yield. Read left to right as short loans to long loans. "
            "The height is the annual interest rate, in percent."
        )
    st.markdown(
        mini_table(["Maturity", "Yield"], [[label, f"{value:.2f}%"] for label, value in snap["yields_percent"].items()]),
        unsafe_allow_html=True,
    )

    st.markdown("## Drawing a smooth line")
    _guide_text(
        "Eleven dots are not enough for a desk. Someone always asks for the rate at four years, or at twelve. "
        "Three formulas fill in the gaps. Nelson-Siegel is the simple one: a long-run rate, a slope, and a single bend. "
        "Svensson is the same idea with a second bend, used when the curve humps in the middle. "
        "The smoothing spline stays closer to the dots and is less interested in a story about why the curve bends."
    )
    _guide_text(
        f"The miss is measured in basis points. On {snap['date']} Nelson-Siegel misses by {snap['ns_rmse_bp']:.2f} bp, "
        f"Svensson by {snap['nss_rmse_bp']:.2f} bp, and the spline by {snap['spline_rmse_bp']:.2f} bp. "
        "A smaller number means the line sits closer to the published yields. "
        "Closer is not always more useful: a line that hits every dot can bend for a reason that will not repeat tomorrow."
    )
    fits = go.Figure()
    fits.add_trace(
        go.Scatter(x=snap["maturities"], y=snap["observed_percent"], mode="markers", name="Published", marker={"size": 8})
    )
    fits.add_trace(go.Scatter(x=snap["grid"], y=snap["ns_percent"], mode="lines", name="Nelson-Siegel"))
    fits.add_trace(go.Scatter(x=snap["grid"], y=snap["nss_percent"], mode="lines", name="Svensson"))
    fits.add_trace(go.Scatter(x=snap["grid"], y=snap["spline_percent"], mode="lines", name="Spline"))
    style_figure(fits, xaxis_title="Years", yaxis_title="Yield (%)")
    with st.container(border=True):
        st.plotly_chart(fits, use_container_width=True)
        _guide_caption(
            "The dots are the market. The lines are three ways to connect them. "
            "Where a line leaves a dot, that is the fitting error quoted above."
        )

    st.markdown("## The dashed forward line")
    _guide_text(
        "The yield at ten years is an average rate from today out to year ten. "
        "The forward is the rate that average implies for a moment along the way: what the curve says about year nine, or year fifteen, on its own. "
        "The dashed line on the desk is that forward, taken from the Svensson fit."
    )
    _guide_text(
        "It is the forward of the smooth curve drawn through the published yields. "
        "A different exercise, stripping individual Treasury bonds into a discount curve, is not what this line shows."
    )
    forwards = go.Figure()
    forwards.add_trace(go.Scatter(x=snap["grid"], y=snap["nss_percent"], mode="lines", name="Svensson yield"))
    forwards.add_trace(
        go.Scatter(x=snap["grid"], y=snap["forward_percent"], mode="lines", name="Forward", line={"dash": "dash"})
    )
    style_figure(forwards, xaxis_title="Years", yaxis_title="Rate (%)")
    with st.container(border=True):
        st.plotly_chart(forwards, use_container_width=True)
        _guide_caption(
            "The solid line is the fitted yield. The dashed line is the forward implied by that same fit. "
            "When the dashed line sits above the solid line, the curve is saying that rates further out are higher than the average so far."
        )

    st.markdown("## What a bond loses if rates rise")
    price_sentence = (
        f"Take a 10-year bond with a 4% coupon, priced off the Svensson curve on {snap['date']}. "
        f"Its price is {snap['bond_price']:.2f} per 100 of face value, so "
        + (
            "the coupon is rich relative to the curve and the bond is worth more than face."
            if snap["bond_price"] >= 100
            else "the coupon is cheap relative to the curve and the bond is worth less than face."
        )
    )
    _guide_text(price_sentence)
    _guide_text(
        f"Modified duration is {snap['bond_duration']:.2f}. "
        f"DV01 is the practical version of that number: if every yield rises by one basis point, this bond loses about ${snap['bond_dv01']:.3f} per $100 of face value. "
        f"Split by region of the curve, the 2-year bump accounts for ${snap['key_dv01'].get('2', 0):.3f}, "
        f"the 5-year for ${snap['key_dv01'].get('5', 0):.3f}, "
        f"the 10-year for ${snap['key_dv01'].get('10', 0):.3f}, "
        f"and the 30-year for ${snap['key_dv01'].get('30', 0):.3f}. "
        "A 10-year bond should feel the 10-year region most."
    )
    keys = ["2", "5", "10", "30"]
    risk_fig = go.Figure(
        go.Bar(
            x=[f"{key}-year" for key in keys],
            y=[snap["key_dv01"].get(key, 0) for key in keys],
            marker_color=THEMES[active_theme()]["accent"],
            name="Key-rate DV01",
        )
    )
    style_figure(risk_fig, yaxis_title="Price loss per $100 face")
    with st.container(border=True):
        st.plotly_chart(risk_fig, use_container_width=True)
        _guide_caption(
            "Each bar is the loss on the sample 10-year bond if only that part of the curve rises by one basis point. "
            "The tallest bar is the region the bond actually depends on."
        )

    st.markdown("## The three ways the curve moves")
    share = snap["explained_total"] * 100
    _guide_text(
        "Most days the whole curve shifts up or down together. That pattern is the level. "
        "Some days the long end moves more than the short end. That is the slope. "
        "A less common day bends the middle relative to the wings. That is the curvature. "
        f"On this history, the first three patterns describe {share:.1f}% of the day-to-day changes. "
        "The remainder is small and noisy. The calculation uses changes in yields, so it describes motion, not the height of the curve itself."
    )
    loadings_fig = go.Figure()
    for name, values in snap["loadings"].items():
        loadings_fig.add_trace(go.Scatter(x=snap["maturities"], y=values, mode="lines+markers", name=name))
    style_figure(loadings_fig, xaxis_title="Years", yaxis_title="Loading")
    with st.container(border=True):
        st.plotly_chart(loadings_fig, use_container_width=True)
        _guide_caption(
            "A flat line across maturities is a level move: every yield changes together. "
            "A line that rises from left to right is a slope move. A hump in the middle is curvature. "
            "The name on each line is assigned from that shape."
        )

    st.markdown("## A rule, tested only on the past")
    _guide_text(
        "The backtest follows one rule: when the gap between the 10-year and the 2-year yield is unusually wide or unusually tight versus its own recent history, lean the other way, on the idea that the gap will drift back. "
        "The rule is only allowed to learn from the prior year, then it is tested on the next few weeks, then it learns again. "
        "It waits a day before the position changes, and a small cost is charged when it trades. "
        "That discipline is what keeps the path from peeking at the future."
    )
    _guide_text(
        f"The Sharpe ratio on this path is {snap['sharpe']:.2f}. "
        "Sharpe compares the average result with how bumpy the path was. A number near zero means the gains and the setbacks cancelled. "
        f"The worst fall from a peak is {snap['max_drawdown_bp']:.1f} basis points, "
        f"and the longest stretch below a peak lasted {snap['underwater_days']} days. "
        "This is a research path for a published yield spread. It is not a statement about money in an account, and it is not a forecast."
    )
    pnl_fig = go.Figure(go.Scatter(x=snap["pnl_dates"], y=snap["pnl"], name="Cumulative result"))
    style_figure(pnl_fig, yaxis_title="Cumulative result (bp)")
    with st.container(border=True):
        st.plotly_chart(pnl_fig, use_container_width=True)
        _guide_caption(
            "The line adds up the daily change in the 10-year minus 2-year gap, after the trading cost, in basis points. "
            "A rise means the rule was on the right side of the next day's move. A fall means it was not."
        )

    st.markdown("## The AI report and the analyst")
    _guide_text(
        "AI Report, on each tab, writes a formal note with three parts: what the figures say, which model or tenor produced them, and what they do not identify. "
        "The Analyst page holds a conversation with the same limit. Both see only the brief already on screen. "
        "Neither is given a news feed, and neither should invent a number. Ten requests are allowed every five hours."
    )
    _guide_text(
        "Research is the paper theme. Desk is the dark screen. The figures are the same in both."
    )
    if st.button("Return to the desk"):
        st.switch_page("app.py")


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
    title_col, switch_col = st.columns([4, 1.5], vertical_alignment="center", gap="small")
    with title_col:
        st.markdown(
            "<div class='kicker'>U.S. Treasury</div>"
            "<div class='app-title'>Yield Curve</div>"
            f"<div class='meta-line'>As of {data.index.max().date()} · {escape(str(source))} · {len(data):,} observations</div>",
            unsafe_allow_html=True,
        )
    with switch_col:
        st.markdown("<div class='kicker header-controls'>Theme</div>", unsafe_allow_html=True)
        theme_col, help_col = st.columns([3.2, 1.3], vertical_alignment="center", gap="small")
        with theme_col:
            st.radio(
                "Theme",
                ["research", "desk"],
                horizontal=True,
                format_func=lambda value: "Research" if value == "research" else "Desk",
                label_visibility="collapsed",
                key="theme",
            )
        with help_col:
            st.markdown("<div class='help-control'></div>", unsafe_allow_html=True)
            if st.button("HELP ?", key="open-guide", help="How to read this desk"):
                st.switch_page("pages/guide.py")
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
