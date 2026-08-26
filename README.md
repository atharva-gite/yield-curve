# Yield Curve Construction & Term Structure Modeling Engine

Production-ready fixed-income analytics app for U.S. Treasury curve construction, Nelson-Siegel / Nelson-Siegel-Svensson / smoothing-spline fitting, yield-change PCA, walk-forward curve-spread backtesting, and Groq-backed quant commentary. The project is structured as a resume-grade internal quant tool: numerical logic is testable outside the UI, external data and LLM calls are isolated, and the Streamlit frontend exposes diagnostics rather than hiding model fragility.

## Architecture

- `logic.py`: math, data access, optimizers, backtests, Groq wrapper, SQLite-backed chat rate limiting, and serializable result objects. It contains no Streamlit calls, so all core behavior can be unit tested.
- `app.py`: Streamlit presentation layer. It handles layout, loading states, error cards, charts, and session state while delegating analytics to `logic.py`.
- `tests/`: numerical regression tests, mocked Groq tests, rate-limiter tests, walk-forward backtest tests, and a Streamlit smoke test.
- `data/treasury_yields_cache.csv`: local cache written by the loader so FRED outages do not break the app.

```text
FRED API / CSV / Cache -> logic.py analytics -> app.py Streamlit UI
                              |
                              +-> Groq context-grounded analyst
                              +-> pytest / Streamlit smoke tests
```

## Design Decisions

### NSS via SNLLS

NSS is ill-conditioned when the two decay parameters are close. The implementation uses separable nonlinear least squares: grid-search `lambda1/lambda2`, solve betas by conditional OLS for each grid point, then refine the best seeds with bounded L-BFGS-B while penalizing lambda collapse. This avoids treating all six parameters as equally nonlinear.

### PCA on Yield Changes

PCA is computed on first differences, not yield levels. Level PCA on nonstationary rate levels tends to overstate the first component and blur regime dynamics. The app labels Level / Slope / Curvature by inspecting loading shape and sign, then shows a diagnostic if the first three PCs explain less than 98%.

### Walk-Forward Backtesting

The backtester avoids look-ahead thresholding. Each test segment uses thresholds estimated only from the prior train window. Strategies share the same transaction-cost, slippage, and one-day execution-lag mechanics. Sharpe, PnL, max drawdown, and drawdown duration are reported together.

### Public Demo Rate Limiting

The AI Quant Analyst is protected by a SQLite-backed sliding-window limiter: 10 chat requests per 5-hour window per detected identifier. On proxied deployments, the app attempts to use forwarded client-IP headers; if those Streamlit internals are unavailable, it falls back to a browser-session identifier, and finally to a conservative shared identifier. The counter lives in `data/rate_limits.db`, so it persists across normal Streamlit reruns within the same instance.

## Setup

Use Python 3.12 for the pinned scientific stack:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create local secrets:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
FRED_API_KEY=your_fred_key
GROQ_API_KEY=your_groq_key
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
```

`FRED_API_KEY` is optional. If it is missing or invalid, the app falls back to FRED's public CSV endpoint, then to the local cache. `GROQ_API_KEY` is required only for the AI Quant Analyst tab; missing keys render an inline error instead of a traceback.

Create a free Groq key at `https://console.groq.com/keys`. Groq's free tier does not require a credit card at signup. The default model is `openai/gpt-oss-20b`; override it with `GROQ_MODEL` if you want a different Groq-hosted model.

## Run

```bash
.venv/bin/streamlit run app.py
```

## Test

```bash
.venv/bin/pytest tests/
```

## Docker

```bash
docker build -t yield-curve-engine .
docker run --rm -p 8501:8501 --env-file .env yield-curve-engine
```

Then open `http://localhost:8501`.

## Streamlit Community Cloud

For Streamlit Cloud, do not upload `.env`. Add secrets through the Streamlit Cloud settings UI. Use `.streamlit/secrets.toml.example` as the template for expected keys.
