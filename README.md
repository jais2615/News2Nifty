# News2Nifty: Market Movement Predictor using News Sentiment & Technicals

**News2Nifty** combines **NLP-based news sentiment analysis** with **technical indicators** (RSI, SMA, MACD, Bollinger Bands) and a **TensorFlow LSTM model** to forecast the **next-day closing price of the NIFTY 50 index**. Delivers real-time predictions using both headline impact and historical market behavior.

---

## Key Features

- **Live Market Pulse** (flagship page) — recent NIFTY/Sensex/RBI-relevant headlines, sentiment-scored live with color-coded badges, one-click "Analyze →" straight into the impact prediction
- **Real-time Sentiment Scoring** using VADER NLP — via a live headline, pasted headline, or article URL
- **LSTM-based Price Forecasting** with TensorFlow
- **Interactive Price Chart** (Plotly) — NIFTY 50 close with SMA-20 and Bollinger Bands overlay, RSI subplot, and your latest prediction plotted as a projected point on the chart
- **Prediction History** — every prediction you run in a session is logged and charted (price-at-prediction vs. predicted price) so you can see your own analysis history at a glance
- **Technical Indicators**: 14-day RSI, 20-day SMA, MACD (12/26/9), Bollinger Bands (%B)
- **Live Market Data** via YFinance
- **Multi-page Streamlit Interface** with top navigation between Live Market Pulse, Analyze Impact, Price Chart, and Prediction History

> **Note on indicators:** RSI and SMA are fed into the LSTM prediction model. MACD and Bollinger Bands are currently **informational only** (shown on the Price Chart page) — not yet part of the model's input features. See [Roadmap](#roadmap) below.

### Navigating the app
The app has four pages, switchable via the buttons at the top:
- **Live Market Pulse** — the default landing page. Browse live, sentiment-scored headlines and jump straight into analysis.
- **Analyze Impact** — run a prediction from a live headline, a pasted headline/URL, or whatever you selected on Market Pulse.
- **Price Chart** — interactive Plotly chart with SMA/Bollinger overlay, RSI subplot, and your latest prediction plotted on the chart.
- **Prediction History** — a running, in-session log of every prediction you've made, charted as price-at-prediction vs. predicted price. This resets when the app restarts — it's a session log, not a backtested accuracy report.

---

## Requirements

**Python 3.11 is recommended.** TensorFlow does not yet support Python 3.14 (as of writing, official support tops out around 3.13), so if your default `python`/`venv` points to 3.14, create your virtual environment with 3.11 specifically:
```powershell
py -0p                          # list installed Python versions
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
Also avoid stacking a venv inside an active conda environment (e.g. `(.venv) (base) PS ...`) — run `conda deactivate` first to avoid dependency conflicts.

Install dependencies via `pip`:
```
vaderSentiment==3.3.2
joblib==1.3.2
streamlit==1.29.0
tensorflow==2.15.0
yfinance==0.2.31
requests==2.31.0
beautifulsoup4==4.12.2
python-dotenv==1.0.1
```

---

## Quick Start

### 1. Clone the Repository
```
git clone https://github.com/yourusername/news2nifty.git
cd news2nifty
```

### 2. Install Required Packages
```
pip install -r requirements.txt
```
Note: use Python 3.11 (see [Requirements](#-requirements) above) — TensorFlow doesn't yet support Python 3.14.

You'll also need these files in the root directory:
```
model.h5      – Trained LSTM model
scaler.pkl    – MinMaxScaler used for feature scaling
```

### 3. (Optional) Enable Live Headlines
The live-headline picker needs a free NewsData.io API key:

1. Get a free key at [newsdata.io](https://newsdata.io/)
2. Set it using **one** of these methods:

   **Option A — `.env` file (recommended, works the same on every OS):**
   A `.env` file is already included in this repo as a template. Open it and replace the placeholder:
   ```
   NEWSDATA_API_KEY=your_key_here
   ```
   `fin.py` loads this automatically via `python-dotenv` — no terminal commands needed.
   ⚠️ `.env` is already listed in `.gitignore`, so your real key won't get committed. Never remove it from `.gitignore`.

   **Option B — environment variable for the current terminal session:**
   ```powershell
   # Windows PowerShell
   $env:NEWSDATA_API_KEY="your_key_here"
   ```
   ```bash
   # macOS/Linux
   export NEWSDATA_API_KEY="your_key_here"
   ```

   **Option C — Streamlit Cloud:** add `NEWSDATA_API_KEY` under your app's **Secrets** in the dashboard.

If you skip this step, the app still works fully via manual headline/URL paste — live headlines will just show a note that the key isn't set.

### 4. Run the Streamlit App
```
streamlit run fin.py
```

---

## How to Test It

### Basic functionality check
1. Run the app (`streamlit run fin.py`) — it should open in your browser at `http://localhost:8501`.
2. Confirm the **current NIFTY 50 price** loads at the top. If it shows a fallback value (₹23,851.65) instead of a live number, check your internet connection or `yfinance`'s status.
3. You should land on **📰 Live Market Pulse** by default. If headlines don't load, check your `NEWSDATA_API_KEY` setup.

### Test the Live Market Pulse → Analyze flow
1. On the Market Pulse page, confirm each headline shows a colored sentiment badge (green/red/gray) with a score.
2. Click **"Analyze →"** on any headline card.
3. You should land on the **🎯 Analyze Impact** page with that headline pre-filled and a green "Using: ..." confirmation.
4. Click **Analyze Impact** — you should see a VADER sentiment score, a predicted price impact, and a projected price.

### Test manual headline input
1. On the Analyze page, switch to **"Paste headline / URL"**.
2. Paste this sample headline:
   ```
   RBI maintains repo rate at 6.5% amid inflation concerns
   ```
3. Click **Analyze Impact** and confirm sentiment + prediction output appears.
4. Try a URL instead of raw text — the app should extract the headline automatically and show it in an info box before you click Analyze.

### Test the Price Chart page
1. Go to **📊 Price Chart**.
2. Confirm RSI, MACD, and Bollinger %B metrics show real numbers, not blank/error states.
3. Confirm the chart renders with a blue price line, an orange dotted SMA-20 line, and a shaded Bollinger Band region.
4. After running at least one prediction on the Analyze page, revisit this chart — you should see a dashed star-marked line projecting from the last actual close to your predicted price.

### Test Prediction History
1. Run 2–3 predictions on different headlines from the Analyze page.
2. Go to **🕒 Prediction History** — confirm each shows up in the chart and the table below it, with correct timestamps and sentiment scores.
3. Click **"Clear history"** and confirm the log empties.

### Sanity-check the prediction direction
- A clearly positive headline (e.g. `"Markets rally as inflation cools faster than expected"`) should produce a **positive** sentiment score and (usually) a positive price impact.
- A clearly negative headline (e.g. `"Markets plunge amid recession fears"`) should produce a **negative** sentiment score and (usually) a negative price impact.
- This isn't a guarantee for every input — the LSTM weighs sentiment against current technicals — but wildly inverted results on obviously one-sided headlines would indicate a bug worth investigating.

### Common issues
| Symptom | Likely cause |
|---|---|
| `ERROR: No matching distribution found for tensorflow` | Your venv is on an unsupported Python version (e.g. 3.14). Create a new venv with Python 3.11 — see Requirements section |
| `streamlit : not recognized` right after a failed `pip install` | Nothing actually installed because an earlier package (often tensorflow) failed — fix the install first, then retry |
| `No module named 'sklearn'` when loading a prediction | `scaler.pkl` needs scikit-learn to unpickle — make sure it's in `requirements.txt` and installed |
| yfinance downloads return 0 rows silently, no error | Outdated `yfinance` version incompatible with Yahoo's current API — run `pip install --upgrade yfinance` |
| Current price shows fallback value | `yfinance` couldn't reach Yahoo Finance, or an old version is silently failing — see yfinance troubleshooting above |
| "Prediction Error" message | `model.h5` or `scaler.pkl` missing from root directory, or version mismatch with TensorFlow |
| Live headlines always empty | Missing/invalid `NEWSDATA_API_KEY`, or free-tier daily quota exhausted |
| Headline extraction fails on a URL | Some sites block scraping (paywalls, bot detection) — paste the headline text directly instead |
| `set_page_config() can only be called once` | Some other `st.*` call happened before it — it must be the very first Streamlit command, including before any `st.secrets` access |

---

## Project Structure

```
news2nifty/
├── fin.py                  # Streamlit frontend and prediction logic
├── data scraping.ipynb     # Historical headline scraping (Selenium)
├── model.h5                # Pretrained LSTM model
├── scaler.pkl              # Feature scaler
├── requirements.txt
└── README.md                # You're here
```

---

## Roadmap

- [ ] Retrain the LSTM with MACD/Bollinger Bands included as model input features (currently display-only)
- [ ] Evaluate FinBERT vs. VADER for sentiment scoring on a recent data slice
- [ ] Backtesting module to validate directional accuracy against historical headlines
- [ ] Multi-index support (Bank Nifty, Sensex)

---

## Changelog

- **Added:** Multi-page navigation (Live Market Pulse, Analyze Impact, Price Chart, Prediction History)
- **Added:** "Live Market Pulse" flagship page — sentiment-scored headline cards with one-click analysis
- **Added:** Interactive Plotly price chart with SMA-20/Bollinger overlay, RSI subplot, and projected-prediction marker
- **Added:** In-session Prediction History log and chart (price-at-prediction vs. predicted price)
- **Added:** Live headline sourcing via NewsData.io, with graceful fallback to manual paste
- **Added:** MACD and Bollinger Bands as informational technical indicators
- **Fixed:** Chrome driver options bug in the historical scraping notebook (custom flags were previously being silently ignored)
- **Fixed:** yfinance MultiIndex column handling and stale/incompatible version issues
- **Cleaned:** Consolidated 10+ duplicated scraping cells into a single resumable extraction loop

---

This project is for educational and research purposes only. It is not financial advice. Always do your own due diligence before making any investment decisions.