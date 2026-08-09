import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
# Must be the very first Streamlit command in the script — before ANY other
# st.* call, including st.secrets access below. Calling st.secrets first
# (even inside a try/except) can trigger an internal Streamlit render when
# no secrets.toml exists, which then breaks set_page_config() if it runs after.
st.set_page_config(page_title="NIFTY 50 Impact Analyzer", layout="wide", page_icon="📈")

# Load variables from a local .env file, if present (e.g. NEWSDATA_API_KEY=...).
load_dotenv()

# --- NewsData.io API key (live headlines feature) ---
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")
if not NEWSDATA_API_KEY:
    try:
        NEWSDATA_API_KEY = st.secrets.get("NEWSDATA_API_KEY", "")
    except Exception:
        NEWSDATA_API_KEY = ""

# --- Global styling ---
st.markdown("""
<style>
    .main-header {color:#1E88E5; font-size:30px; font-weight:bold; margin-bottom:4px;}
    .sub-header {color:#888; font-size:14px; margin-bottom:20px;}
    .result-box {background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:20px; text-align:center;}
    .sentiment-score {margin:10px 0; padding:8px; background-color:#f8f8f8; border-radius:4px;}
    .headline-card {
        background-color:#ffffff10; border:1px solid #ffffff22; border-left:4px solid #1E88E5;
        border-radius:8px; padding:14px 16px; margin-bottom:12px;
    }
    .badge {
        display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600;
    }
    .nav-tag {font-size:13px; color:#888; margin-bottom:2px;}
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if "page" not in st.session_state:
    st.session_state.page = "pulse"  # land on the live headlines page by default
if "selected_headline" not in st.session_state:
    st.session_state.selected_headline = ""
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []  # list of dicts: timestamp, headline, sentiment, price_at_prediction, predicted_price

analyzer = SentimentIntensityAnalyzer()

# --- Technical Indicator Calculation ---
def calculate_rsi(data, window=14):
    deltas = data['NIFTY_50_Close'].diff()
    gain = (deltas.where(deltas > 0, 0)).rolling(window).mean()
    loss = (-deltas.where(deltas < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram). Display-only — not fed
    into model.h5, which was trained on a fixed 4-feature set. Adding this
    to the model's input would require retraining."""
    close = data['NIFTY_50_Close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Returns (upper_band, lower_band, %B). Display-only, same reasoning
    as MACD above — informational, not a model input."""
    close = data['NIFTY_50_Close']
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    percent_b = (close - lower) / (upper - lower)
    return upper, lower, percent_b

# --- Fetch Live Data ---
@st.cache_data(ttl=60)
def get_live_data():
    try:
        data = yf.download("^NSEI", period="60d", interval="1d")
        # Newer yfinance versions can return MultiIndex columns (ticker, field)
        # even for a single ticker — flatten defensively so renaming below works.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # Defensive: drop any duplicate column labels — duplicate labels cause
        # single-cell lookups like latest['RSI'] to return a Series instead of
        # a scalar, which breaks f-string formatting (":.1f") downstream.
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.rename(columns={'Close': 'NIFTY_50_Close'})
        data['SMA_20'] = data['NIFTY_50_Close'].rolling(20).mean()
        data['RSI'] = calculate_rsi(data)
        macd_line, signal_line, histogram = calculate_macd(data)
        data['MACD'] = macd_line
        data['MACD_Signal'] = signal_line
        data['MACD_Hist'] = histogram
        upper, lower, percent_b = calculate_bollinger_bands(data)
        data['BB_Upper'] = upper
        data['BB_Lower'] = lower
        data['BB_PercentB'] = percent_b
        return data.dropna()
    except Exception as e:
        st.error(f"Data Fetch Error: {str(e)}")
        return pd.DataFrame(columns=[
            'NIFTY_50_Close', 'SMA_20', 'RSI',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_PercentB'
        ])

live_data = get_live_data()

if not live_data.empty:
    current_price = float(live_data['NIFTY_50_Close'].iloc[-1])
else:
    current_price = 23851.65  # fallback value

# --- Live Headlines (NewsData.io) ---
@st.cache_data(ttl=600)  # cache 10 min — free plan has a daily credit limit
def fetch_live_headlines(max_results=30, max_pages=3):
    """Pull recent NIFTY/Indian-market-relevant headlines from NewsData.io's
    'latest' endpoint (free plan covers roughly the past 48 hours). Only
    headline TITLES are used — the free plan's 'content' field often just
    returns a paywall placeholder string, not real text.

    Paginates via NewsData's `nextPage` token, up to `max_pages` requests
    (~10 articles per page), so this uses up to `max_pages` of your daily
    free-tier request quota (25/day) PER CACHE MISS — i.e. once every 10
    minutes at most, since results are cached. Previously this only ever
    made a single request and silently capped out around ~10 articles.

    Returns (headlines, fetched_at). fetched_at is computed INSIDE this
    cached function, so it only changes when a real API call happens (cache
    miss) — a reliable way to prove to the user whether 'refresh' actually
    re-fetched, versus NewsData's own feed simply not having new articles."""
    # Explicitly IST — pd.Timestamp.now() alone uses the SERVER's local
    # clock, which is UTC (or something else) on Streamlit Cloud, not your
    # timezone. Pinning to Asia/Kolkata keeps this correct regardless of
    # where the app is actually hosted.
    fetched_at = pd.Timestamp.now(tz="Asia/Kolkata")
    if not NEWSDATA_API_KEY:
        return [], fetched_at
    headlines = []
    page_token = None
    try:
        for _ in range(max_pages):
            params = {
                "apikey": NEWSDATA_API_KEY,
                "q": '("NIFTY 50" OR "Sensex" OR "RBI" OR "Indian stock market" OR "BSE")',
                "language": "en",
                "country": "in",
                "category": "business",
            }
            if page_token:
                params["page"] = page_token

            resp = requests.get("https://newsdata.io/api/1/latest", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                break

            results = data.get("results", []) or []
            headlines.extend(item.get("title", "").strip() for item in results if item.get("title"))

            page_token = data.get("nextPage")
            if not page_token or len(headlines) >= max_results:
                break

        return headlines[:max_results], fetched_at
    except Exception:
        # Return whatever we managed to collect before the failure, rather
        # than discarding partial results from earlier successful pages.
        return headlines[:max_results], fetched_at

# --- Headline Extraction ---
def extract_headline(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        headline = None
        h1_tags = soup.find_all('h1')
        if h1_tags and len(h1_tags) > 0:
            headline = h1_tags[0].get_text().strip()
        if not headline or len(headline) < 5:
            for cls in ['headline', 'article-title', 'entry-title', 'post-title']:
                element = soup.find(class_=cls)
                if element:
                    headline = element.get_text().strip()
                    break
        if not headline or len(headline) < 5:
            title_tag = soup.find('title')
            if title_tag:
                headline = title_tag.get_text().strip()
                headline = headline.split(' | ')[0].split(' - ')[0].strip()
        return headline if headline and len(headline) > 5 else "Could not extract headline"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Prepare Model Input Sequence ---
def prepare_sequence(sentiment):
    required_rows = 30
    columns_needed = ['NIFTY_50_Close', 'SMA_20', 'RSI']
    if live_data.empty or len(live_data) < 5:
        df = pd.DataFrame({
            'vader_score': [0.0] * required_rows,
            'NIFTY_50_Close': [current_price] * required_rows,
            'SMA_20': [current_price * 0.995] * required_rows,
            'RSI': [50.0] * required_rows,
        })
        df.loc[required_rows-1, 'vader_score'] = float(sentiment)
        return df
    tech_data = live_data[columns_needed].copy()
    if len(tech_data) < required_rows:
        padding_needed = required_rows - len(tech_data)
        first_row = tech_data.iloc[0].to_dict()
        padding = pd.DataFrame([first_row] * padding_needed)
        tech_data = pd.concat([padding, tech_data], ignore_index=True)
    elif len(tech_data) > required_rows:
        tech_data = tech_data.tail(required_rows).reset_index(drop=True)
    else:
        tech_data = tech_data.reset_index(drop=True)
    tech_data['vader_score'] = 0.0
    tech_data.loc[required_rows-1, 'vader_score'] = float(sentiment)
    return tech_data[['vader_score', 'NIFTY_50_Close', 'SMA_20', 'RSI']]

# --- Model Prediction ---
@st.cache_resource
def load_prediction_model():
    """Loads model.h5 and scaler.pkl once per session and reuses them.
    Previously these reloaded from disk on every single 'Analyze Impact'
    click, which is the main reason predictions felt slow — TensorFlow
    model loading is expensive and doesn't need to repeat."""
    from tensorflow.keras.models import load_model  # imported lazily here,
    # not at module top-level, so pages that don't need TF (e.g. Live Market
    # Pulse) aren't stuck waiting on TensorFlow's own slow import at startup.
    scaler = joblib.load('scaler.pkl')
    model = load_model('model.h5')
    return scaler, model

def predict_impact(sentiment):
    try:
        scaler, model = load_prediction_model()
        seq = prepare_sequence(sentiment)
        if seq.shape != (30, 4):
            seq = seq.tail(30).reset_index(drop=True)
        scaled = scaler.transform(seq.values)
        X = scaled.reshape(1, 30, 4)
        raw_pct_change = float(model.predict(X)[0][0])
        weighted_pct_change = abs(raw_pct_change) * sentiment
        price_impact = float(current_price * (weighted_pct_change / 100))
        return weighted_pct_change, price_impact
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return 0.0, 0.0

def sentiment_badge(score):
    if score > 0.05:
        return "Positive", "#1b5e20", "#4CAF5033"
    elif score < -0.05:
        return "Negative", "#b71c1c", "#F4433633"
    else:
        return "Neutral", "#555", "#9E9E9E33"

# =========================================================================
# --- Top Navigation ---
# =========================================================================
st.markdown("<div class='main-header'>📈 NIFTY 50 Impact Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>News sentiment + technicals + LSTM forecasting for the NIFTY 50</div>", unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
with nav_col1:
    if st.button("📰 Live Market Pulse", use_container_width=True, type="primary" if st.session_state.page == "pulse" else "secondary"):
        st.session_state.page = "pulse"
with nav_col2:
    if st.button("🎯 Analyze Impact", use_container_width=True, type="primary" if st.session_state.page == "analyze" else "secondary"):
        st.session_state.page = "analyze"
with nav_col3:
    if st.button("📊 Price Chart", use_container_width=True, type="primary" if st.session_state.page == "chart" else "secondary"):
        st.session_state.page = "chart"
with nav_col4:
    if st.button("🕒 Prediction History", use_container_width=True, type="primary" if st.session_state.page == "history" else "secondary"):
        st.session_state.page = "history"

st.markdown(f"""
<div class='result-box'>
    <div style="font-size:18px; font-weight:500; color: #000000;">
        Current NIFTY 50: ₹{current_price:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================================================
# --- PAGE: Live Market Pulse (flagship headline page) ---
# =========================================================================
if st.session_state.page == "pulse":
    st.subheader("📰 Live Market Pulse")
    st.caption("Recent NIFTY / Sensex / RBI-relevant headlines, scored live. Pick one to analyze its market impact.")

    if not NEWSDATA_API_KEY:
        st.warning(
            "Live headlines need a free NewsData.io API key. Set NEWSDATA_API_KEY in your "
            ".env file (or as a Streamlit secret) to enable this page."
        )
    else:
        refresh_col, ts_col = st.columns([1, 5])
        with refresh_col:
            if st.button("🔄 Refresh headlines"):
                fetch_live_headlines.clear()
                st.rerun()

        with st.spinner("Fetching live headlines..."):
            live_headlines, fetched_at = fetch_live_headlines()

        with ts_col:
            st.caption(
                f"Last fetched at {fetched_at.strftime('%H:%M:%S')} IST — fetches up to 3 pages "
                "(~30 headlines) per refresh, cached for 10 minutes. This timestamp only changes "
                "on a real API call, not on page reruns. Identical headlines after refreshing "
                "usually just means NewsData's feed hasn't published anything new in that window."
            )

        if not live_headlines:
            st.info("No live headlines available right now — try refreshing shortly, or paste a headline manually on the Analyze page.")
        else:
            pos_count = neg_count = neu_count = 0
            scored = []
            for h in live_headlines:
                score = analyzer.polarity_scores(h)['compound']
                scored.append((h, score))
                if score > 0.05:
                    pos_count += 1
                elif score < -0.05:
                    neg_count += 1
                else:
                    neu_count += 1

            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 Positive", pos_count)
            m2.metric("🔴 Negative", neg_count)
            m3.metric("⚪ Neutral", neu_count)

            # --- Aggregate impact across ALL fetched headlines ---
            # Mirrors how the original training data was built: one averaged
            # sentiment score per day across multiple headlines, not per
            # individual headline. This gives a "overall market mood" read
            # instead of just one story's impact.
            avg_sentiment = float(np.mean([s for _, s in scored]))
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"""
                <div class='headline-card' style="border-left-color:#673AB7;">
                    <div style="font-size:14px; color:#aaa;">Average sentiment across all {len(scored)} fetched headlines</div>
                    <div style="font-size:20px; font-weight:600; margin-top:4px;">{avg_sentiment:+.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"🌐 Analyze Overall Market Impact (all {len(scored)} headlines)", type="primary"):
                    with st.spinner("Calculating aggregate market impact..."):
                        weighted_pct_change, price_impact = predict_impact(avg_sentiment)
                        new_price = float(current_price) + float(price_impact)

                    color = "#4CAF50" if weighted_pct_change > 0 else "#F44336" if weighted_pct_change < 0 else "#9E9E9E"
                    impact_text = "Positive Impact" if weighted_pct_change > 0 else "Negative Impact" if weighted_pct_change < 0 else "Neutral Impact"
                    emoji = "📈" if weighted_pct_change > 0 else "📉" if weighted_pct_change < 0 else "⚖️"

                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; background:{color}15; border-left:4px solid {color}; margin-top:10px;">
                        <div style="font-size:18px; font-weight:500; margin-bottom:10px;">
                            {emoji} {impact_text} (aggregate of {len(scored)} headlines)
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                            <div>
                                <div style="font-size:14px; color:#666;">Price Impact</div>
                                <div style="font-size:16px; font-weight:500;">₹{price_impact:+.2f}</div>
                            </div>
                            <div>
                                <div style="font-size:14px; color:#666;">Projected Price</div>
                                <div style="font-size:16px; font-weight:500;">₹{new_price:,.2f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.session_state.prediction_history.append({
                        "timestamp": pd.Timestamp.now(tz="Asia/Kolkata"),
                        "headline": f"[Aggregate of {len(scored)} live headlines]",
                        "sentiment": avg_sentiment,
                        "price_at_prediction": current_price,
                        "predicted_price": new_price,
                    })

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Individual headlines")

            for idx, (h, score) in enumerate(scored):
                label, text_color, bg_color = sentiment_badge(score)
                card_col, btn_col = st.columns([5, 1])
                with card_col:
                    st.markdown(f"""
                    <div class='headline-card'>
                        <span class='badge' style="color:{text_color}; background:{bg_color};">{label} ({score:+.2f})</span>
                        <div style="margin-top:8px; font-size:15px;">{h}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with btn_col:
                    if st.button("Analyze →", key=f"pulse_btn_{idx}"):
                        st.session_state.selected_headline = h
                        st.session_state.page = "analyze"
                        st.rerun()

# =========================================================================
# --- PAGE: Analyze Impact ---
# =========================================================================
elif st.session_state.page == "analyze":
    st.subheader("🎯 Analyze Headline Impact")

    input_mode = st.radio(
        "Headline source:",
        ["Paste headline / URL", "Pick from live headlines", "Use headline from Market Pulse"],
        horizontal=True,
        index=2 if st.session_state.selected_headline else 0,
    )

    headline = ""

    if input_mode == "Paste headline / URL":
        news_input = st.text_input("Enter news headline/article URL:")
        headline = news_input
        if news_input and news_input.startswith(('http://', 'https://')):
            with st.spinner("Extracting headline..."):
                headline = extract_headline(news_input)
                st.info(f"Extracted: **{headline}**")

    elif input_mode == "Pick from live headlines":
        if not NEWSDATA_API_KEY:
            st.warning("Live headlines need a free NewsData.io API key. Set NEWSDATA_API_KEY to enable this.")
        else:
            with st.spinner("Fetching live headlines..."):
                live_headlines, _ = fetch_live_headlines()
            if not live_headlines:
                st.info("No live headlines available right now — try pasting one instead.")
            else:
                headline = st.selectbox("Select a recent headline:", live_headlines)

    else:  # Use headline from Market Pulse
        if st.session_state.selected_headline:
            st.success(f"Using: **{st.session_state.selected_headline}**")
            headline = st.session_state.selected_headline
        else:
            st.info("No headline selected yet — pick one from 📰 Live Market Pulse first.")

    if st.button("Analyze Impact", type="primary") and headline and len(headline) > 3:
        sentiment_scores = analyzer.polarity_scores(headline)
        sentiment = sentiment_scores['compound']

        sentiment_label = "Negative" if sentiment < 0 else "Positive" if sentiment > 0 else "Neutral"
        sentiment_color = "#F44336" if sentiment < 0 else "#4CAF50" if sentiment > 0 else "#9E9E9E"
        st.markdown(f"""
        <div class='sentiment-score' style="color:{sentiment_color};">
            VADER Sentiment Score: <b>{sentiment:.2f}</b> ({sentiment_label})
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Calculating market impact..."):
            weighted_pct_change, price_impact = predict_impact(sentiment)
            new_price = float(current_price) + float(price_impact)

        color = "#4CAF50" if weighted_pct_change > 0 else "#F44336" if weighted_pct_change < 0 else "#9E9E9E"
        impact_text = "Positive Impact" if weighted_pct_change > 0 else "Negative Impact" if weighted_pct_change < 0 else "Neutral Impact"
        emoji = "📈" if weighted_pct_change > 0 else "📉" if weighted_pct_change < 0 else "⚖️"

        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background:{color}15; border-left:4px solid {color};">
            <div style="font-size:18px; font-weight:500; margin-bottom:10px;">
                {emoji} {impact_text}
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div>
                    <div style="font-size:14px; color:#666;">Price Impact</div>
                    <div style="font-size:16px; font-weight:500;">₹{price_impact:+.2f}</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#666;">Projected Price</div>
                    <div style="font-size:16px; font-weight:500;">₹{new_price:,.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Log this prediction so it shows up on the Price Chart (as a projected
        # point) and the Prediction History page.
        st.session_state.prediction_history.append({
            "timestamp": pd.Timestamp.now(tz="Asia/Kolkata"),
            "headline": headline,
            "sentiment": sentiment,
            "price_at_prediction": current_price,
            "predicted_price": new_price,
        })

# =========================================================================
# --- PAGE: Price Chart & Indicators ---
# =========================================================================
elif st.session_state.page == "chart":
    st.subheader("📊 Price Chart & Technical Indicators")

    if live_data.empty:
        st.info("Live chart data unavailable right now.")
    else:
        latest = live_data.iloc[-1]
        rsi_val = float(latest['RSI'])
        macd_val = float(latest['MACD'])
        macd_signal_val = float(latest['MACD_Signal'])
        bb_percent_val = float(latest['BB_PercentB'])

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("RSI (14)", f"{rsi_val:.1f}")
            st.caption("Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")
        with m2:
            st.metric("MACD", f"{macd_val:.2f}")
            st.caption("Bullish crossover" if macd_val > macd_signal_val else "Bearish crossover")
        with m3:
            st.metric("Bollinger %B", f"{bb_percent_val:.2f}")
            st.caption(
                "Above upper band" if bb_percent_val > 1
                else "Below lower band" if bb_percent_val < 0
                else f"{bb_percent_val*100:.0f}% of band"
            )

        # --- Main interactive price chart ---
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
            vertical_spacing=0.05, subplot_titles=("Price with SMA-20 & Bollinger Bands", "RSI (14)")
        )

        fig.add_trace(go.Scatter(x=live_data.index, y=live_data['BB_Upper'], line=dict(width=1, color='rgba(150,150,150,0.4)'),
                                  name="Bollinger Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=live_data.index, y=live_data['BB_Lower'], line=dict(width=1, color='rgba(150,150,150,0.4)'),
                                  fill='tonexty', fillcolor='rgba(150,150,150,0.1)', name="Bollinger Lower", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=live_data.index, y=live_data['NIFTY_50_Close'], line=dict(color='#1E88E5', width=2),
                                  name="NIFTY 50 Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=live_data.index, y=live_data['SMA_20'], line=dict(color='orange', width=1.5, dash='dot'),
                                  name="SMA 20"), row=1, col=1)

        # Overlay the most recent prediction as a projected point, if any exist this session.
        if st.session_state.prediction_history:
            last_pred = st.session_state.prediction_history[-1]
            last_date = live_data.index[-1]
            next_date = last_date + timedelta(days=1)
            pred_color = "#4CAF50" if last_pred["predicted_price"] >= last_pred["price_at_prediction"] else "#F44336"
            fig.add_trace(go.Scatter(
                x=[last_date, next_date],
                y=[last_pred["price_at_prediction"], last_pred["predicted_price"]],
                mode="lines+markers",
                line=dict(color=pred_color, width=2, dash='dash'),
                marker=dict(size=8, symbol='star'),
                name="Latest Prediction"
            ), row=1, col=1)

        fig.add_trace(go.Scatter(x=live_data.index, y=live_data['RSI'], line=dict(color='#9C27B0', width=1.5),
                                  name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line=dict(color='rgba(244,67,54,0.4)', dash='dot'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='rgba(76,175,80,0.4)', dash='dot'), row=2, col=1)

        fig.update_layout(
            height=600, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# --- PAGE: Prediction History (session-tracked predicted vs actual) ---
# =========================================================================
elif st.session_state.page == "history":
    st.subheader("🕒 Prediction History (this session)")
    st.caption(
        "Every prediction you run on the Analyze page is logged here — the price at the moment "
        "you ran it, and what the model projected. This resets when you restart the app; it's a "
        "session log, not a backtested accuracy report."
    )

    if not st.session_state.prediction_history:
        st.info("No predictions yet — head to 🎯 Analyze Impact to run one.")
    else:
        hist_df = pd.DataFrame(st.session_state.prediction_history)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["price_at_prediction"],
            mode="lines+markers", name="Price at prediction time", line=dict(color="#1E88E5")
        ))
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["predicted_price"],
            mode="lines+markers", name="Predicted price", line=dict(color="#FF9800", dash="dash")
        ))
        fig.update_layout(height=400, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Log")
        display_df = hist_df[["timestamp", "headline", "sentiment", "price_at_prediction", "predicted_price"]].copy()
        display_df.columns = ["Time", "Headline", "Sentiment", "Price at Prediction", "Predicted Price"]
        display_df["Time"] = display_df["Time"].dt.strftime("%H:%M:%S")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if st.button("Clear history"):
            st.session_state.prediction_history = []
            st.rerun()