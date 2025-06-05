import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="NIFTY Impact Analyzer", layout="centered")
st.markdown("""
<style>
    .main-header {color:#1E88E5; font-size:28px; font-weight:bold; text-align:center; margin-bottom:20px;}
    .result-box {background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:20px; text-align:center;}
    .news-input {margin-bottom:15px;}
    .sentiment-score {margin:10px 0; padding:8px; background-color:#f8f8f8; border-radius:4px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📈 NIFTY 50 Impact Analyzer</h1>", unsafe_allow_html=True)

# --- Technical Indicator Calculation ---
def calculate_rsi(data, window=14):
    deltas = data['NIFTY_50_Close'].diff()
    gain = (deltas.where(deltas > 0, 0)).rolling(window).mean()
    loss = (-deltas.where(deltas < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Fetch Live Data ---
@st.cache_data(ttl=60)
def get_live_data():
    try:
        data = yf.download("^NSEI", period="60d", interval="1d")
        data = data.rename(columns={'Close': 'NIFTY_50_Close'})
        data['SMA_20'] = data['NIFTY_50_Close'].rolling(20).mean()
        data['RSI'] = calculate_rsi(data)
        return data.dropna()
    except Exception as e:
        st.error(f"Data Fetch Error: {str(e)}")
        return pd.DataFrame(columns=['NIFTY_50_Close', 'SMA_20', 'RSI'])

live_data = get_live_data()

# --- Current Price Display ---
if not live_data.empty:
    current_price = float(live_data['NIFTY_50_Close'].iloc[-1])
else:
    current_price = 23851.65  # fallback value

st.markdown(f"""
<div class='result-box'>
    <div style="font-size:18px; font-weight:500; color: #000000;">
        Current NIFTY 50: ₹{current_price:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)

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
def predict_impact(sentiment):
    try:
        scaler = joblib.load('scaler.pkl')
        model = load_model('model.h5')
        seq = prepare_sequence(sentiment)
        if seq.shape != (30, 4):
            seq = seq.tail(30).reset_index(drop=True)
        scaled = scaler.transform(seq.values)
        X = scaled.reshape(1, 30, 4)
        
        # Get the raw model prediction
        raw_pct_change = float(model.predict(X)[0][0])
        
        # Use the absolute value of the model prediction and multiply by sentiment
        # This ensures direction matches sentiment and magnitude is proportional to both
        weighted_pct_change = abs(raw_pct_change) * sentiment
        
        # Calculate price impact
        price_impact = float(current_price * (weighted_pct_change / 100))
        return weighted_pct_change, price_impact
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return 0.0, 0.0

# --- Main Interface ---
st.text("Enter news headline/article URL:")
news_input = st.text_input("", label_visibility="collapsed")

headline = news_input
if news_input and news_input.startswith(('http://', 'https://')):
    with st.spinner("Extracting headline..."):
        extracted_headline = extract_headline(news_input)
        headline = extracted_headline
        st.info(f"Extracted: **{headline}**")

if st.button("Analyze Impact") and headline and len(headline) > 3:
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(headline)
    sentiment = sentiment_scores['compound']

    # Show VADER sentiment score
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

    # --- UI: Impact without percent display ---
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
