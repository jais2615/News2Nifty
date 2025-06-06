# 📰 News2Nifty: Market Movement Predictor using News Sentiment & Technicals

**News2Nifty** combines **NLP-based news sentiment analysis** with **technical indicators** (like RSI and SMA) and a **TensorFlow LSTM model** to forecast the **next-day closing price of the NIFTY 50 index**. This app delivers real-time predictions using both headline impact and historical market behavior.

---

## ⚙️ Key Features

- 📡 **Real-time Sentiment Scoring** using VADER NLP  
- 🧠 **LSTM-based Price Forecasting** with TensorFlow  
- 📈 **Technical Indicators**  
  - 14-day **Relative Strength Index (RSI)**  
  - 50-day **Simple Moving Average (SMA)**  
- 🔄 **Live Market Data** via YFinance  
- 🖥️ **Interactive Web Interface** using Streamlit

---

## 📦 Requirements

Built and tested with **Python 3.9**  
Install dependencies via `pip`:
```
vaderSentiment==3.3.2
joblib==1.3.2
streamlit==1.29.0
tensorflow==2.9.1
yfinance==0.2.31
```
---

## 🚀 Quick Start

### 1. Clone the Repository
git clone https://github.com/yourusername/news2nifty.git
cd news2nifty
2. Install Required Packages

pip install required packages [Note:-Tensorflow is not yet available for python versions above 3.10 so it is suggested to use python 3.9
```
vaderSentiment==3.3.2
joblib==1.3.2
streamlit==1.29.0
tensorflow==2.9.1
yfinance==0.2.31
```
Note: You’ll also need the following files in the root directory:
```
model.h5 – Trained LSTM model

scaler.pkl – MinMaxScaler used for feature scaling
```

3. Run the Streamlit App
```
streamlit run fin.py
```
🧪 Sample Input (for Testing)
Paste the following into the input box:

```
"RBI maintains repo rate at 6.5% amid inflation concerns"
```
The app will:

Extract the headline sentiment.

Combine it with technical indicators.

Feed it to the LSTM model.

Predict and visualize the price impact on the NIFTY 50 index.

📌 Project Structure

news2nifty/
├── fin.py               # Streamlit frontend and logic
├── model.h5             # Pretrained LSTM model
├── scaler.pkl           # Feature scaler
└── README.md            # You're here

This project is for educational and research purposes only. It is not financial advice. Always do your own due diligence before making any investment.
![Screenshot (417)](https://github.com/user-attachments/assets/481e77a2-4f57-43b6-8c1c-e242343aa380)

![Screenshot (418)](https://github.com/user-attachments/assets/02da2a03-6120-495d-8f31-528c4b3af0dc)

Made by Jaismeen Kaur 230121030
