import yfinance as yf

print(f"yfinance version: {yf.__version__}")
print("Attempting to download ^NSEI (NIFTY 50) data...")
print()

try:
    data = yf.download("^NSEI", period="60d", interval="1d")
    print(f"Rows returned: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print()
    if data.empty:
        print("Empty result - yfinance ran without error but returned no data.")
        print("This usually means: outdated yfinance version incompatible with")
        print("Yahoo current API, a network/firewall block, or Yahoo rate-limiting.")
    else:
        print("Data fetched successfully. Last 3 rows:")
        print(data.tail(3))
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
