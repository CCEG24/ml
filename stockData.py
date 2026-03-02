import yfinance as yf

df = yf.download("GOOGL", start="2020-01-01", end="2025-01-01", interval="1wk")
df.columns = df.columns.get_level_values(0)
df["direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()
df.to_csv("datasets/googl_stock.csv")

print(df.head())
print(df.columns.tolist())
print(df.shape)