import yfinance as yf

prices = yf.download("AAPL", start="2015-01-01", end="2024-12-31")
prices["return_1d"] = prices["Close"].pct_change().shift(-1)

prices.to_csv("yahoo_2.csv",index=False)
