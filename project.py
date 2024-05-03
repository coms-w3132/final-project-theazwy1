import yfinance as yf

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

print(sp500)

sp500.plot.line(y="Close", use_index=True)

del sp500['Dividends']
del sp500['Stock Splits']