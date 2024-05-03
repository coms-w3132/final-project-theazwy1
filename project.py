import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

# print(sp500)

sp500.plot.line(y='Close', use_index=True)
# plt.show()

del sp500['Dividends']
del sp500['Stock Splits']

sp500['Tomorrow'] = sp500['Close'].shift(-1)
sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)

sp500 = sp500.loc["1990-01-01":].copy()
print(sp500)