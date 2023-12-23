# Calculating Returns
# Calculate daily returns

import numpy_financial as npf  # Import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Setting up Jupyter Notebooks
# %matplotlib inline

# Retrieving financial data using yfinance
ticker = "AAPL"  # Apple Inc. as an example
start_date = "2022-01-01"
end_date = "2023-01-01"

# Fetch historical data
stock_data = yf.download(ticker, start=start_date, end=end_date)

stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()

# Calculate cumulative returns
stock_data['Cumulative_Return'] = (1 + stock_data['Daily_Return']).cumprod()

# Plotting the closing prices and cumulative returns
plt.figure(figsize=(12, 6))
# plt.plot(stock_data['Adj Close'], label='Adjusted Close Price')
plt.plot(stock_data['Cumulative_Return'], label='Cumulative Return')
plt.title(f"{ticker} Stock Price and Cumulative Return")
plt.xlabel('Date')
plt.ylabel('Price/Return')
plt.legend()
plt.show()

