# Install necessary libraries
# !pip install pandas matplotlib yfinance

# Import libraries
import numpy as np
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

# Display the first few rows of the data
print(stock_data.head())

# Exploratory Data Analysis (data_processing)
# Summary statistics
print("\nSummary statistics:")
print(stock_data.describe())

# Plotting the closing prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Adj Close'], label='Adjusted Close Price')
plt.title(f"{ticker} Adjusted Close Price Over Time")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Basic Financial Calculations with Python
# Calculate daily returns
stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()

# Calculate cumulative returns
stock_data['Cumulative_Return'] = (1 + stock_data['Daily_Return']).cumprod()

# Plotting the closing prices and cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Adj Close'], label='Adjusted Close Price')
plt.plot(stock_data['Cumulative_Return'], label='Cumulative Return')
plt.title(f"{ticker} Stock Price and Cumulative Return")
plt.xlabel('Date')
plt.ylabel('Price/Return')
plt.legend()
plt.show()
