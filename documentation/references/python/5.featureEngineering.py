# feature_engineering.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load financial data (replace 'your_stock_data.csv' with your actual data file)
stock_data = pd.read_csv('your_stock_data.csv')

# Assuming 'Close' is the closing price column

# Lagged Returns
for i in range(1, 6):  # Create lag features for the past 5 days
    stock_data[f'Lag_{i}_Return'] = stock_data['Close'].pct_change(i)

# Moving Averages
stock_data['5_Day_MA'] = stock_data['Close'].rolling(window=5).mean()

# Relative Strength Index (RSI)
period = 14
delta = stock_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=period, min_periods=1).mean()
avg_loss = loss.rolling(window=period, min_periods=1).mean()
rs = avg_gain / avg_loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Moving Average Convergence Divergence (MACD)
short_term = 12
long_term = 26
ema_short = stock_data['Close'].ewm(span=short_term, adjust=False).mean()
ema_long = stock_data['Close'].ewm(span=long_term, adjust=False).mean()
macd_line = ema_short - ema_long
signal_line = macd_line.ewm(span=9, adjust=False).mean()
stock_data['MACD_Histogram'] = macd_line - signal_line

# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
stock_data['Scaled_Feature'] = scaler.fit_transform(stock_data[['Close']])

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
stock_data['Standardized_Feature'] = scaler.fit_transform(stock_data[['Close']])

# Display the modified dataset
print(stock_data.head())
