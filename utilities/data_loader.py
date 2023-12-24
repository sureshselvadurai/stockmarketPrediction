import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta


def load_data_from_yahoo(stock_symbol):
    # Load hourly stock data from Yahoo Finance for the past year
    yf.pdr_override()  # Enable Yahoo Finance API

    # Set the end date to today
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Set the start date to one year ago
    start_date_hourly = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch hourly interval data
    hourly_data = pdr.get_data_yahoo(stock_symbol, start=start_date_hourly, end=end_date, interval='60m')

    # Add 'Timestamp' column with datetime index
    hourly_data['Timestamp'] = hourly_data.index

    return hourly_data


def load_stocks():
    # Load information about different stocks
    stocks_path = "input/stocks.csv"
    stocks = pd.read_csv(stocks_path)
    return stocks
