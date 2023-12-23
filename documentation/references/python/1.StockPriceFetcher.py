# Install necessary libraries
# !pip install yfinance

# Import libraries
import yfinance as yf

# Setting up Jupyter Notebooks
# %matplotlib inline

# Retrieving financial data using yfinance
ticker = "AAPL"  # Apple Inc. as an example
start_date = "2022-01-01"
end_date = "2022-02-01"

# Fetch historical data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the data
print(stock_data.head())
