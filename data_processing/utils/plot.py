import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import data_processing.utils.statistics.adf_test as ADF
from data_processing.utils.statistics.adf_test import validate as ADFValidate


def isValid(data):

    # Time Series Plotting
    # plt.figure(figsize=(14, 7))
    # plt.plot(data['Close'], label='Closing Prices')
    # plt.title('Stock Price Time Series Plot')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Close Price')
    # plt.legend()
    # plt.show()

    # Summary statistics
    # summary_stats = data.describe()
    # print("\nSummary statistics:")
    # print(summary_stats)

    # Stationary Check (ADF Test)
    # ADFValidate(data)

    return None
