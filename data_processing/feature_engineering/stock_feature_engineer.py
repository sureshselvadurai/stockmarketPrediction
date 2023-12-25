import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utilities.utils import concatenate_dataframes
from statsmodels.tsa.seasonal import seasonal_decompose


class StockFeatureEngineer:

    def __init__(self, data):
        self.df = data
        self.previous_version_df = data.copy()

    def normalize_data(self):
        target_column = 'Close'
        target = self.df[target_column].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaled = scaler.fit_transform(target)
        self.df['target_scaled'] = target_scaled
        return scaler, target_scaled

    def create_lag_feature(self):
        self.df['lag_1'] = self.df['Close'].shift(1)

    def create_time_feature(self):
        # Convert the 'Timestamp' column to datetime
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])

        # Extract day of the week
        self.df['DayOfWeek'] = self.df['Timestamp'].dt.dayofweek

        # Check if it's a holiday (assuming you have a list of holidays)
        holidays = ['2022-12-25', '2022-12-31']  # Add your holiday dates
        self.df['IsHoliday'] = self.df['Timestamp'].dt.strftime('%Y-%m-%d').isin(holidays)

        # Extract day of the month
        self.df['DayOfMonth'] = self.df['Timestamp'].dt.day

        # Extract month of the year
        self.df['MonthOfYear'] = self.df['Timestamp'].dt.month

        # Extract time of the day (morning, evening, noon)
        self.df['TimeOfDay'] = pd.cut(
            self.df['Timestamp'].dt.hour,
            bins=[0, 12, 17, 24],
            labels=['Morning', 'Evening', 'Noon'],
            include_lowest=True
        )

    def write_to_csv(self, stock_name):
        self.df.to_csv(f"data/raw/{stock_name.lower()}_processed.csv", index=True)

    def create_log_normalization(self):
        self.df['log_price'] = np.log1p(self.df['Close'])
        self.df['diff_price'] = self.df['Close'].diff()

