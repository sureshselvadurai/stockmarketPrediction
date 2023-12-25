import pandas as pd
from data_processing.feature_engineering.stock_feature_engineer import StockFeatureEngineer


class FeatureEngineer:

    def __init__(self, stock_name):
        self._stock_name = stock_name
        data = pd.read_csv(f"data/raw/{stock_name.lower()}_cleaned.csv", parse_dates=['Timestamp'])
        self._data = data

    def create(self):
        stock_feature_engineer = StockFeatureEngineer(self._data)
        stock_feature_engineer.normalize_data()
        stock_feature_engineer.create_lag_feature()
        stock_feature_engineer.create_time_feature()
        stock_feature_engineer.create_log_normalization();

        stock_feature_engineer.write_to_csv(self._stock_name)
        pass
