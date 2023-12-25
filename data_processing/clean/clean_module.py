from data_processing.clean.stock_data_cleaner import StockDataCleaner
import pandas as pd


class Clean:

    def __init__(self, stock_name):
        self._stock_name = stock_name
        data = pd.read_csv(f"data/raw/{stock_name.lower()}.csv", parse_dates=['Timestamp'], index_col='Timestamp')
        self._data = data

    def clean_data(self):
        stock_cleaner = StockDataCleaner(self._data)
        stock_cleaner.remove_duplicate_rows()
        stock_cleaner.sort_by_timestamp()
        stock_cleaner.fill_missing_values()
        stock_cleaner.normalize_data()
        stock_cleaner.create_lag_feature()

        stock_cleaner.write_to_csv(self._stock_name)
        pass
