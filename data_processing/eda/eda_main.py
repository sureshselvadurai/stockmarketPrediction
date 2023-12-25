import pandas as pd
import data_processing.utils.plot as Plot


class EDATest:

    def __init__(self, stock_name):
        data = pd.read_csv(f"data/raw/{stock_name.lower()}.csv", parse_dates=['Timestamp'], index_col='Timestamp')
        self._data = data

    def isValid(self):
        Plot.isValid(self._data)
        return not self._data.isnull().values.any()

    def display_info(self):
        print(self._data.info())

    def display_head(self):
        print(self._data.head())
