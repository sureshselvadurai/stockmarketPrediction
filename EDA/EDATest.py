import pandas as pd


class EDATest:

    def __init__(self, stock_name):
        data = pd.read_csv(f"data/raw/{stock_name.lower()}.csv")
        self._data = data

    def isValid(self):
        return not self._data.isnull().values.any()

    def display_info(self):
        print(self._data.info())

    def display_head(self):
        print(self._data.head())
