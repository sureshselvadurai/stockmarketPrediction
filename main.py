from utilities.data_loader import load_stocks, load_data_from_yahoo
from data_processing.eda.eda_main import EDATest
from data_processing.clean.clean_module import Clean
from data_processing.feature_engineering.feature_engineer import FeatureEngineer


def main():
    stocks = load_stocks()
    for index, stock in stocks.iterrows():
        stock_name = stock['name']
        stock_value = stock['value']
        print(f"Processing stock: ({stock_name}), Value: {stock_value}")
        data = load_data_from_yahoo(stock_value)
        data.to_csv(f"data/raw/{stock_name.lower()}.csv", index=False)

        # Clean Data
        clean_data = Clean(stock_name)
        clean_data.clean_data()

        # FeatureEngineering
        feature_engineer = FeatureEngineer(stock_name)
        feature_engineer.create()

        # Perform data_processing
        eda = EDATest(stock_name)
        # eda.display_info()
        # eda.display_head()

        if eda.isValid():
            pass


if __name__ == "__main__":
    main()
