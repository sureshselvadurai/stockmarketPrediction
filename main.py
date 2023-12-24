from utilities.data_loader import load_stocks, load_data_from_yahoo
from EDA.EDATest import EDATest


def main():
    stocks = load_stocks()
    for index, stock in stocks.iterrows():
        stock_name = stock['name']
        stock_value = stock['value']
        print(f"Processing stock: ({stock_name}), Value: {stock_value}")
        # data = load_data_from_yahoo(stock_value)
        # data.to_csv(f"data/raw/{stock_name.lower()}.csv", index=False)

        # Perform EDA
        eda = EDATest(stock_name)
        eda.display_info()
        eda.display_head()

        if eda.isValid():
            pass


if __name__ == "__main__":
    main()
