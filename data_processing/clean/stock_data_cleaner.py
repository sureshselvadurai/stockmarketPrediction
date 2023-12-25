import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utilities.utils import concatenate_dataframes


class StockDataCleaner:
    def __init__(self, data):
        self.df = data
        self.previous_version_df = data.copy()

    def test_remover(self):
        # Remove a random row
        random_row_to_remove = np.random.choice(self.df.index)
        self.df = self.df.drop(random_row_to_remove)

        # Change the value of a random row
        random_row_to_modify = np.random.choice(self.df.index)
        random_column_to_modify = np.random.choice(self.df.columns)
        new_value = "New Value"  # You can replace this with the desired new value
        self.df.at[random_row_to_modify, random_column_to_modify] = 600

    def display_changes(self, description):
        print(f"==== {description} ====")

        changed_rows = self.df.loc[self.df.ne(self.previous_version_df).any(axis=1), :].fillna("NA")

        new_index_values = self.previous_version_df.index.difference(self.df.index)

        # Create a new DataFrame with the same columns as self.df
        not_common_df = pd.DataFrame(index=new_index_values, columns=self.df.columns)

        # Concatenate changes
        changes = concatenate_dataframes(changed_rows,not_common_df)

        # Debugging: Print the changes dataframe
        print("Changes DataFrame:")
        print(changes.shape)
        if not changes.empty:
            print(changes)

        # Update the previous version dataframe
        self.previous_version_df = self.df

    def remove_duplicate_rows(self):
        self.df.drop_duplicates(inplace=True)
        self.display_changes("Remove Duplicate Rows")

    def convert_timestamp_to_datetime(self):
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.display_changes("Convert Timestamp to Datetime")

    def sort_by_timestamp(self):
        self.df.sort_values('Timestamp', inplace=True)
        self.display_changes("Sort by Timestamp")

    def use_timestamp_as_index(self):
        self.df.set_index('Timestamp', inplace=True)
        self.display_changes("Use Timestamp as Index")

    def fill_missing_values(self):
        self.df.interpolate(method='linear', inplace=True)
        self.display_changes("Fill Missing Values")

    def normalize_data(self):
        target_column = 'Close'
        target = self.df[target_column].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaled = scaler.fit_transform(target)
        self.df['target_scaled'] = target_scaled
        self.display_changes("Normalize Data")
        return scaler, target_scaled

    def create_sequences_and_labels(self, sequence_length=10):
        sequences, next_values = [], []

        for i in range(len(self.df) - sequence_length):
            seq = self.df[i:i + sequence_length]
            label = self.df[i + sequence_length]
            sequences.append(seq)
            next_values.append(label)

        x = np.array(sequences)
        y = np.array(next_values)

        return x, y

    def write_to_csv(self, stock_name):
        self.df.to_csv(f"data/raw/{stock_name.lower()}_cleaned.csv", index=True)

    def create_lag_feature(self):
        self.df['lag_1'] = self.df['Close'].shift(1)
