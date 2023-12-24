import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file into a DataFrame
file_path = '../data/raw/apple.csv'  # Replace with the actual file path
df = pd.read_csv(file_path, parse_dates=['Timestamp'])

# Convert the 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Drop rows with missing or invalid timestamp values
df = df.dropna(subset=['Timestamp'])

# Group by date and count the number of rows for each date
grouped = df.groupby(df['Timestamp'].dt.date).size().reset_index(name='count')

# Get the minimum and maximum dates
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

# Create a date range
date_range = pd.date_range(start=min_date, end=max_date, freq='D').date

# Merge the date range with the grouped data
result = pd.DataFrame({'Timestamp': date_range})
result = pd.merge(result, grouped, on='Timestamp', how='left')

# Fill NaN values with 0 for dates with no values
result['count'] = result['count'].fillna(0).astype(int)

# Find dates where the number of rows is not equal to 7
dates_with_invalid_rows = result[result['count'] != 7]['Timestamp']

# Print the result
print("Dates with number of rows not equal to 7:")
print(dates_with_invalid_rows)
