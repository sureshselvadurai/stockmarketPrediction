import pandas as pd
from datetime import datetime, timedelta


def concatenate_dataframes(changed_rows, not_common_df):
    """
    Concatenate two DataFrames while handling empty or all-NA entries.

    Parameters:
    - changed_rows: DataFrame
    - not_common_df: DataFrame

    Returns:
    - changes: DataFrame
    """

    changed_rows = changed_rows.fillna("NA")
    not_common_df = not_common_df.fillna("NA")

    if not changed_rows.empty and not not_common_df.empty:
        # Both DataFrames are non-empty, concatenate them
        changes = pd.concat([changed_rows, not_common_df])
    elif not changed_rows.empty:
        # Only changed_rows is non-empty, no need to concatenate
        changes = changed_rows.copy()
    elif not not_common_df.empty:
        # Only not_common_df is non-empty, no need to concatenate
        changes = not_common_df.copy()
    else:
        # Both DataFrames are empty
        changes = pd.DataFrame(columns=changed_rows.columns)

    return changes


def generate_close_timestamps(start_date_str, end_date_str):
    # Define the regular trading hours (9:30 AM to 4:00 PM)
    trading_hours_start = datetime.strptime("09:30", "%H:%M").time()
    trading_hours_end = datetime.strptime("16:00", "%H:%M").time()

    # Initialize the current timestamp with the start date and trading hours start time
    current_timestamp = datetime.strptime(start_date_str[:19], "%Y-%m-%d %H:%M:%S")

    # Create a list to store the generated close timestamps
    close_timestamps = []

    # Iterate through the days from the start date to the end date
    while current_timestamp.date() <= datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S").date():
        # Check if the current day is a weekend or holiday (for simplicity, we assume holidays are weekends)
        if current_timestamp.weekday() < 5:
            # Check if the current time is within regular trading hours
            current_time = current_timestamp.time()
            if trading_hours_start <= current_time <= trading_hours_end:
                # Add the current timestamp to the list
                close_timestamps.append(current_timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        # Increment the current timestamp by 1 hour
        current_timestamp += timedelta(hours=1)

    return close_timestamps
