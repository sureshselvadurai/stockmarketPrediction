import pandas as pd


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
