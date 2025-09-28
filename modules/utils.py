import pandas as pd
from typing import List, Tuple


def safe_merge_many(
    dfs: List[pd.DataFrame], how: str = "left", validate: str = "one_to_one"
) -> pd.DataFrame:
    """
    Merge multiple DataFrames by index using pandas merge,
    with validation to ensure correct join behavior.

    Parameters:
    - dfs: List of DataFrames to merge (must share the same index).
    - how: Type of merge (default: 'inner').
    - validate: pandas merge validation mode (default: 'one_to_one').

    Returns:
    - A single merged DataFrame.
    """
    if not dfs:
        raise ValueError("List of DataFrames is empty.")

    result = dfs[0]
    for i, df in enumerate(dfs[1:], start=2):
        try:
            result = result.merge(
                df, left_index=True, right_index=True, how=how, validate=validate
            )
        except Exception as e:
            raise RuntimeError(f"Merge failed at step {i} with shape {df.shape}: {e}")

    return result


def time_split(
    df: pd.DataFrame, start: str, end: str, *, tz: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given test start and end dates, split the DataFrame into 1-year train and 1-month test.

    Parameters:
    - df: DataFrame with DatetimeIndex
    - start: start of test period (e.g., "2024-09-01")
    - end: end of test period (e.g., "2024-09-30")
    - tz: timezone (e.g., "Europe/Riga")

    Returns:
    - (train_df, test_df)
    """
    start_ts = pd.Timestamp(start + " 00:00", tz=tz)
    end_ts = pd.Timestamp(end + " 23:00", tz=tz)

    train_start = start_ts - pd.DateOffset(years=1)
    train_end = start_ts - pd.Timedelta(hours=1)

    train_df = df.loc[train_start:train_end]
    test_df = df.loc[start_ts:end_ts]

    return train_df, test_df
