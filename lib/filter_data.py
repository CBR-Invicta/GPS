import pandas as pd


def filter_data(df: pd.DataFrame, filter_serie: pd.Series) -> pd.DataFrame:

    filtered_df = df[filter_serie]
    # print(f'Removed records:  {len(df) - len(filtered_df)}')

    return filtered_df
