from typing import List
from math import floor, ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_agg_df_filled_with_zeros(
    df: pd.DataFrame,
    key_column: str,
    nan_column: str,
    min_range: int,
    max_range: int,
    round_digits: int,
) -> pd.DataFrame:

    agg_df = (
        df.groupby(by=key_column)
        .mean()
        .sort_index()
        .reset_index()[[key_column, nan_column]]
    )

    step = 1
    if round_digits > 0:
        for _i in range(0, round_digits):
            step /= 10
    if round_digits < 0:
        for _i in range(0, round_digits, -1):
            step *= 10
    missing_keys = [
        round(i, round_digits)
        for i in np.arange(min_range, max_range, step)
        if round(i, round_digits) not in agg_df[key_column].values
    ]
    for missing_key in missing_keys:
        agg_df = agg_df.append(
            {key_column: missing_key, nan_column: np.NaN}, ignore_index=True
        )

    agg_df = agg_df.sort_values(by=key_column)
    agg_df = agg_df[agg_df[key_column] >= min_range]
    agg_df = agg_df[agg_df[key_column] <= max_range]
    agg_df.fillna(0, inplace=True)

    return agg_df


def get_range_for_n_percent_cases(
    df: pd.DataFrame, key_column: str, n_percent: int
) -> int:

    total = 0
    min_range = floor(df[key_column].min())

    vc = df[key_column].value_counts().sort_index()
    for index, value in zip(vc.index, vc.values):
        total += value
        if total >= (n_percent / 100) * len(df[~df[key_column].isnull()]):
            return min_range, ceil(index)

    return min_range, ceil(df[key_column].max())


def plot_column_relation_with_targets(
    df: pd.DataFrame,
    column: str,
    target_cols: List[str],
    n_percent: int,
    round_digits: int = 0,
) -> plt.plot:

    df = df.copy()
    df[f"{column}_rounded"] = df[column].round(round_digits)

    _fig, axes = plt.subplots(1, 2 + len(target_cols), figsize=(15, 5))
    plt.title(f"{column}: N = {len(df[~df[column].isnull()])}")

    axes[0].hist(df[column])
    axes[0].set_ylabel("count")
    axes[0].set_xlabel(column)

    min_range, max_range = get_range_for_n_percent_cases(df, column, n_percent)

    axes[1].hist(df[column], range=(min_range, max_range))
    axes[1].set_ylabel("count")
    axes[1].set_xlabel(f"{column}: ({min_range} - {max_range})")

    for ax_number, target_col in enumerate(target_cols):

        agg_df = get_agg_df_filled_with_zeros(
            df, f"{column}_rounded", target_col, min_range, max_range, round_digits
        )
        axes[2 + ax_number].plot(agg_df[f"{column}_rounded"], agg_df[target_col])
        axes[2 + ax_number].set_ylabel(target_col)
        axes[2 + ax_number].set_xlabel(f"{column}_rounded")

    plt.show()
