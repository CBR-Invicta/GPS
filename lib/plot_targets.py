from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_targets(
    df: pd.DataFrame, title: str, target_cols: List[str], max_range: int
) -> plt.plot:

    _fig, axes = plt.subplots(1, len(target_cols), figsize=(15, 5))
    plt.title(f"{title}: N = {len(df)}")

    for ax_number, target_col in enumerate(target_cols):
        axes[ax_number].hist(df[target_col], range=(0, max_range), bins=31)
        axes[ax_number].set_ylabel("count")
        axes[ax_number].set_xlabel(target_col)

    plt.show()


def plot_scatter_and_trend(input_df: pd.DataFrame, x_col: str, y_col: str):

    _fig = plt.figure(figsize=(15, 5))

    plt.scatter(input_df[x_col], input_df[y_col], c="k", label="data", s=5)

    z = np.polyfit(input_df[x_col], input_df[y_col], 1)
    p = np.poly1d(z)
    plt.plot(input_df[x_col], p(input_df[x_col]), "r--", label="trend")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(loc="upper left")
    plt.show()
