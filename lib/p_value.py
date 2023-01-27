from typing import List, Generator
from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import tee
from IPython.display import display
from scipy.stats import ttest_ind


def calculate_t(x: pd.core.series.Series, y: pd.core.series.Series):
    # print(x, x.var())
    # print(y, y.var())
    # print(((x.var() / len(x)) + (y.var() / len(y))))
    return (x.mean() - y.mean()) / (((x.var() / len(x)) + (y.var() / len(y)))+0.000001)


def calculate_bootstrap_p_value(x: List[float], y: List[float], b_count: int) -> float:

    x = pd.Series(x)
    y = pd.Series(y)

    t = calculate_t(x, y)

    xx = x - x.mean() + pd.concat([x, y]).mean()
    yy = y - y.mean() + pd.concat([x, y]).mean()

    sum_positive = 0
    for _i in range(0, b_count):
        xxx = xx.sample(len(xx), replace=True)
        yyy = yy.sample(len(yy), replace=True)
        ttt = calculate_t(xxx, yyy)
        if ttt >= t:
            sum_positive += 1

    return sum_positive / b_count


def calculate_ttest_ind_p_value(x: List[float], y: List[float]) -> float:

    p_value = ttest_ind(x, y)[1]

    return p_value


def highlight_interesting_rows(row_serie: pd.core.series.Series) -> List[str]:

    if row_serie["is_interesting"]:
        background_color = "orange"
    else:
        background_color = "white"

    style_array = []
    style_array += [f"background-color: {background_color}"] * len(row_serie)
    return style_array


def calculate_bootstrap_p_value_for_bool_columns(
    proc_df: pd.DataFrame,
    proc_patients_df: pd.DataFrame,
    dawkj_patients_df: pd.DataFrame,
    cols_gen: Generator,
    target_col: str,
    b_count: int,
    eta_it_per_s: float = 1.4,
):

    cols_gen, cols_gen_tee = tee(cols_gen)
    number_of_combinations = sum(1 for _ in cols_gen)
    print(f"# Number of column combinations: {number_of_combinations}")
    estimated_time = timedelta(
        seconds=(round(number_of_combinations / eta_it_per_s)))
    print(f"# ETA: {estimated_time}")

    p_value_dict = {}
    for columns in tqdm(cols_gen_tee):

        proc_filter_true = np.logical_and.reduce(
            [proc_df[column] for column in columns]
        )
        proc_df_true = proc_df[proc_filter_true]
        proc_df_false = proc_df[~proc_filter_true]
        proc_number_true = len(proc_df_true)
        proc_percent_true = round(100 * len(proc_df_true) / len(proc_df), 2)
        target_true_mean = proc_df_true[target_col].mean()
        target_false_mean = proc_df_false[target_col].mean()
        if len(proc_df_true) != 0 and len(proc_df_false) != 0:
            p_value = calculate_bootstrap_p_value(
                list(proc_df_true[target_col]),
                list(proc_df_false[target_col]),
                b_count,
            )
        else:
            p_value = None
            continue

        proc_patients_filter_true = np.logical_and.reduce(
            [proc_patients_df[column] for column in columns]
        )
        proc_patients_df_true = proc_patients_df[proc_patients_filter_true]
        proc_patients_number_true = len(proc_patients_df_true)
        proc_patients_percent_true = round(
            100 * len(proc_patients_df_true) / len(proc_patients_df), 2
        )

        dawkj_patients_filter_true = np.logical_and.reduce(
            [dawkj_patients_df[column] for column in columns]
        )
        dawkj_patients_df_true = dawkj_patients_df[dawkj_patients_filter_true]
        dawkj_patients_number_true = len(dawkj_patients_df_true)
        dawkj_patients_percent_true = round(
            100 * len(dawkj_patients_df_true) / len(dawkj_patients_df), 2
        )

        is_interesting = True
        if target_true_mean is None:
            is_interesting = False
        if target_false_mean is None:
            is_interesting = False
        if abs(target_true_mean - target_false_mean) < 2:
            is_interesting = False
        if proc_patients_percent_true < 5:
            is_interesting = False
        if proc_patients_percent_true > 95:
            is_interesting = False

        p_value_dict[str(list(columns))] = {
            "p_value": p_value,
            "proc_percent_true": proc_percent_true,
            "proc_number_true": proc_number_true,
            "proc_patients_percent_true": proc_patients_percent_true,
            "proc_patients_number_true": proc_patients_number_true,
            "dawkj_patients_percent_true": dawkj_patients_percent_true,
            "dawkj_patients_number_true": dawkj_patients_number_true,
            "target_true_mean": target_true_mean,
            "target_false_mean": target_false_mean,
            "is_interesting": is_interesting,
        }

    p_value_df = pd.DataFrame.from_dict(p_value_dict, orient="index")
    p_value_df = p_value_df.sort_values(by="p_value")

    display(
        p_value_df.style.apply(
            highlight_interesting_rows,
            axis=1,
        )
    )

    return p_value_df
