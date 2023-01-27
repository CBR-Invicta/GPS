from typing import Generator, List, Dict, Any, Optional
from datetime import timedelta
import json
from itertools import combinations, chain, tee
import pandas as pd
from tqdm import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import lightgbm as lgb

from lib.data_series import DataSerie
from lib.train import train_data_series, TrainResults

from lib.data_series import prepare_data_serie


def color_negative(val):
    if val < 0:
        return "background-color: gold"


def print_cols(cols: List[Any]):

    print(f"Number of columns: {len(cols)}")
    print("-")

    for col in cols:
        print(f'    "{col}",')
    print("-")


def compare_cols(cols1: List[str], cols2: List[str]) -> None:

    if set(cols1) == set(cols2):
        print("Column lists are equal.")
    else:
        print("Column lists differ.")
    print("-")

    for col in sorted(list(set(cols1 + cols2))):
        print(
            f'{col.ljust(50, " ")}: '
            f'{str(col in cols1).ljust(5, " ")} : '
            f'{str(col in cols2).ljust(5, " ")}'
        )


def get_gain(
    results: TrainResults,
    base_results: TrainResults,
    model_suffix: str,
    data_serie: str,
    metric: str,
) -> float:

    return (
        results.train_infos[data_serie].avg_test_errors_info.errors_info[
            (metric, model_suffix)
        ]
        - base_results.train_infos[data_serie].avg_test_errors_info.errors_info[
            (metric, model_suffix)
        ]
    )


def gen_combinations(
    allowed_cols: List[Any],
    min_cols: Optional[int] = None,
    max_cols: Optional[int] = None,
) -> chain:

    if min_cols is None:
        min_cols = 0
    if max_cols is None:
        max_cols = len(allowed_cols)

    return chain(
        *[combinations(allowed_cols, r) for r in range(min_cols, max_cols + 1)]
    )


def gen_flattened_combinations(
    allowed_cols: List[List[str]],
    min_cols: Optional[int] = None,
    max_cols: Optional[int] = None,
) -> chain:

    return chain.from_iterable(gen_combinations(allowed_cols, min_cols, max_cols))


def gen_product_generators_2(gen1: Generator, gen2: Generator) -> Generator:

    list_gen1 = [cols for cols in gen1]
    list_gen2 = [cols for cols in gen2]

    for cols_gen1 in list_gen1:
        for cols_gen2 in list_gen2:
            yield cols_gen1 + cols_gen2


def train_generated_cols(
    cols_gen: Generator,
    lgb_params_base: Dict[str, Any],
    data_series: Dict[str, DataSerie],
    base_cols_2: List[str],
    base_results: TrainResults,
    model_suffix: str,
    sort_metric: str,
    top_n_elements: int,
    eta_it_per_s: float = 2.50,
):

    cols_gen, cols_gen_tee = tee(cols_gen)

    number_of_combinations = sum(1 for _ in cols_gen)
    print(f"# Number of column combinations: {number_of_combinations}")
    estimated_time = timedelta(seconds=(round(number_of_combinations / eta_it_per_s)))
    print(f"# ETA: {estimated_time}")

    results_dict = {}

    metrics_in_order = [sort_metric]

    for cols in tqdm(cols_gen_tee):

        try:
            results = train_data_series(
                lgb_params_base, data_series, base_cols_2 + list(cols), [model_suffix]
            )

            gain_dict = {}
            for metric in metrics_in_order:
                for data_serie_name, _data_serie in data_series.items():
                    gain_dict[f"{metric}_{data_serie_name}"] = get_gain(
                        results, base_results, model_suffix, data_serie_name, metric
                    )

            results_dict[json.dumps(list(cols))] = gain_dict

            del results
        except lgb.basic.LightGBMError as exception:
            print(f"ERROR: {exception} when training cols: {cols}")

    print("=====================================")
    print(f"model: {model_suffix}")
    print(f"sorted_by: {sort_metric}")
    print("=====================================")

    results_df = pd.DataFrame.from_dict(results_dict, orient="index")

    best_gains_dict = {}
    for data_serie_name in data_series.keys():
        for col in results_df.columns:
            best_gains_dict[col] = list(
                results_df.sort_values(by=col)[col].head(1).values
            )[0]
    best_gains_df = pd.DataFrame.from_dict(
        best_gains_dict, orient="index"
    ).reset_index()
    display(best_gains_df)

    for data_serie_name, _data_serie in data_series.items():
        display(
            results_df.sort_values(f"{sort_metric}_{data_serie_name}")
            .head(top_n_elements)
            .style.applymap(color_negative)
        )

    for data_serie_name in data_series.keys():
        print(f"{sort_metric}_{data_serie_name}")
        plt.show(results_df[f"{sort_metric}_{data_serie_name}"].hist(bins=100))

    return results_df


def train_generated_cols_sum(
    cols_gen: Generator,
    lgb_params_base: Dict[str, Any],
    data_series: Dict[str, DataSerie],
    base_cols_2: List[str],
    base_results: TrainResults,
    model_suffix: str,
    sort_metric: str,
    top_n_elements: int,
    eta_it_per_s: float = 2.50,
):

    cols_gen, cols_gen_tee = tee(cols_gen)

    number_of_combinations = sum(1 for _ in cols_gen)
    print(f"# Number of column combinations: {number_of_combinations}")
    estimated_time = timedelta(seconds=(round(number_of_combinations / eta_it_per_s)))
    print(f"# ETA: {estimated_time}")

    results_dict = {}

    metrics_in_order = [sort_metric]

    for cols in tqdm(cols_gen_tee):

        try:
            data_series["900_day_0_mii"].input_df["SOM_cols"] = (
                data_series["900_day_0_mii"].input_df[list(cols)].sum(axis=1)
            )
            data_series["900_cumulus_denuded"].input_df["SOM_cols"] = (
                data_series["900_cumulus_denuded"].input_df[list(cols)].sum(axis=1)
            )

            data_series_tmp = {}
            data_series_tmp["900_cumulus_denuded"] = prepare_data_serie(
                data_series["900_cumulus_denuded"].input_df, "cumulus_denuded", 5
            )
            data_series_tmp["900_day_0_mii"] = prepare_data_serie(
                data_series["900_day_0_mii"].input_df, "day_0_mii", 5
            )

            results = train_data_series(
                lgb_params_base,
                data_series_tmp,
                base_cols_2 + ["SOM_cols"],
                [model_suffix],
            )

            gain_dict = {}
            for metric in metrics_in_order:
                for data_serie_name, _data_serie in data_series_tmp.items():
                    gain_dict[f"{metric}_{data_serie_name}"] = get_gain(
                        results, base_results, model_suffix, data_serie_name, metric
                    )

            results_dict[json.dumps(list(cols))] = gain_dict

            del results
        except lgb.basic.LightGBMError as exception:
            print(f"ERROR: {exception} when training cols: {cols}")

    print("=====================================")
    print(f"model: {model_suffix}")
    print(f"sorted_by: {sort_metric}")
    print("=====================================")

    results_df = pd.DataFrame.from_dict(results_dict, orient="index")

    best_gains_dict = {}
    for data_serie_name in data_series.keys():
        for col in results_df.columns:
            best_gains_dict[col] = list(
                results_df.sort_values(by=col)[col].head(1).values
            )[0]
    best_gains_df = pd.DataFrame.from_dict(
        best_gains_dict, orient="index"
    ).reset_index()
    display(best_gains_df)

    for data_serie_name, _data_serie in data_series.items():
        display(
            results_df.sort_values(f"{sort_metric}_{data_serie_name}")
            .head(top_n_elements)
            .style.applymap(color_negative)
        )

    for data_serie_name in data_series.keys():
        print(f"{sort_metric}_{data_serie_name}")
        plt.show(results_df[f"{sort_metric}_{data_serie_name}"].hist(bins=100))

    return results_df


def get_cols_from_combination_results(
    results_df: pd.DataFrame,
    sort_by: str,
    head_value: int,
) -> List[str]:

    cols = []
    for res_cols in list(results_df.sort_values(by=sort_by).head(head_value).index):
        res_cols = eval(res_cols)
        cols += res_cols

    return sorted(list(set(cols)))
