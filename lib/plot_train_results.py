from typing import Tuple, Optional, List, Any, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lib.train import TrainInfo
from lib.metrics import RMSE
from lib.p_value import calculate_bootstrap_p_value, calculate_ttest_ind_p_value


def contains_prediction(df: pd.DataFrame) -> bool:
    if "prediction_upp" not in df.columns:
        return False
    if "prediction_low" not in df.columns:
        return False
    return True


def plot_results_sorted_by_target_and_prediction(
    train_info: TrainInfo, model_name: str, prediction_suffix: str = "l2"
) -> plt.plot:

    target_col = train_info.target_col
    df = pd.concat([fold_info.test_df for fold_info in train_info.fold_infos])

    df = df.sort_values(by=[target_col, "prediction_l2"])
    df = df.reset_index(drop=True)

    _fig = plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[target_col], "-", markersize=10, label=target_col)

    plt.plot(
        df.index,
        df[f"prediction_{prediction_suffix}"],
        "r-",
        label=f"Prediction {prediction_suffix}",
    )
    plt.plot(df.index, df["prediction_upp"], "k-")
    plt.plot(df.index, df["prediction_low"], "k-")
    plt.fill(
        np.concatenate([df.index, df.index[::-1]]),
        np.concatenate([df["prediction_upp"], df["prediction_low"][::-1]]),
        alpha=0.5,
        fc="b",
        ec="None",
        label="95% prediction interval",
    )

    plt.title(f"Model: {model_name}, target: {target_col}, N={len(df)}")
    plt.legend(loc="upper left")
    plt.xlabel(f"order by ({target_col}, prediction_{prediction_suffix})")
    plt.show()


def plot_results_sorted_by_amh_and_target(
    train_info: TrainInfo,
    model_name: str,
    source: str,
    amh_column: str = "test_amh_r",
    prediction_suffix: str = "l2",
    filter_tuple: Optional[Tuple[str, str]] = None,
) -> plt.plot:

    target_col = train_info.target_col
    if source == "input":
        df = train_info.input_df.copy()
    if source == "test":
        df = pd.concat(
            [fold_info.test_df for fold_info in train_info.fold_infos])

    if amh_column == "ds1_pech_licz_10_pon":
        df = df[df[amh_column] != 0]

    df = df.sort_values(by=[amh_column, target_col])
    df = df.reset_index(drop=True)
    if filter_tuple is not None:
        df = df[df[filter_tuple[0]] == filter_tuple[1]]

    if f"prediction_{prediction_suffix}" in df.columns:
        rmse_mid = RMSE(df[f"prediction_{prediction_suffix}"], df[target_col])
    else:
        rmse_mid = np.NaN

    _fig = plt.figure(figsize=(15, 5))
    plt.plot(df[amh_column], df[target_col], "-",
             markersize=10, label=target_col)

    if contains_prediction(df):
        plt.plot(
            df[amh_column],
            df[f"prediction_{prediction_suffix}"],
            "r-",
            label=f"Prediction {prediction_suffix}",
        )
        plt.plot(df[amh_column], df["prediction_upp"],
                 "y-", label="Prediction upper")
        plt.plot(df[amh_column], df["prediction_low"],
                 "g-", label="Prediction lower")
        plt.fill(
            np.concatenate([df[amh_column], df[amh_column][::-1]]),
            np.concatenate([df["prediction_upp"], df["prediction_low"][::-1]]),
            alpha=0.5,
            fc="b",
            ec="None",
            label="95% prediction interval",
        )

    if source == "test":
        plt.title(
            f"Model: {model_name}, source: {source}, "
            + f"target: {target_col}, filter: {filter_tuple}, "
            + f'N={len(df)}, RMSE={"%.2f"%rmse_mid}'
        )
    plt.legend(loc="upper left")
    plt.xlabel(amh_column)
    plt.show()


def plot_amh_histogram(df: pd.DataFrame, amh_column: str = "test_amh_r") -> plt.plot:

    _fig = plt.figure(figsize=(15, 5))
    bins = len(df[amh_column].unique())
    plt.hist(df[amh_column], bins=bins)
    plt.xlabel(amh_column)
    plt.ylabel("liczba przypadków")
    plt.show()


def plot_results_with_segments_and_groups(
    df: pd.DataFrame,
    target_col: str,
    target_agg: str,
    segmentation_col: str,
    top_segments: int,
    segments: Optional[List[Any]],
    groupby_col: str,
    groupby_round: int,
    groupby_min: Optional[float],
    groupby_max: Optional[float],
    groupby_dict: Optional[Dict[Any, str]],
    title: str,
    legend_location: Optional[str] = "upper left",
):

    assert target_agg in ["avg", "sum", "count", "median"]

    df = df.copy()
    df = df[~df[groupby_col].isnull()]
    if groupby_min is not None and groupby_max is not None:
        df = df[df[groupby_col].between(groupby_min, groupby_max)]

    groupby_values = set()

    if segments is None:
        segments = list(
            df[segmentation_col].value_counts().head(top_segments).index)

    segment_percentage = {}
    for segment in segments:
        segment_percentage[segment] = round(
            100 * len(df[df[segmentation_col] == segment]) / len(df)
        )

    res_dict = {}
    res_lists = {}

    for _index, row in df.iterrows():
        if row[segmentation_col] not in segments:
            continue

        groupby_value = row[groupby_col]
        if groupby_round is not None:
            groupby_value = round(groupby_value, groupby_round)
        if groupby_dict is not None:
            if groupby_value in groupby_dict:
                groupby_value = groupby_dict[groupby_value]
            else:
                groupby_value = "OTHER"
        groupby_values.add(groupby_value)

        if groupby_value not in res_dict:
            res_dict[groupby_value] = {}
            for segment in segments:
                res_dict[groupby_value][f"{target_col}_count_{segment}"] = 0
                res_dict[groupby_value][f"{target_col}_sum_{segment}"] = 0
                res_dict[groupby_value][f"{target_col}_values_{segment}"] = []

        target = row[target_col]
        if np.isnan(target):
            continue

        segment = row[segmentation_col]
        res_dict[groupby_value][f"{target_col}_count_{segment}"] += 1
        res_dict[groupby_value][f"{target_col}_sum_{segment}"] += target
        res_dict[groupby_value][f"{target_col}_values_{segment}"].append(
            target)

        if (groupby_value, segment) not in res_lists:
            res_lists[(groupby_value, segment)] = []
        res_lists[(groupby_value, segment)].append(target)
    groupby_values = sorted(groupby_values)

    res_df = pd.DataFrame.from_dict(res_dict, orient="index").sort_index()
    res_df = res_df.reset_index().rename(columns={"index": groupby_col})
    for segment in segments:
        res_df[f"{target_col}_avg_{segment}"] = (
            res_df[f"{target_col}_sum_{segment}"]
            / res_df[f"{target_col}_count_{segment}"]
        )
        res_df[f"{target_col}_median_{segment}"] = res_df[f"{target_col}_values_{segment}"].apply(
            np.median)

    p_values = {}
    if len(segments) == 2:
        for groupby_value in groupby_values:

            if (groupby_value, segments[0]) in res_lists:
                x = pd.Series(res_lists[(groupby_value, segments[0])])
            else:
                x = pd.Series()
            if (groupby_value, segments[1]) in res_lists:
                y = pd.Series(res_lists[(groupby_value, segments[1])])
            else:
                y = pd.Series()

            if len(x) > 1 and len(y) > 1:
                # p_values[groupby_value] = 1 - \
                #     calculate_bootstrap_p_value(x, y, 1000)
                p_values[groupby_value] = calculate_ttest_ind_p_value(
                    x, y)
            else:
                p_values[groupby_value] = np.NaN

    x = np.arange(len(res_df))
    width = 1 / (len(segments) + 1)

    _fig, ax = plt.subplots(figsize=(10, 5))

    rects_dict = {}
    for idx, segment in enumerate(segments):
        rects_dict[segment] = ax.bar(
            x + width * idx - width / 2,
            res_df[f"{target_col}_{target_agg}_{segment}"],
            width,
            label=f"{segment} ({segment_percentage[segment]}%)",
        )

    max_groupby_value_len = 0
    for groupby_value in groupby_values:
        if len(str(groupby_value)) > max_groupby_value_len:
            max_groupby_value_len = len(str(groupby_value))
    rotation = 0
    if max_groupby_value_len > 5:
        rotation = 60
    if len(x) > 25:
        rotation = 90

    ax.set_xticks(x)
    ax.set_xticklabels(res_df[groupby_col], rotation=rotation)
    ax.set_ylabel(f"{target_agg}({target_col})")
    ax.set_xlabel(groupby_col)
    if legend_location is not None:
        ax.legend(loc=legend_location)
    ax.set_title(title, loc="left", y=1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Display count for each bar
    for segment, rects in rects_dict.items():
        for rect, groupby_value in zip(rects, groupby_values):
            value = res_dict[groupby_value][f"{target_col}_count_{segment}"]
            height = rect.get_height()
            ax.annotate(
                f"{value}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    if len(segments) == 2:
        avg_cols = [f"{target_col}_avg_{segment}" for segment in segments]
        for idx, groupby_value in enumerate(res_df[groupby_col]):
            value = p_values[groupby_value]
            height = res_df[res_df[groupby_col] ==
                            groupby_value][avg_cols].max(axis=1)

            ax.annotate(
                f'{"%.2f"%value}',
                xy=(idx, height),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
