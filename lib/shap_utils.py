from typing import Dict, Iterable, List, Optional, Tuple, Any
import random
import re
import shap
import numpy as np
import pandas as pd
from IPython.display import display

from lib.train import TrainResults, train_data_series
from lib.data_series import DataSerie

import matplotlib.pyplot as plt


def shap_force_plot(
    train_results: TrainResults,
    data_serie_name: str,
    folds: Iterable[int],
    model_suffix: str = "l2",
) -> None:

    data_serie = train_results.train_infos[data_serie_name]
    for fold in folds:
        fold_info = data_serie.fold_infos[fold]
        explainer = shap.TreeExplainer(fold_info.lgb_models[model_suffix])

        test_df = fold_info.test_df[fold_info.train_cols]
        if len(test_df) > 500:
            print(f"Sampling data to {500} records")
            test_df = test_df.sample(500, random_state=42)

        train_shap_values = explainer.shap_values(test_df)

        plot = shap.force_plot(
            explainer.expected_value, train_shap_values, test_df[fold_info.train_cols]
        )
        display(plot)


def shap_force_plots(
    train_results: TrainResults,
    data_serie_name: str,
    folds: Iterable[int],
    model_suffix: str = "l2",
    number_of_cases: int = 0,
    filter_tuple: Optional[Tuple[str, Any]] = None,
) -> None:

    data_serie = train_results.train_infos[data_serie_name]
    for fold in folds:
        fold_info = data_serie.fold_infos[fold]
        explainer = shap.TreeExplainer(fold_info.lgb_models[model_suffix])

        test_df = fold_info.test_df

        # Filter data
        if filter_tuple is not None:
            if filter_tuple[1] is None:
                test_df = test_df[test_df[filter_tuple[0]].isnull()]
            else:
                test_df = test_df[test_df[filter_tuple[0]] == filter_tuple[1]]

        if len(test_df) == 0:
            continue

        test_shap_values = explainer.shap_values(test_df[fold_info.train_cols])

        for position in random.sample(range(0, len(test_df)), number_of_cases):

            prediction_cols = []
            if "prediction_low" in test_df.columns:
                prediction_cols.append("prediction_low")
            prediction_cols.append(f"prediction_{model_suffix}")
            if "prediction_upp" in test_df.columns:
                prediction_cols.append("prediction_upp")
            row = test_df.iloc[[position]][
                ["process_number"]
                + [fold_info.target_col]
                + prediction_cols
                + fold_info.train_cols
            ]
            display(row)

            plot = shap.force_plot(
                explainer.expected_value,
                test_shap_values[position],
                row[fold_info.train_cols],
            )
            display(plot)


def explain_model(
    train_results: TrainResults,
    data_serie_name: str,
    folds: Iterable[int],
    model_suffix: str = "l2",
    only_summary: bool = True,
    filter_tuple: Optional[Tuple[str, Any]] = None,
    shap_dict: bool = False,
) -> None:

    data_serie = train_results.train_infos[data_serie_name]
    input_df = data_serie.input_df

    feature_names = []
    for train_col in train_results.train_cols:
        feature_name = train_col

        # Columns only in test_df and test_df
        if train_col not in input_df.columns:
            feature_names.append(feature_name)
            continue
        if input_df[train_col].dtypes == np.dtype("float"):
            true_percentage = round(
                100 *
                len(input_df[~input_df[train_col].isnull()]) / len(input_df)
            )
            feature_name = f"{train_col}"
            # feature_name = f"{train_col} ({true_percentage}% filled)"
        if input_df[train_col].dtypes == np.dtype("bool"):
            true_percentage = round(
                100 * len(input_df[input_df[train_col]]) / len(input_df)
            )
            feature_name = f"{train_col} ({true_percentage}% true)"
        if re.match(r"^cause_.*", train_col):
            true_percentage = round(
                100 * len(input_df[input_df[train_col] == 1.0]) / len(input_df)
            )
            feature_name = f"{train_col} ({true_percentage}% true)"
        feature_names.append(feature_name)

    all_train_shap_values = np.empty(shape=(0, len(train_results.train_cols)))
    all_test_dfs = pd.DataFrame()
    shap_dict_vals = {}

    for fold in folds:
        fold_info = data_serie.fold_infos[fold]
        explainer = shap.TreeExplainer(fold_info.lgb_models[model_suffix])

        test_df_full = fold_info.test_df

        # Filter data
        if filter_tuple is not None:
            test_df_full = test_df_full[test_df_full[filter_tuple[0]]
                                        == filter_tuple[1]]

        test_df = test_df_full[fold_info.train_cols].copy()
        if len(test_df) == 0:
            continue

        train_shap_values = explainer.shap_values(test_df)
        train_shap_values_df = pd.DataFrame(train_shap_values, columns=[
                                            name+'_impact' for name in train_results.train_cols])
        all_train_shap_values = np.append(
            all_train_shap_values, train_shap_values, axis=0
        )
        all_test_dfs = pd.concat([all_test_dfs, test_df])
        test_df_full['prediction_l2'] = fold_info.lgb_models[model_suffix].predict(
            test_df)
        if shap_dict:
            shap_dict_vals[fold] = (test_df_full, train_shap_values_df)

        if not only_summary:
            print(f"Train, fold: {fold}")
            shap.summary_plot(
                train_shap_values, fold_info.test_df[fold_info.train_cols], sort=False
            )

    if len(all_test_dfs) == 0:
        return

    if only_summary:
        x = plt.figure(figsize=(1, 1), dpi=600)
        shap.summary_plot(
            all_train_shap_values,
            all_test_dfs[train_results.train_cols],
            feature_names=feature_names,
        )
        fig = plt.gcf()  # gcf means "get current figure"
        fig.set_figheight(11)
        fig.set_figwidth(9)
        plt.rcParams['font.size'] = '22'
        plt.show()
        y = plt.figure(figsize=(1, 1), dpi=600)
        shap.summary_plot(
            all_train_shap_values,
            all_test_dfs[train_results.train_cols],
            feature_names=feature_names,
            plot_type="bar",
        )
        fig = plt.gcf()  # gcf means "get current figure"
        fig.set_figheight(11)
        fig.set_figwidth(9)
        plt.rcParams['font.size'] = '12'
        plt.show()
        # summary_plot_detailed(
        #     all_train_shap_values,
        #     all_test_dfs[train_results.train_cols],
        #     feature_names=feature_names
        #     )
    return (all_train_shap_values, all_test_dfs[train_results.train_cols], shap_dict_vals)


def summary_plot_detailed(df_shap, df, feature_names):
    import matplotlib.pyplot as plt

    # import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = feature_names
    shap_v.columns = feature_list
    df.columns = feature_list
    df_v = df.copy().reset_index().drop("index", axis=1)
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(
        0
    )
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ["Variable", "Corr"]
    corr_df["Sign"] = np.where(corr_df["Corr"] > 0, "red", "blue")

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ["Variable", "SHAP_abs"]
    k2 = k.merge(corr_df, left_on="Variable", right_on="Variable", how="inner")
    k2 = k2.sort_values(by="SHAP_abs", ascending=False)
    k2.reset_index(inplace=True, drop=True)
    k2 = k2.iloc[:20, :].copy()
    k2 = k2.sort_values(by="SHAP_abs", ascending=True)
    colorlist = k2["Sign"]
    k2.plot.barh(
        x="Variable",
        y="SHAP_abs",
        color=colorlist,
        figsize=(10, 6),
        legend=False,
        xlabel="SHAP Value (Red = Positive Impact)",
    )
    plt.show()
    # import plotly.graph_objects as go
    # fig = go.Figure(go.Bar(
    #             x=k2.loc[:,'SHAP_abs'],
    #             y=k2.loc[:,'Variable'],
    #             marker = dict(
    #                 color = k2.loc[:,'Sign']),
    #             orientation='h'))
    # fig.update_layout(
    # autosize=False,
    # width=1200,
    # height = 500,
    # margin=dict(
    #     l=50,
    #     r=50,
    #     b=100,
    #     t=100,
    #     pad=4
    #     ))

    # fig.show()


def shap_get_feature_importance_df(
    train_results: TrainResults,
    data_series: Dict[str, DataSerie],
    model_suffix: str = "l2",
) -> pd.DataFrame:

    feature_importance_dict = {train_col: {}
                               for train_col in train_results.train_cols}
    for data_serie_name in data_series.keys():
        for fold_number, fold_info in enumerate(
            train_results.train_infos[data_serie_name].fold_infos
        ):
            explainer = shap.TreeExplainer(fold_info.lgb_models[model_suffix])
            train_shap_values = explainer.shap_values(
                fold_info.test_df[fold_info.train_cols]
            )

            for train_col, mean_abs_shap_value in zip(
                fold_info.train_cols, np.abs(train_shap_values).mean(0)
            ):

                feature_importance_dict[train_col][
                    f"{data_serie_name}_fold_{fold_number}"
                ] = mean_abs_shap_value

    feature_importance_df = pd.DataFrame.from_dict(
        feature_importance_dict, orient="index"
    )
    for data_serie_name, data_serie in data_series.items():
        data_serie_fold_keys = [
            f"{data_serie_name}_fold_{fold_number}"
            for fold_number in range(len(data_serie.splits))
        ]
        feature_importance_df[f"{data_serie_name}_folds_sum"] = feature_importance_df[
            data_serie_fold_keys
        ].sum(axis=1)
        feature_importance_df[f"{data_serie_name}_folds_max"] = feature_importance_df[
            data_serie_fold_keys
        ].max(axis=1)
        feature_importance_df[f"{data_serie_name}_folds_mean"] = feature_importance_df[
            data_serie_fold_keys
        ].mean(axis=1)
        feature_importance_df[f"{data_serie_name}_folds_count"] = (
            feature_importance_df[data_serie_fold_keys].astype(
                bool).sum(axis=1)
        )

    all_fold_keys = [
        f"{data_serie_name}_fold_{fold_number}"
        for fold_number in range(len(data_serie.splits))
        for data_serie_name, data_serie in data_series.items()
    ]
    feature_importance_df[f"all_folds_sum"] = feature_importance_df[all_fold_keys].sum(
        axis=1
    )
    feature_importance_df = feature_importance_df[
        feature_importance_df["all_folds_sum"] > 0
    ]

    return feature_importance_df


def get_most_important_cols(
    feature_importance_df: pd.DataFrame,
    data_series: Dict[str, DataSerie],
    n_head: int = 10,
) -> List[str]:

    most_important_cols = set()
    for data_serie_name in data_series.keys():
        most_important_cols.update(
            set(
                feature_importance_df[
                    feature_importance_df[f"{data_serie_name}_folds_count"] >= 3
                ].index
            )
        )
        most_important_cols.update(
            set(
                feature_importance_df.sort_values(
                    by=f"{data_serie_name}_folds_sum", ascending=False
                )
                .head(n_head)
                .index
            )
        )
        most_important_cols.update(
            set(
                feature_importance_df.sort_values(
                    by=f"{data_serie_name}_folds_max", ascending=False
                )
                .head(n_head)
                .index
            )
        )

    return list(sorted(most_important_cols))


def shap_dependence_plot(
    train_results: TrainResults,
    data_serie_name: str,
    ind: str,
    interaction_index: str,
    folds: Iterable[int],
    model_suffix: str = "l2",
    only_summary: bool = True,
) -> None:

    data_serie = train_results.train_infos[data_serie_name]
    input_df = data_serie.input_df

    feature_names = []
    for train_col in train_results.train_cols:
        feature_name = train_col
        if input_df[train_col].dtypes == np.dtype("float"):
            true_percentage = round(
                100 *
                len(input_df[~input_df[train_col].isnull()]) / len(input_df)
            )
            feature_name = f"{train_col}"
            # feature_name = f"{train_col} ({true_percentage}% filled)"
        if input_df[train_col].dtypes == np.dtype("bool"):
            true_percentage = round(
                100 * len(input_df[input_df[train_col]]) / len(input_df)
            )
            feature_name = f"{train_col} ({true_percentage}% true)"
        if re.match(r"^cause_.*", train_col):
            true_percentage = round(
                100 * len(input_df[input_df[train_col]]) / len(input_df)
            )
            feature_name = f"{train_col} ({true_percentage}% true)"
        feature_names.append(feature_name)

    all_train_shap_values = np.empty(shape=(0, len(train_results.train_cols)))
    all_test_dfs = pd.DataFrame()

    for fold in folds:
        fold_info = data_serie.fold_infos[fold]
        explainer = shap.TreeExplainer(fold_info.lgb_models[model_suffix])

        test_df = fold_info.test_df[fold_info.train_cols]
        train_shap_values = explainer.shap_values(test_df)
        all_train_shap_values = np.append(
            all_train_shap_values, train_shap_values, axis=0
        )
        all_test_dfs = pd.concat([all_test_dfs, test_df])

        if not only_summary:
            shap.dependence_plot(
                ind,
                train_shap_values,
                fold_info.test_df[fold_info.train_cols],
                interaction_index=interaction_index,
            )

    if only_summary:
        shap.dependence_plot(
            ind,
            all_train_shap_values,
            all_test_dfs[train_results.train_cols],
            interaction_index=interaction_index,
        )


def shap_get_important_cols(
    LGB_PARAMS_BASE: Dict[str, Any],
    DATA_SERIES: Dict[str, DataSerie],
    searched_train_cols: List[str],
    additional_train_cols: List[str],
    model_suffixes_filter: List[str],
) -> List[str]:

    TRAIN_RESULTS = train_data_series(
        LGB_PARAMS_BASE,
        DATA_SERIES,
        searched_train_cols + additional_train_cols,
        model_suffixes_filter=model_suffixes_filter,
    )

    results = []
    for model_suffix in model_suffixes_filter:
        feature_importance_df = shap_get_feature_importance_df(
            TRAIN_RESULTS, DATA_SERIES, model_suffix
        )

        most_important_cols = list(
            feature_importance_df.sort_values(
                by="all_folds_sum", ascending=False)
            .head(50 + len(additional_train_cols))
            .index
        )
        results += most_important_cols

    results = list(set(results))

    for additional_train_col in additional_train_cols:
        if additional_train_col in results:
            results.remove(additional_train_col)

    return results


def shap_waterfall_plots(
    train_results: TrainResults,
    data_serie_name: str,
    folds: Iterable[int],
    model_suffix: str = "l2",
    number_of_cases: int = 0,
    filter_tuple: Optional[Tuple[str, Any]] = None,
    display_frame: bool = False,
) -> None:

    data_serie = train_results.train_infos[data_serie_name]
    for fold in folds:
        fold_info = data_serie.fold_infos[fold]
        lgb_model = fold_info.lgb_models[model_suffix]
        explainer = shap.TreeExplainer(lgb_model)

        test_df = fold_info.test_df

        # Filter data
        if filter_tuple is not None:
            if filter_tuple[1] is None:
                test_df = test_df[test_df[filter_tuple[0]].isnull()]
            else:
                test_df = test_df[test_df[filter_tuple[0]] == filter_tuple[1]]

        if len(test_df) == 0:
            continue

        test_shap_values = explainer(test_df[fold_info.train_cols])
        for position in random.sample(range(0, len(test_df)), number_of_cases):

            prediction_cols = []
            if "prediction_low" in test_df.columns:
                prediction_cols.append("prediction_low")
            prediction_cols.append(f"prediction_{model_suffix}")
            if "prediction_upp" in test_df.columns:
                prediction_cols.append("prediction_upp")
            row = test_df.iloc[[position]][
                ["process_number"]
                + [fold_info.target_col]
                + prediction_cols
                + fold_info.train_cols
            ]
            if display_frame:
                display(row)
            x = plt.figure(figsize=(1, 1), dpi=600)
            shap.plots.waterfall(
                shap_values=test_shap_values[position],
            )
            fig = plt.gcf()  # gcf means "get current figure"
            fig.set_figheight(11)
            fig.set_figwidth(9)
            plt.rcParams['font.size'] = '22'
            plt.show()
