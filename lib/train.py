from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import pandas as pd
import numpy as np
import lightgbm as lgb
from math import sqrt

from lib.metrics import RMSE, ErrorsInfo, errors_info_keys_generator
from lib.data_series import DataSerie


def custom_asymmetric_train(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    residual = (y_pred - y_true).astype("float")
    grad = np.where(y_true < 4, -2 * 5.0 * residual, -2 * residual)
    hess = np.where(y_true < 4, 2 * 5.0, 2.0)
    return grad, hess


def custom_asymmetric_valid(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    residual = (y_pred - y_true).astype("float")
    loss = np.where(y_true < 4, (residual ** 2) * 5.0, residual ** 2)
    return "custom_asymmetric_eval", np.mean(loss), False


DEFAULT_MODEL_SUFFIXES = ["upp", "l2", "low"]


MONOTONIC_COLS = [
    "test_amh_r",
    "day_0_mii",
    "AMH",
    "AFC on the first day of the stimulation",
    "Number of cumulus denuded in previous process",
    "Number of MII oocytes in previous process",
    "amh_including_ds1",
    "ds_1_result_num_AMH",
    "amh_qual_result_num",
    "ds1_pech_licz_10_pon",
    "ds1_bubble_count",
    "prev_proc-cumulus_denuded",
    "prev_proc-day_0_mii",
    "prev_proc-mii_cells_count",
    "patient_age"
    # "ds1_licz_pech",
    # "bmi",
    # 'ds1_3_dawka_dzienna',
    # 'ds4_7_dawka_dzienna',
    # 'gonadotropiny_na_bmi_1_3',
    # 'gonadotropiny_na_bmi_4_7'
    # "Age"
]


@dataclass
class FoldInfo:
    target_col: str
    train_cols: List[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    lgb_models: Dict[str, lgb.basic.Booster]
    test_errors_info: ErrorsInfo
    train_errors_info: ErrorsInfo
    evals_results: Dict[str, Dict[str, OrderedDict]]


@dataclass
class TrainInfo:
    input_df: pd.DataFrame
    target_col: str
    fold_infos: List[FoldInfo]
    avg_test_errors_info: ErrorsInfo
    avg_train_errors_info: ErrorsInfo

    def plot_learning_progress(self, plot_metrics: List[str], plot_sufixes: List[str]):

        for model_suffix in plot_sufixes:
            for metric in plot_metrics:
                for fold, fold_info in enumerate(self.fold_infos):
                    evals_result = fold_info.evals_results[model_suffix]

                    title = f"model: {model_suffix}, metric: {metric}, fold: {fold}"
                    lgb.plot_metric(evals_result, metric=metric, title=title)


@dataclass
class TrainResults:
    train_cols: List[str]
    train_infos: Dict[str, TrainInfo]

    def print_errors(
        self,
        base_results_list: Optional[List["TrainResults"]] = None,
        print_suffixes: Optional[List[str]] = None,
        print_metrics: Optional[List[str]] = None,
        print_folds: bool = False,
        print_avg: bool = False,
        print_train: bool = False,
    ):

        # Results for folds
        if print_folds:
            for model_suffix, error_metric in errors_info_keys_generator():
                if print_suffixes is not None and model_suffix not in print_suffixes:
                    continue
                if print_metrics is not None and error_metric not in print_metrics:
                    continue
                for data_serie_name, train_info in self.train_infos.items():
                    if base_results_list is not None:
                        base_test_errors_infos = [
                            base_results.train_infos[
                                data_serie_name
                            ].avg_test_errors_info
                            for base_results in base_results_list
                        ]
                        base_train_errors_infos = [
                            base_results.train_infos[
                                data_serie_name
                            ].avg_train_errors_info
                            for base_results in base_results_list
                        ]
                    else:
                        base_test_errors_infos = None
                        base_train_errors_infos = None
                    for fold, fold_info in enumerate(train_info.fold_infos):
                        fold_info.test_errors_info.print_error(
                            data_serie_name,
                            f" test_fold_{fold}",
                            model_suffix,
                            error_metric,
                            base_test_errors_infos,
                        )
                    if print_train:
                        for fold, fold_info in enumerate(train_info.fold_infos):
                            fold_info.train_errors_info.print_error(
                                data_serie_name,
                                f"train_fold_{fold}",
                                model_suffix,
                                error_metric,
                                base_train_errors_infos,
                            )
                    print("-")

        # Results for avg
        if print_avg:
            for data_serie_name, train_info in self.train_infos.items():
                prev_model_suffix = None
                for model_suffix, error_metric in errors_info_keys_generator():
                    if (
                        print_suffixes is not None
                        and model_suffix not in print_suffixes
                    ):
                        continue
                    if print_metrics is not None and error_metric not in print_metrics:
                        continue
                    if prev_model_suffix != model_suffix:
                        print("-")
                    if base_results_list is not None:
                        base_test_errors_infos = [
                            base_results.train_infos[
                                data_serie_name
                            ].avg_test_errors_info
                            for base_results in base_results_list
                        ]
                        base_train_errors_infos = [
                            base_results.train_infos[
                                data_serie_name
                            ].avg_train_errors_info
                            for base_results in base_results_list
                        ]
                    else:
                        base_test_errors_infos = None
                        base_train_errors_infos = None
                    train_info.avg_test_errors_info.print_error(
                        data_serie_name,
                        f" test_fold_avg",
                        model_suffix,
                        error_metric,
                        base_test_errors_infos,
                    )
                    if print_train:
                        train_info.avg_train_errors_info.print_error(
                            data_serie_name,
                            f"train_fold_avg",
                            model_suffix,
                            error_metric,
                            base_train_errors_infos,
                        )
                    prev_model_suffix = model_suffix
                print("-----------------------------")

    def get_merged_test_dfs_from_folds(self, data_serie_name: str) -> pd.DataFrame:
        return pd.concat(
            [
                fold_info.test_df
                for fold_info in self.train_infos[data_serie_name].fold_infos
            ]
        )

    def print_rmse_for_filter(self, filter_tuples: List[Tuple[str, str]]):

        for data_serie_name, train_info in self.train_infos.items():

            for filter_tuple in filter_tuples:
                df = pd.concat(
                    [fold_info.test_df for fold_info in train_info.fold_infos]
                )
                df = df[df[filter_tuple[0]] == filter_tuple[1]]
                rmse_mid = RMSE(df["prediction_l2"], df[train_info.target_col])

                print(
                    f'{data_serie_name.ljust(25, " ")}: '
                    f'{filter_tuple[1].ljust(35, " ")}: '
                    f'count: {str(len(df)).ljust(5, " ")}    : '
                    f'RMSE: {"%.2f"%rmse_mid}'
                )
            print("-")


def train_fold(
    LGB_PARAMS_BASE: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_cols: List[str],
    target_col: str,
    model_suffixes_filter: List[str],
    weight_col: str = None
) -> FoldInfo:

    # NOTE:
    # We pass 'num_boost_round' and 'early_stopping_round' in LGB_PARMAS_BASE
    # but we delete them from LGB_PARAMS_{suffix}
    # and pass them to lgb.train() as parameters
    # to avoid nasty warnings

    if weight_col not in train_df.columns:
        # print('Using default observation_weights')
        weight_col = 'observation_weight'
        train_df['observation_weight'] = 1
        test_df['observation_weight'] = 1
    train_dataset = lgb.Dataset(
        train_df[train_cols],
        label=train_df[target_col],
        weight=train_df[weight_col]
    )
    test_dataset = lgb.Dataset(
        test_df[train_cols],
        label=test_df[target_col],
        weight=test_df[weight_col],
        reference=train_dataset,
    )
    if len(test_df) == 0:
        valid_datasets = [train_dataset]
    else:
        valid_datasets = [train_dataset, test_dataset]

    evals_results = {model_suffix: {}
                     for model_suffix in model_suffixes_filter}

    lgb_models = {}
    if "l2" in model_suffixes_filter:
        LGB_PARAMS_L2 = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_L2["objective"] = "regression_l2"
        LGB_PARAMS_L2["monotone_constraints_method"] = "advanced"
        LGB_PARAMS_L2["monotone_constraints"] = [
            int(column in MONOTONIC_COLS) for column in train_cols
        ]
        del LGB_PARAMS_L2["num_boost_round"]
        del LGB_PARAMS_L2["early_stopping_round"]

        lgb_model_l2 = lgb.train(
            LGB_PARAMS_L2,
            train_set=train_dataset,
            valid_sets=valid_datasets,
            evals_result=evals_results["l2"],
            verbose_eval=False,
            keep_training_booster=True,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_l2"] = lgb_model_l2.predict(
                test_df[train_cols])
        train_df["prediction_l2"] = lgb_model_l2.predict(train_df[train_cols])
        lgb_models["l2"] = lgb_model_l2

    if "custom" in model_suffixes_filter:
        LGB_PARAMS_L2 = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_L2["objective"] = "regression_l2"
        LGB_PARAMS_L2["monotone_constraints_method"] = "advanced"
        LGB_PARAMS_L2["monotone_constraints"] = [
            int(column in MONOTONIC_COLS) for column in train_cols
        ]
        del LGB_PARAMS_L2["num_boost_round"]
        del LGB_PARAMS_L2["early_stopping_round"]

        lgb_model_l2 = lgb.train(
            LGB_PARAMS_L2,
            train_set=train_dataset,
            valid_sets=valid_datasets,
            evals_result=evals_results["custom"],
            fobj=custom_asymmetric_train,
            feval=custom_asymmetric_valid,
            verbose_eval=False,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_custom"] = lgb_model_l2.predict(
                test_df[train_cols])
        train_df["prediction_custom"] = lgb_model_l2.predict(
            train_df[train_cols])
        lgb_models["custom"] = lgb_model_l2

    if "h20" in model_suffixes_filter:

        train_dataset_h20 = lgb.Dataset(
            train_df[train_cols],
            label=train_df["hiper_20"],
        )
        test_dataset_h20 = lgb.Dataset(
            test_df[train_cols],
            label=test_df["hiper_20"],
            reference=train_dataset_h20,
        )
        if len(test_df) == 0:
            valid_datasets_h20 = [train_dataset_h20]
        else:
            valid_datasets_h20 = [train_dataset_h20, test_dataset_h20]

        LGB_PARAMS_H20 = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_H20["objective"] = "binary"
        LGB_PARAMS_H20["scale_pos_weight"] = 1
        LGB_PARAMS_H20["monotone_constraints_method"] = "advanced"
        LGB_PARAMS_H20["monotone_constraints"] = [
            int(column in MONOTONIC_COLS) for column in train_cols
        ]
        del LGB_PARAMS_H20["num_boost_round"]
        del LGB_PARAMS_H20["early_stopping_round"]

        lgb_model_h20 = lgb.train(
            LGB_PARAMS_H20,
            train_set=train_dataset_h20,
            valid_sets=valid_datasets_h20,
            evals_result=evals_results["h20"],
            verbose_eval=False,
            keep_training_booster=True,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_h20"] = lgb_model_h20.predict(
                test_df[train_cols])
        train_df["prediction_h20"] = lgb_model_h20.predict(
            train_df[train_cols])
        lgb_models["h20"] = lgb_model_h20

    if "h25" in model_suffixes_filter:

        train_dataset_h25 = lgb.Dataset(
            train_df[train_cols],
            label=train_df["hiper_25"],
        )
        test_dataset_h25 = lgb.Dataset(
            test_df[train_cols],
            label=test_df["hiper_25"],
            reference=train_dataset_h20,
        )
        if len(test_df) == 0:
            valid_datasets_h25 = [train_dataset_h25]
        else:
            valid_datasets_h25 = [train_dataset_h25, test_dataset_h25]

        LGB_PARAMS_H25 = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_H25["objective"] = "binary"
        LGB_PARAMS_H25["monotone_constraints_method"] = "advanced"
        LGB_PARAMS_H25["monotone_constraints"] = [
            int(column in MONOTONIC_COLS) for column in train_cols
        ]
        del LGB_PARAMS_H25["num_boost_round"]
        del LGB_PARAMS_H25["early_stopping_round"]

        lgb_model_h25 = lgb.train(
            LGB_PARAMS_H25,
            train_set=train_dataset_h25,
            valid_sets=valid_datasets_h25,
            evals_result=evals_results["h25"],
            verbose_eval=False,
            keep_training_booster=True,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_h25"] = lgb_model_h25.predict(
                test_df[train_cols])
        train_df["prediction_h25"] = lgb_model_h25.predict(
            train_df[train_cols])
        lgb_models["h25"] = lgb_model_h25

    if "log_l2" in model_suffixes_filter:

        train_dataset_log_l2 = lgb.Dataset(
            train_df[train_cols],
            label=np.log(1 + train_df[target_col]),
        )
        test_dataset_log_l2 = lgb.Dataset(
            test_df[train_cols],
            label=np.log(1 + test_df[target_col]),
            reference=train_dataset_log_l2,
        )
        if len(test_df) == 0:
            valid_datasets_log_l2 = [train_dataset_log_l2]
        else:
            valid_datasets_log_l2 = [train_dataset_log_l2, test_dataset_log_l2]

        LGB_PARAMS_LOG_L2 = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_LOG_L2["objective"] = "regression_l2"
        LGB_PARAMS_LOG_L2["monotone_constraints_method"] = "advanced"
        LGB_PARAMS_LOG_L2["monotone_constraints"] = [
            int(column in MONOTONIC_COLS) for column in train_cols
        ]
        del LGB_PARAMS_LOG_L2["num_boost_round"]
        del LGB_PARAMS_LOG_L2["early_stopping_round"]

        lgb_model_log_l2 = lgb.train(
            LGB_PARAMS_LOG_L2,
            train_set=train_dataset_log_l2,
            valid_sets=valid_datasets_log_l2,
            evals_result=evals_results["log_l2"],
            verbose_eval=False,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_log_l2"] = (
                np.exp(lgb_model_log_l2.predict(test_df[train_cols])) - 1
            )
        train_df["prediction_log_l2"] = (
            np.exp(lgb_model_log_l2.predict(train_df[train_cols])) - 1
        )
        lgb_models["log_l2"] = lgb_model_log_l2

    if "l1" in model_suffixes_filter:
        LGB_PARAMS_L1 = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_L1["objective"] = "regression_l1"
        del LGB_PARAMS_L1["num_boost_round"]
        del LGB_PARAMS_L1["early_stopping_round"]

        lgb_model_l1 = lgb.train(
            LGB_PARAMS_L1,
            train_set=train_dataset,
            valid_sets=valid_datasets,
            evals_result=evals_results["l1"],
            verbose_eval=False,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_l1"] = lgb_model_l1.predict(
                test_df[train_cols])
        train_df["prediction_l1"] = lgb_model_l1.predict(train_df[train_cols])
        lgb_models["l1"] = lgb_model_l1

    if "mape" in model_suffixes_filter:
        LGB_PARAMS_MAPE = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_MAPE["objective"] = "mape"
        del LGB_PARAMS_MAPE["num_boost_round"]
        del LGB_PARAMS_MAPE["early_stopping_round"]

        lgb_model_mape = lgb.train(
            LGB_PARAMS_MAPE,
            train_set=train_dataset,
            valid_sets=valid_datasets,
            evals_result=evals_results["mape"],
            verbose_eval=False,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_mape"] = lgb_model_mape.predict(
                test_df[train_cols])
        train_df["prediction_mape"] = lgb_model_mape.predict(
            train_df[train_cols])
        lgb_models["mape"] = lgb_model_mape

    if "upp" in model_suffixes_filter:
        LGB_PARAMS_UPP = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_UPP["objective"] = "quantile"
        LGB_PARAMS_UPP["alpha"] = 0.75
        LGB_PARAMS_UPP["monotone_constraints_method"] = None
        LGB_PARAMS_UPP["monotone_constraints"] = None
        del LGB_PARAMS_UPP["num_boost_round"]
        del LGB_PARAMS_UPP["early_stopping_round"]

        lgb_model_upp = lgb.train(
            LGB_PARAMS_UPP,
            train_set=train_dataset,
            valid_sets=valid_datasets,
            evals_result=evals_results["upp"],
            verbose_eval=False,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_upp"] = lgb_model_upp.predict(
                test_df[train_cols])
        train_df["prediction_upp"] = lgb_model_upp.predict(
            train_df[train_cols])
        lgb_models["upp"] = lgb_model_upp

    if "low" in model_suffixes_filter:
        LGB_PARAMS_LOW = LGB_PARAMS_BASE.copy()
        LGB_PARAMS_LOW["objective"] = "quantile"
        LGB_PARAMS_LOW["alpha"] = 0.25
        LGB_PARAMS_LOW["monotone_constraints_method"] = None
        LGB_PARAMS_LOW["monotone_constraints"] = None
        del LGB_PARAMS_LOW["num_boost_round"]
        del LGB_PARAMS_LOW["early_stopping_round"]

        lgb_model_low = lgb.train(
            LGB_PARAMS_LOW,
            train_set=train_dataset,
            valid_sets=valid_datasets,
            evals_result=evals_results["low"],
            verbose_eval=False,
            keep_training_booster=True,
            num_boost_round=LGB_PARAMS_BASE["num_boost_round"],
            early_stopping_rounds=LGB_PARAMS_BASE["early_stopping_round"],
        )
        if len(test_df) != 0:
            test_df["prediction_low"] = lgb_model_low.predict(
                test_df[train_cols])
        train_df["prediction_low"] = lgb_model_low.predict(
            train_df[train_cols])
        lgb_models["low"] = lgb_model_low

    fold_info = FoldInfo(
        target_col=target_col,
        train_cols=train_cols,
        train_df=train_df,
        test_df=test_df,
        lgb_models=lgb_models,
        test_errors_info=ErrorsInfo().calculate_errors(
            test_df, target_col, model_suffixes_filter
        ),
        train_errors_info=ErrorsInfo().calculate_errors(
            train_df, target_col, model_suffixes_filter
        ),
        evals_results=evals_results,
    )
    return fold_info


def train_data_serie(
    LGB_PARAMS_BASE: Dict[str, Any],
    data_serie: DataSerie,
    train_cols: List[str],
    model_suffixes_filter: List[str],
    use_weights: bool = False
) -> TrainInfo:

    for train_col in train_cols:
        if train_col not in data_serie.columns:
            print(f"WARNING: Column {train_col} not in data serie")
            return TrainInfo(
                input_df=pd.DataFrame(),
                target_col=data_serie.target_col,
                fold_infos=[],
                avg_test_errors_info=ErrorsInfo(
                    count=0,
                    target_avg=0,
                    errors_info={},
                ),
                avg_train_errors_info=ErrorsInfo(
                    count=0,
                    target_avg=0,
                    errors_info={},
                ),
            )

    fold_infos = []
    for _fold, split in enumerate(data_serie.splits):
        fold_info = train_fold(
            LGB_PARAMS_BASE,
            split.train_df.copy(),
            split.test_df.copy(),
            train_cols,
            data_serie.target_col,
            model_suffixes_filter,
            weight_col='observation_weight'*use_weights
        )
        fold_infos.append(fold_info)

    train_info = TrainInfo(
        input_df=data_serie.input_df,
        target_col=data_serie.target_col,
        fold_infos=fold_infos,
        avg_test_errors_info=ErrorsInfo().calculate_avg_test_errors(
            [fold_info.test_df for fold_info in fold_infos],
            data_serie.target_col,
            model_suffixes_filter,
        ),
        avg_train_errors_info=ErrorsInfo().calculate_avg_train_errors(
            [fold_info.train_errors_info for fold_info in fold_infos],
            model_suffixes_filter,
        ),
    )

    return train_info


def train_data_series(
    LGB_PARAMS_BASE: Dict[str, Any],
    data_series: Dict[str, DataSerie],
    train_cols: List[str],
    model_suffixes_filter: List[str],
    use_weights: bool = False
) -> TrainResults:

    train_infos = {}
    for data_serie_name in data_series.keys():
        train_infos[data_serie_name] = train_data_serie(
            LGB_PARAMS_BASE,
            data_series[data_serie_name],
            train_cols,
            model_suffixes_filter,
            use_weights
        )

    return TrainResults(train_cols=train_cols, train_infos=train_infos)
