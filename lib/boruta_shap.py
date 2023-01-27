from typing import List, Dict, Any
import pandas as pd
from IPython.display import display

import lightgbm as lgb
from BorutaShap import BorutaShap, load_data

from lib.split_utils import split_train_test
from lib.combinations import print_cols
from lib.data_series import DataSerie
from lib.train import train_data_series, TrainResults
from lib.shap_utils import explain_model


#################################################################
# THERE IS A LOT OF COPY-PASTE IN boruta.py and boruta_shap.py  #
#################################################################


def get_classifier():

    return lgb.LGBMRegressor(
        objective="regression_l2",
        boosting="gbdt",
        learning_rate=0.05,
        max_bin=63,
        max_depth=16,
        num_leaves=5,
        num_boost_round=100,
        verbose=-1,
    )


def add_cols(
    selection_results: Dict[str, Dict[str, bool]],
    selected_cols: List[str],
    serie_description: str,
) -> Dict[str, Dict[str, bool]]:

    for selected_col in selected_cols:
        if selected_col not in selection_results:
            selection_results[selected_col] = {}
        selection_results[selected_col][serie_description] = 1

    return selection_results


def color_one(val):
    if val == 1:
        return "background-color: gold"


def boruta_shap_select_cols(
    df: pd.DataFrame,
    cols: List[str],
    target_col: str,
    importance_measure: str,
    which_features: str,
    random_state: int,
    details: bool,
):

    df = df.copy()

    feature_selector = BorutaShap(
        model=get_classifier(),
        importance_measure=importance_measure,
        classification=False,
    )

    feature_selector.fit(
        X=df[cols], y=df[target_col], n_trials=100, random_state=random_state
    )

    if details:
        feature_selector.plot(
            which_features=which_features, X_size=8, figsize=(12, 8), y_scale="log"
        )

    return list(feature_selector.Subset().columns)


def boruta_shap_selection(
    data_serie: DataSerie,
    cols: List[str],
    n_folds: int,
    importance_measure: str,
    which_feature: str,
    random_state: int,
    details: bool,
) -> List[str]:

    boruta_results = {}

    splits = split_train_test(data_serie.input_df, n_folds=n_folds)
    for split_number, split in enumerate(splits):

        boruta_results = add_cols(
            boruta_results,
            boruta_shap_select_cols(
                split.train_df,
                cols,
                data_serie.target_col,
                importance_measure,
                which_feature,
                random_state,
                False,
            ),
            f"train_{split_number}",
        )

    boruta_results = add_cols(
        boruta_results,
        boruta_shap_select_cols(
            data_serie.input_df,
            cols,
            data_serie.target_col,
            importance_measure,
            which_feature,
            random_state,
            True,
        ),
        f"input",
    )

    boruta_df = pd.DataFrame.from_dict(
        boruta_results,
        orient="index",
        columns=[f"train_{split_number}" for split_number in range(0, len(splits))]
        + ["input"],
    )
    boruta_df.fillna(0, inplace=True)
    for col in list(boruta_df.columns):
        boruta_df[col] = boruta_df[col].astype(int)
    if details:
        display(boruta_df.style.applymap(color_one))

    selected_cols = list(boruta_df[boruta_df["input"] == 1].index)
    print_cols(selected_cols)

    return selected_cols


def boruta_shap_selection_and_train(
    data_serie_name: DataSerie,
    cols: List[str],
    n_folds: int,
    importance_measure: str,
    which_features: str,
    random_state: int,
    details: bool,
    LGB_PARAMS_BASE: Dict[str, Any],
    DATA_SERIES: Dict[str, DataSerie],
    BASE_COLS_2: List[str],
    model_suffixes_filter: List[str],
    BASE_RESULTS: TrainResults,
    BASE_RESULTS_2: TrainResults,
) -> List[str]:

    boruta_cols = boruta_shap_selection(
        DATA_SERIES[data_serie_name],
        cols,
        n_folds,
        importance_measure,
        which_features,
        random_state,
        details,
    )

    Y_RESULTS = train_data_series(
        LGB_PARAMS_BASE,
        DATA_SERIES,
        boruta_cols,
        n_folds,
        model_suffixes_filter=model_suffixes_filter,
    )

    Y_RESULTS.print_errors(
        [BASE_RESULTS, BASE_RESULTS_2], details="avg", model_suffixes_filter=["l2"]
    )

    explain_model(Y_RESULTS, data_serie_name=data_serie_name, folds=range(0, n_folds))

    Y_RESULTS = train_data_series(
        LGB_PARAMS_BASE,
        DATA_SERIES,
        list(set(BASE_COLS_2 + boruta_cols)),
        n_folds,
        model_suffixes_filter=model_suffixes_filter,
    )

    Y_RESULTS.print_errors(
        [BASE_RESULTS, BASE_RESULTS_2], details="avg", model_suffixes_filter=["l2"]
    )

    explain_model(Y_RESULTS, data_serie_name=data_serie_name, folds=range(0, n_folds))
