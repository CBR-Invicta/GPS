from typing import List, Dict, Any
import pandas as pd
from IPython.display import display

import lightgbm as lgb
from boruta import BorutaPy

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


def boruta_select_cols(
    df: pd.DataFrame,
    cols: List[str],
    target_col: str,
    perc: int,
    random_state: int,
    use_weak: bool,
    verbose: int = -1,
):

    boruta_feature_selector = BorutaPy(
        get_classifier(),
        two_step=False,  # Use the original implementation of Boruta with Bonferroni correction
        perc=perc,
        random_state=random_state,
        verbose=-1,
    )
    boruta_feature_selector.fit_transform(df[cols].values, df[target_col])

    if verbose > 0:
        for col_number, col in enumerate(cols):
            print(
                f'{str(col).ljust(75, " ")} '
                f'{str(boruta_feature_selector.ranking_[col_number]).ljust(6, " ")} '
                f'{str(boruta_feature_selector.support_[col_number]).ljust(6, " ")} '
                f'{str(boruta_feature_selector.support_weak_[col_number]).ljust(6, " ")} '
            )

    selected_cols = [
        col
        for col_number, col in enumerate(cols)
        if boruta_feature_selector.support_[col_number]
    ]
    if use_weak:
        selected_cols += [
            col
            for col_number, col in enumerate(cols)
            if boruta_feature_selector.support_weak_[col_number]
        ]

    return selected_cols


def boruta_selection(
    data_serie: DataSerie,
    cols: List[str],
    perc: int,
    use_weak: bool,
    random_state: int,
    details: bool,
) -> List[str]:

    boruta_results = {}

    for split_number, split in enumerate(data_serie.splits):

        boruta_results = add_cols(
            boruta_results,
            boruta_select_cols(
                split.train_df,
                cols,
                data_serie.target_col,
                perc,
                random_state,
                use_weak,
            ),
            f"train_{split_number}",
        )

    boruta_results = add_cols(
        boruta_results,
        boruta_select_cols(
            data_serie.input_df,
            cols,
            data_serie.target_col,
            perc,
            random_state,
            use_weak,
        ),
        f"input",
    )

    boruta_df = pd.DataFrame.from_dict(
        boruta_results,
        orient="index",
        columns=[f"train_{split_number}" for split_number in range(0, len(split))]
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


def boruta_selection_and_train(
    data_serie_name: str,
    cols: List[str],
    perc: int,
    use_weak: bool,
    random_state: int,
    details: bool,
    LGB_PARAMS_BASE: Dict[str, Any],
    DATA_SERIES: Dict[str, DataSerie],
    BASE_COLS_2: List[str],
    model_suffixes_filter: List[str],
    BASE_RESULTS: List[TrainResults],
) -> List[str]:

    boruta_cols = boruta_selection(
        DATA_SERIES[data_serie_name], cols, perc, use_weak, random_state, details
    )

    Y_RESULTS = train_data_series(
        LGB_PARAMS_BASE,
        DATA_SERIES,
        boruta_cols,
        model_suffixes_filter=model_suffixes_filter,
    )

    Y_RESULTS.print_errors(BASE_RESULTS, print_suffixes=["l2"], print_avg=True)

    explain_model(
        Y_RESULTS,
        data_serie_name=data_serie_name,
        folds=range(0, len(DATA_SERIES[data_serie_name].splits)),
    )

    Y_RESULTS = train_data_series(
        LGB_PARAMS_BASE,
        DATA_SERIES,
        list(set(BASE_COLS_2 + boruta_cols)),
        model_suffixes_filter=model_suffixes_filter,
    )

    Y_RESULTS.print_errors(BASE_RESULTS, print_suffixes=["l2"], print_avg=True)

    explain_model(
        Y_RESULTS,
        data_serie_name=data_serie_name,
        folds=range(0, len(DATA_SERIES[data_serie_name].splits)),
    )


def boruta_select_longlist(
    DATA_SERIES: Dict[str, DataSerie],
    ALL_GENES_COLS_900: List[str],
    additional_training_cols: List[str],
):

    boruta_cols_cd = boruta_select_cols(
        DATA_SERIES["900_cumulus_denuded"].input_df,
        ALL_GENES_COLS_900 + additional_training_cols,
        target_col="cumulus_denuded",
        perc=50,
        random_state=42,
        use_weak=True,
        verbose=-1,
    )

    boruta_cols_mii = boruta_select_cols(
        DATA_SERIES["900_day_0_mii"].input_df,
        ALL_GENES_COLS_900 + additional_training_cols,
        target_col="day_0_mii",
        perc=50,
        random_state=42,
        use_weak=True,
        verbose=-1,
    )

    results = list(set(boruta_cols_cd + boruta_cols_mii))

    for additional_training_col in additional_training_cols:
        if additional_training_col in results:
            results.remove(additional_training_col)

    return results


def boruta_select_shortlist(
    DATA_SERIES: Dict[str, DataSerie],
    searched_train_cols: List[str],
    additional_train_cols: List[str],
    perc: int,
):

    boruta_cols_cd = boruta_select_cols(
        DATA_SERIES["900_cumulus_denuded"].input_df,
        searched_train_cols + additional_train_cols,
        target_col="cumulus_denuded",
        perc=perc,
        random_state=42,
        use_weak=True,
    )

    boruta_cols_mii = boruta_select_cols(
        DATA_SERIES["900_day_0_mii"].input_df,
        searched_train_cols + additional_train_cols,
        target_col="day_0_mii",
        perc=perc,
        random_state=42,
        use_weak=True,
    )

    results = list(set(boruta_cols_cd) & set(boruta_cols_mii))

    for additional_train_col in additional_train_cols:
        if additional_train_col in results:
            results.remove(additional_train_col)

    return results
