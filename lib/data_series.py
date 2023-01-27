from typing import Dict, Tuple, List
from dataclasses import dataclass
import pandas as pd

from lib.filter_data import filter_data
from lib.split_utils import TrainTestSplit, split_train_test


@dataclass
class DataSerie:
    input_df: pd.DataFrame
    target_col: str
    splits: List[TrainTestSplit]
    columns: List[str]


def prepare_data_serie(
    input_df: pd.DataFrame, target_col: str, n_folds: int
) -> DataSerie:

    input_df = input_df.copy()

    print("====================================")
    print(f'Original records: {len(input_df)}')
    input_df = filter_data(
        input_df, ~input_df["process_type"].isin(
            ["DAWKJ", "BIOKJ", "DD", "DS"])
    )
    input_df = filter_data(
        input_df, ~input_df["lek_Gonadotropiny"].str.contains("Elonva")
    )
    input_df = filter_data(
        input_df, input_df["ds1_3_dawka_dzienna"] < 1250)  # Elonva
    input_df = filter_data(input_df, ~input_df[target_col].isnull())
    input_df = filter_data(input_df, ~input_df["test_amh_r"].isnull())
    input_df = filter_data(input_df, input_df["test_amh_r"] < 15.0)
    input_df = filter_data(
        input_df, ~input_df['day_0_mii'].isnull())
    input_df.reset_index(inplace=True, drop=True)
    print(f'Filtered records: {len(input_df)}')
    return DataSerie(
        input_df=input_df,
        target_col=target_col,
        splits=split_train_test(input_df, n_folds=n_folds),
        columns=list(input_df.columns),
    )


def prepare_data_series(
    data_900_df: pd.DataFrame, data_2015_df: pd.DataFrame, n_folds: int
) -> Tuple[Dict[str, DataSerie]]:

    DATA_SERIES_900 = {}
    DATA_SERIES_2015 = {}
    DATA_SERIES_900["900_cumulus_denuded"] = prepare_data_serie(
        data_900_df, "cumulus_denuded", n_folds
    )
    DATA_SERIES_900["900_day_0_mii"] = prepare_data_serie(
        data_900_df, "day_0_mii", n_folds
    )
    DATA_SERIES_2015["2015_cumulus_denuded"] = prepare_data_serie(
        data_2015_df, "cumulus_denuded", n_folds
    )
    DATA_SERIES_2015["2015_day_0_mii"] = prepare_data_serie(
        data_2015_df, "day_0_mii", n_folds
    )
    DATA_SERIES_2015["2015_dslast_pech_licz"] = prepare_data_serie(
        data_2015_df, "dslast_pech_licz", n_folds
    )

    return DATA_SERIES_900, DATA_SERIES_2015, {**DATA_SERIES_900, **DATA_SERIES_2015}


def class_prepare_data_series(
    data_900_df: pd.DataFrame, data_2015_df: pd.DataFrame, n_folds: int
) -> Tuple[Dict[str, DataSerie]]:

    DATA_SERIES_900 = {}
    DATA_SERIES_2015 = {}
    DATA_SERIES_900["900_cumulus_denuded"] = prepare_data_serie(
        data_900_df, "cumulus_denuded_group", n_folds
    )
    DATA_SERIES_900["900_day_0_mii"] = prepare_data_serie(
        data_900_df, "mii_group", n_folds
    )
    DATA_SERIES_2015["2015_cumulus_denuded"] = prepare_data_serie(
        data_2015_df, "cumulus_denuded_group", n_folds
    )
    DATA_SERIES_2015["2015_day_0_mii"] = prepare_data_serie(
        data_2015_df, "mii_group", n_folds
    )
    DATA_SERIES_2015["2015_day_0_mii"] = prepare_data_serie(
        data_2015_df, "pech_group", n_folds
    )

    return DATA_SERIES_900, DATA_SERIES_2015, {**DATA_SERIES_900, **DATA_SERIES_2015}
