from typing import List, Dict, Tuple
from datetime import datetime
import pandas as pd

from lib.split_utils import TrainTestSplit
from lib.data_series import DataSerie
from lib.lda import add_lda_topic_from_train_as_feature
from lib.knn import add_jaccard_from_train_knn_target_average


def _create_csv_from_columns(row, columns: List[str]):
    genes = []
    for col in columns:
        if row[col] == 1:
            genes.append(col)
    return ",".join(genes)


def _add_csv_column(df: pd.DataFrame, feature_name_prefix: str, columns: List[str]):

    csv_col = f"{feature_name_prefix}_CSV"
    df[csv_col] = df.apply(_create_csv_from_columns, columns=columns, axis="columns")

    return df, [csv_col]


class FeatureGenerator:
    def __init__(self):
        pass


class CSVFeatureGenerator:
    def __init__(self):
        pass

    def add_features(
        self,
        feature_name_prefix: str,
        cols: List[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        res_train_df, res_train_cols = _add_csv_column(
            train_df, feature_name_prefix, cols
        )
        res_test_df, res_test_cols = _add_csv_column(test_df, feature_name_prefix, cols)

        return res_train_df, res_test_df, list(set(res_train_cols + res_test_cols))


class LDAFeatureGenerator:
    def __init__(self, train_number: int):

        self.train_number = train_number

    def add_features(
        self,
        feature_name_prefix: str,
        _cols: List[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        csv_col = f"{feature_name_prefix}_CSV"
        lda_topic_col = f"{feature_name_prefix}_LDA_TOPIC_TRAIN_{self.train_number}"

        train_df, test_df = add_lda_topic_from_train_as_feature(
            train_df, test_df, csv_col, lda_topic_col
        )

        return train_df, test_df, [lda_topic_col]


class KNNFeatureGenerator:
    def __init__(self, target_cols: List[str], n_neighbors_list: List[int]):

        self.target_cols = target_cols
        self.n_neighbors_list = n_neighbors_list

    def add_features(
        self,
        feature_name_prefix: str,
        _cols: List[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        knn_cols = []
        csv_col = f"{feature_name_prefix}_CSV"
        for n_neighbors in self.n_neighbors_list:
            for target_col in self.target_cols:
                knn_col = (
                    f"{feature_name_prefix}_JACCARD_{n_neighbors}NN_{target_col}_avg"
                )
                train_df, test_df = add_jaccard_from_train_knn_target_average(
                    train_df,
                    test_df,
                    csv_col,
                    knn_col,
                    target_col,
                    n_neighbors,
                )
                knn_cols += [knn_col]

        return train_df, test_df, list(set(knn_cols))


def add_features_for_split(
    split: TrainTestSplit,
    col_sets_as_dict: Dict[str, List[str]],
    feature_generator: FeatureGenerator,
) -> Tuple[TrainTestSplit, List[str]]:

    train_df = split.train_df
    test_df = split.test_df
    result_cols = []

    for feature_name_prefix, cols in col_sets_as_dict.items():
        start_time = datetime.now()
        print(
            f" - {type(feature_generator).__name__} : "
            f"{feature_name_prefix.ljust(40, ' ')}: ",
            end="",
        )
        res_train_df, res_test_df, res_cols = feature_generator.add_features(
            feature_name_prefix,
            cols,
            train_df,
            test_df,
        )
        train_df = res_train_df
        test_df = res_test_df
        result_cols += res_cols
        print(f"DONE: {format(datetime.now() - start_time)}")

    return (
        TrainTestSplit(
            train_df=train_df,
            test_df=test_df,
        ),
        list(set(result_cols)),
    )


def add_features_for_data_serie(
    data_serie: DataSerie,
    col_sets_as_dict: Dict[str, List[str]],
    feature_generator: FeatureGenerator,
) -> Tuple[DataSerie, List[str]]:

    result_splits = []
    result_cols = []

    for split_number, split in enumerate(data_serie.splits):
        print(f"SPLIT: {split_number}")
        res_split, res_cols = add_features_for_split(
            split,
            col_sets_as_dict,
            feature_generator,
        )
        result_splits += [res_split]
        result_cols += res_cols
    result_cols = list(set(result_cols))

    return (
        DataSerie(
            target_col=data_serie.target_col,
            input_df=data_serie.input_df,
            splits=result_splits,
            columns=data_serie.columns + result_cols,
        ),
        result_cols,
    )


def add_features_for_data_series(
    DATA_SERIES: Dict[str, DataSerie],
    col_sets_as_dict: Dict[str, List[str]],
    feature_generator: FeatureGenerator,
) -> Tuple[Dict[str, DataSerie], List[str]]:

    result_data_series = {}
    result_cols = []
    for data_serie_name, data_serie in DATA_SERIES.items():
        res_data_serie, res_cols = add_features_for_data_serie(
            data_serie,
            col_sets_as_dict,
            feature_generator,
        )
        result_data_series[data_serie_name] = res_data_serie
        result_cols += res_cols

    return result_data_series, list(set(result_cols))


def add_features_csv(
    DATA_SERIES: Dict[str, DataSerie], col_sets_as_dict: Dict[str, List[str]]
) -> Tuple[Dict[str, DataSerie], List[str]]:

    feature_generator = CSVFeatureGenerator()
    return add_features_for_data_series(
        DATA_SERIES,
        col_sets_as_dict,
        feature_generator,
    )


def add_features_lda(
    DATA_SERIES: Dict[str, DataSerie],
    col_sets_as_dict: Dict[str, List[str]],
    number_of_trains: int,
) -> Tuple[Dict[str, DataSerie], List[str]]:

    result_cols = []
    for train_number in range(0, number_of_trains):
        print("================================================")
        print(f"TRAIN: {train_number}")
        print("================================================")
        start_time = datetime.now()
        feature_generator = LDAFeatureGenerator(train_number)
        DATA_SERIES, res_cols = add_features_for_data_series(
            DATA_SERIES,
            col_sets_as_dict,
            feature_generator,
        )
        result_cols += res_cols
        print(f"DONE: {format(datetime.now() - start_time)}")

    return DATA_SERIES, list(set(result_cols))


def add_features_knn(
    DATA_SERIES: Dict[str, DataSerie],
    col_sets_as_dict: Dict[str, List[str]],
    target_cols: List[str],
    n_neighbors_list: List[int],
) -> Tuple[Dict[str, DataSerie], List[str]]:

    feature_generator = KNNFeatureGenerator(target_cols, n_neighbors_list)
    return add_features_for_data_series(
        DATA_SERIES,
        col_sets_as_dict,
        feature_generator,
    )
