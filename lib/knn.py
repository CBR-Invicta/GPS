from typing import Tuple
import pandas as pd


def _jaccard_similarity(csv1: str, csv2: str) -> float:

    ww1 = set(csv1.split(","))
    ww2 = set(csv2.split(","))
    total = len(ww1.union(ww2))
    if total == 0:
        return 1.0
    common = len(ww1.intersection(ww2))
    return 1.0 * common / total


def get_jaccard_from_train_knn_target_average(
    processed_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    csv_col: str,
    knn_col: str,
    target_col: str,
    n_neighbors: int,
) -> pd.DataFrame:

    knn_target_averages = {}
    for _idx1, processed_row in processed_df.iterrows():

        processed_row_neighbors = {}
        for _idx2, neighbor_row in neighbors_df.iterrows():
            # Skip records from processed patient
            if processed_row["__id__"] == neighbor_row["__id__"]:
                continue

            processed_row_neighbors[neighbor_row["__id__"]] = {
                "jaccard_similarity": _jaccard_similarity(
                    processed_row[csv_col], neighbor_row[csv_col]
                ),
                target_col: neighbor_row[target_col],
            }
        processed_row_knn_neighbors_df = (
            pd.DataFrame.from_dict(processed_row_neighbors, orient="index")
            .sort_values(by="jaccard_similarity", ascending=False)
            .head(n_neighbors)
        )

        knn_target_averages[processed_row["__id__"]] = {
            knn_col: processed_row_knn_neighbors_df[target_col].mean()
        }

    knn_target_averages_df = (
        pd.DataFrame.from_dict(knn_target_averages, orient="index")
        .reset_index()
        .rename(columns={"index": "patient_id"})
    )

    before_merge = len(processed_df)
    processed_df = processed_df.merge(
        knn_target_averages_df, left_on="__id__", right_on="patient_id", how="inner"
    )
    after_merge = len(processed_df)
    assert before_merge == after_merge

    return processed_df


def add_jaccard_from_train_knn_target_average(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    csv_col: str,
    knn_col: str,
    target_col: str,
    n_neighbors: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    neigbors_df = (
        train_df[["__id__", csv_col, target_col]]
        .groupby(by=["__id__", csv_col])
        .mean()
        .reset_index()
    )

    train_df = get_jaccard_from_train_knn_target_average(
        train_df, neigbors_df, csv_col, knn_col, target_col, n_neighbors
    )

    test_df = get_jaccard_from_train_knn_target_average(
        test_df, neigbors_df, csv_col, knn_col, target_col, n_neighbors
    )

    return train_df, test_df
