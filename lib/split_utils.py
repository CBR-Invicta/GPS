from typing import Iterable, Any, Generator, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass
class TrainTestSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class RandomGroupKFold:
    def __init__(self, n_splits: int):
        self._n_splits = n_splits

    def split(
        self, X: Iterable[Any], groups: Iterable[Any], seed: int = 42
    ) -> Generator:

        # NOTE:
        # This function returns positions in X (not index values of X)

        assert len(X) == len(groups), f"Invalid parameters in RandomGroupKFold.split()"

        groups_unique = list(set(groups))
        np.random.seed(seed)
        np.random.shuffle(groups_unique)
        random_groups = np.array_split(groups_unique, self._n_splits)

        train_test_splits = []
        for random_group in random_groups:
            train_positions = []
            test_positions = []
            for position, group in zip(range(0, len(groups)), groups):
                if group in random_group:
                    test_positions.append(position)
                else:
                    train_positions.append(position)
            train_test_splits.append(
                (np.array(train_positions), np.array(test_positions))
            )

        return (train_test_split for train_test_split in train_test_splits)


def split_train_test(
    input_df: pd.DataFrame, n_folds: int = 1, id_column: str = "__id__"
) -> List[TrainTestSplit]:

    split_generator = GroupKFold(n_splits=n_folds).split(
        input_df, groups=input_df[id_column]
    )
    # split_generator = RandomGroupKFold(n_splits=n_folds).split(
    #    input_df, groups=input_df[id_column], seed=42)

    test_users_cum = set()
    sum_len_test_df = 0
    splits = []
    for fold in range(0, n_folds):
        train_positions, test_positions = next(split_generator)

        train_df = input_df.iloc[train_positions].copy()
        test_df = input_df.iloc[test_positions].copy()
        sum_len_test_df += len(test_df)

        train_users = set(train_df[id_column].unique())
        test_users = set(test_df[id_column].unique())
        assert len(train_df) + len(test_df) == len(
            input_df
        ), "Invalid split - not all records consumed by test/train"
        assert (
            test_users & train_users
        ) == set(), "Invalid split - intersection test_users/train_users not empty"
        assert (
            test_users & test_users_cum
        ) == set(), "Invalid split - intersection test_users/test_users_cum not empty"
        test_users_cum.update(test_users)

        splits.append(
            TrainTestSplit(
                train_df=train_df,
                test_df=test_df,
            )
        )

    assert len(test_users_cum) == len(
        input_df[id_column].unique()
    ), "Invalid split - not all users consumed by test"
    assert sum_len_test_df == len(
        input_df
    ), "Invalid split - not all records consumed by test"

    return splits
