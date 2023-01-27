from typing import Dict, Any, List, Tuple

from lib.train import train_data_series, TrainResults
from lib.data_series import DataSerie


def perform_base_experiments(
    LGB_PARAMS_BASE: Dict[str, Any],
    data_series: Dict[str, DataSerie],
    model_suffixes_filter: List[str],
    print_suffixes: List[str],
    base_cols_list: List[List[str]],
) -> Tuple[TrainResults]:

    BASE_RESULTS = []
    for base_cols in base_cols_list:

        train_results = train_data_series(
            LGB_PARAMS_BASE, data_series, base_cols, model_suffixes_filter
        )

        train_results.print_errors(
            BASE_RESULTS,
            print_suffixes=print_suffixes,
            print_folds=False,
            print_avg=True,
            # print_poseidon_groups=False,
        )

        BASE_RESULTS += [train_results]

    return BASE_RESULTS
