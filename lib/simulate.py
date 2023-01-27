from typing import Iterable, Dict, List, Tuple, Any, Optional
import pandas as pd
from IPython.display import display
import lightgbm as lgb

from lib.train import TrainResults


def simulate_case(
    lgb_models: Dict[str, lgb.basic.Booster],
    case_df: pd.DataFrame,
    simulated_cols: List[str],
    simulated_tuples: List[Tuple[Any]],
    model_suffixes_filter: List[str],
    train_cols: List[str],
    target_col: str,
) -> pd.DataFrame:

    simulated_df = pd.DataFrame()
    for simulated_tuple in simulated_tuples:

        # Update simulated cols
        # Handle 'category' cols with care
        for simulated_col_number, simulated_col in enumerate(simulated_cols):

            if str(case_df[simulated_col].dtype) == "category":
                categories = list(case_df[simulated_col].dtype.categories)
                case_df[simulated_col] = simulated_tuple[simulated_col_number]
                case_df[simulated_col] = case_df[simulated_col].astype("category")
                case_df[simulated_col].cat.set_categories(categories, inplace=True)
            else:
                case_df[simulated_col] = simulated_tuple[simulated_col_number]

        for model_suffix in model_suffixes_filter:
            case_df[f"prediction_{model_suffix}"] = lgb_models[model_suffix].predict(
                case_df[train_cols]
            )

        columns = (
            simulated_cols
            + [target_col]
            + [f"prediction_{model_suffix}" for model_suffix in model_suffixes_filter]
        )
        simulated_df = simulated_df.append(case_df[columns])

    simulated_df.reset_index(drop=True, inplace=True)
    return simulated_df


def highlight_row_matching_col_value(
    row_serie: pd.core.series.Series,
    simulated_cols: List[str],
    target_col: str,
    maximalized_col: str,
    value: float,
) -> List[str]:

    if not row_serie.loc[maximalized_col] == value:
        return ["background-color: "] * len(row_serie)

    style_array = []
    for iter_col in row_serie.index:
        if iter_col == maximalized_col:
            style_array += ["background-color: orange"]
            continue
        if iter_col == target_col:
            style_array += ["background-color: lightgreen"]
            continue
        if iter_col in simulated_cols:
            style_array += ["background-color: gold"]
            continue
        style_array += ["background-color: lightgrey"]
    return style_array


def highlight_rows(
    row_serie: pd.core.series.Series, min_value: int, max_value: int, color: str
) -> List[str]:

    style_array = []
    style_array += ["background-color: "] * min_value
    style_array += [f"background-color: {color}"] * (max_value - min_value)
    style_array += ["background-color: "] * (len(row_serie) - max_value)

    return style_array


def highlight_cols(
    col_serie: pd.core.series.Series, min_value: int, max_value: int, color: str
) -> List[str]:

    style_array = []
    style_array += ["background-color: "] * min_value
    style_array += [f"background-color: {color}"] * (max_value - min_value)
    style_array += ["background-color: "] * (len(col_serie) - max_value)

    return style_array


def highlight_max(col_serie: pd.core.series.Series, skip_rows: int) -> List[str]:

    style_array = ["background-color: "] * skip_rows
    max_value = max(col_serie[skip_rows:].values)

    for iter_row_value in col_serie[skip_rows:]:
        if iter_row_value == max_value:
            style_array += ["background-color: orange"]
        else:
            style_array += ["background-color: "]
    return style_array


def format_float(value: Any):
    if isinstance(value, float):
        return "%.2f" % value
    return value


def simulate_columns(
    train_results: TrainResults,
    data_serie_name: str,
    target_col: str,
    model_suffixes_filter: List[str],
    maximalized_model_suffix: str,
    simulated_cols: List[str],
    simulated_tuples: List[Tuple[Any]],
    folds: Iterable[int],
    sample_size: Optional[int],
    display_details: bool,
    filter_tuple: Optional[Tuple[str, Any]] = None,
):

    for simulated_col in simulated_cols:
        assert (
            simulated_col in train_results.train_cols
        ), f"Column {simulated_col} not present in train cols"

    maximalized_col = f"prediction_{maximalized_model_suffix}"
    case_cols = train_results.train_cols.copy()
    for simulated_col in simulated_cols:
        case_cols.remove(simulated_col)

    winners_df = pd.DataFrame(
        [(case_col,) + ("",) * (len(simulated_cols) - 1) for case_col in case_cols]
        + [(target_col,) + ("",) * (len(simulated_cols) - 1)]
        + simulated_tuples,
        columns=simulated_cols,
    )
    simulated_df_list = []

    data_serie = train_results.train_infos[data_serie_name]
    counter = 0
    for fold in folds:
        fold_info = data_serie.fold_infos[fold]

        sample_df = fold_info.test_df

        # Filter data
        if filter_tuple is not None:
            if filter_tuple[1] is None:
                sample_df = sample_df[sample_df[filter_tuple[0]].isnull()]
            else:
                sample_df = sample_df[sample_df[filter_tuple[0]] == filter_tuple[1]]

        if sample_size is not None and sample_size < len(sample_df):
            sample_df = sample_df.sample(sample_size)

        for pos in range(0, len(sample_df)):
            case_df = sample_df.iloc[[pos]].copy()

            simulated_df = simulate_case(
                fold_info.lgb_models,
                case_df,
                simulated_cols,
                simulated_tuples,
                model_suffixes_filter,
                fold_info.train_cols,
                target_col,
            )
            simulated_df_list += [simulated_df]

            max_value = simulated_df[maximalized_col].max()

            case_values = case_df[case_cols + [target_col]].astype(str).values.flatten()
            winners_filt = list(
                (simulated_df[maximalized_col] == max_value).astype(int)
            )
            winners_filt = simulated_df[maximalized_col]

            winners_list = list(case_values) + list(winners_filt)
            winners_df[f"{counter}"] = winners_list
            counter += 1

    display(
        winners_df.style.apply(highlight_max, skip_rows=len(case_cols) + 1, axis=0)
        .apply(
            highlight_rows,
            min_value=0,
            max_value=len(simulated_cols),
            color="lightgrey",
            axis=1,
        )
        .apply(
            highlight_cols,
            min_value=0,
            max_value=len(train_results.train_cols) - len(simulated_cols),
            color="lightblue",
            axis=0,
        )
        .apply(
            highlight_cols,
            min_value=len(train_results.train_cols) - len(simulated_cols),
            max_value=len(train_results.train_cols) - len(simulated_cols) + 1,
            color="lightgreen",
            axis=0,
        )
        .format(format_float)
    )

    if display_details:
        for simulated_df in simulated_df_list:
            max_value = simulated_df[maximalized_col].max()
            display(
                simulated_df.style.apply(
                    highlight_rows,
                    min_value=0,
                    max_value=len(simulated_cols),
                    color="lightgrey",
                    axis=1,
                )
                .apply(
                    highlight_rows,
                    min_value=len(simulated_cols),
                    max_value=len(simulated_cols) + 1,
                    color="lightgreen",
                    axis=1,
                )
                .apply(
                    highlight_row_matching_col_value,
                    simulated_cols=simulated_cols,
                    target_col=target_col,
                    maximalized_col=maximalized_col,
                    value=max_value,
                    axis=1,
                )
                .format(format_float)
            )
