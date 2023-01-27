from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display


def two_proprotions_test(
    success_a: int, size_a: int, success_b: int, size_b: int
) -> Tuple[float, float]:
    # http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html#Comparing-Two-Proportions

    """
    (Wald)

    A/B test for two proportions;
    given a success a trial size of group A and B compute
    its zscore and pvalue

    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    Returns
    -------
    zscore : float
        test statistic for the two proportion z-test

    pvalue : float
        p-value for the two proportion z-test
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    prop_pooled = (success_a + success_b) / (size_a + size_b)
    var = prop_pooled * (1 - prop_pooled) * (1 / size_a + 1 / size_b)
    zscore = np.abs(prop_b - prop_a) / np.sqrt(var)
    one_side = 1 - stats.norm(loc=0, scale=1).cdf(zscore)
    pvalue = one_side * 2
    return zscore, pvalue


def format_variant_values(val):

    if isinstance(val, str):
        return val

    if isinstance(val, int):
        return val

    if isinstance(val, np.int64):
        return val

    if isinstance(val, float):
        return "%.2f" % val

    if isinstance(val, dict):
        if "pvalue" in val and "gain" in val and "pvalue" in val:
            if val["pvalue"] <= 0.05:
                return f'{"%.2f"%val["value"]} ({val["gain"]}) ({"%.3f"%val["pvalue"]}) (*)'
            else:
                return f'{"%.2f"%val["value"]} ({val["gain"]}) ({"%.3f"%val["pvalue"]})'

        if "gain" in val:
            return f'{"%.2f"%val["value"]} ({val["gain"]})'

        return f'{"%.2f"%val["value"]}'


def color_statistical_important(val):

    if isinstance(val, dict) and "pvalue" in val and val["pvalue"] <= 0.05:
        return "background-color: gold"


def add_mean_and_gain(
    df: pd.DataFrame,
    result_dict: Dict[str, Any],
    result_dict_key: str,
    column: str,
    basic_factors_all: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    mean = df[column].mean()
    if basic_factors_all is None:
        gain = None
    else:
        gain = round(mean - basic_factors_all[result_dict_key]["value"], 2)
        if gain > 0:
            gain = f'+{"%.2f"%gain}'
        else:
            gain = f'{"%.2f"%gain}'

    mean_and_gain_dict = {"value": mean}
    if gain is not None:
        mean_and_gain_dict["gain"] = gain
    result_dict[result_dict_key] = mean_and_gain_dict

    return result_dict


def get_basic_factors(
    df: pd.DataFrame, basic_factors_all: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    result_dict = {}

    for result_dict_key, column in [
        ("AVG_PREV_CD", "prev_proc-cumulus_denuded"),
        ("AVG_PREV_MII", "prev_proc-day_0_mii"),
        ("AVG_AMH", "test_amh_r"),
        ("AVG_AGE", "patient_age"),
        ("AVG_DS1_10_PON", "ds1_pech_licz_10_pon"),
        ("AVG_ds_789_result_num_E2", "ds_789_result_num_E2"),
        ("AVG_CUMULUS_DENUDED", "cumulus_denuded"),
        ("AVG_MII", "day_0_mii"),
    ]:

        result_dict = add_mean_and_gain(
            df, result_dict, result_dict_key, column, basic_factors_all
        )

    return result_dict


def group_size(df: pd.DataFrame):
    return len(df["__id__"].unique())


def percentage_true(
    df: pd.DataFrame, variant: str, control_percentage: Optional[float] = None
) -> str:

    success = group_size(df[df[variant]])
    size = group_size(df)
    percentage = success / size

    if control_percentage is None:
        gain = None
    else:
        gain = round(percentage - control_percentage, 2)
        if gain > 0:
            gain = f'+{"%.2f"%gain}'
        else:
            gain = f'{"%.2f"%gain}'

    return success, size, percentage, gain


def variant_analysis(
    variants: List[str], input_df: pd.DataFrame, SHAP_INFLUENCE: Dict[str, str]
):

    patient_groups = sorted(list(input_df["patient_group"].unique()))
    poseidon_groups = sorted(list(input_df["poseidon_group"].unique()))

    results_dict = {}

    control_filter = input_df["patient_group"] == "4-control_multiple_donors"

    all_dict = {}
    all_dict["SHAP_INFLUENCE"] = ""
    all_dict["CONTROL-4-control_multiple_donors"] = group_size(input_df[control_filter])
    all_dict["-"] = ""
    for patient_group in patient_groups:
        all_dict[patient_group] = group_size(
            input_df[input_df["patient_group"] == patient_group]
        )
    all_dict["--"] = ""
    for poseidon_group in poseidon_groups:
        all_dict[poseidon_group] = group_size(
            input_df[input_df["poseidon_group"] == poseidon_group]
        )
    all_dict["---"] = ""
    basic_factors_all = get_basic_factors(input_df, None)
    for key, value in basic_factors_all.items():
        all_dict[key] = value
    results_dict["SAMPLE_SIZE / BASE VALUE"] = all_dict

    for variant in variants:
        variant_dict = {}
        if variant in SHAP_INFLUENCE:
            variant_dict["SHAP_INFLUENCE"] = SHAP_INFLUENCE[variant]
        else:
            variant_dict["SHAP_INFLUENCE"] = ""

        (
            control_success,
            control_size,
            control_percentage,
            _control_gain,
        ) = percentage_true(input_df[control_filter], variant)

        variant_dict["CONTROL-4-control_multiple_donors"] = control_percentage

        variant_dict["-"] = ""
        for patient_group in patient_groups:
            success, size, percentage, gain = percentage_true(
                input_df[input_df["patient_group"] == patient_group],
                variant,
                control_percentage,
            )

            _zscore, pvalue = two_proprotions_test(
                control_success, control_size, success, size
            )
            variant_dict[patient_group] = {
                "value": percentage,
                "gain": gain,
                "pvalue": pvalue,
            }
        variant_dict["--"] = ""
        for poseidon_group in poseidon_groups:
            success, size, percentage, gain = percentage_true(
                input_df[input_df["poseidon_group"] == poseidon_group],
                variant,
                control_percentage,
            )

            _zscore, pvalue = two_proprotions_test(
                control_success, control_size, success, size
            )
            variant_dict[poseidon_group] = {
                "value": percentage,
                "gain": gain,
                "pvalue": pvalue,
            }
        variant_dict["---"] = ""
        variant_basic_factors = get_basic_factors(
            input_df[input_df[variant]], basic_factors_all
        )
        for key, value in variant_basic_factors.items():
            variant_dict[key] = value

        results_dict[variant] = variant_dict

    result_df = pd.DataFrame.from_dict(results_dict, orient="columns")
    display(
        result_df.style.format(format_variant_values).applymap(
            color_statistical_important
        )
    )
