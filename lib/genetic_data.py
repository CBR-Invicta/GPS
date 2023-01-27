from typing import List, Dict, Any, Optional
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from lib.variant_analysis import two_proprotions_test
from lib.data_series import (
    DataSerie,
    prepare_data_series,
    prepare_data_serie,
    class_prepare_data_series,
)
from lib.train import train_data_series

OUTPUT_DIR = "/home/mzielen/output/"
CONTROL_GROUP = "4-control_multiple_donors"
CONTROL_GROUP_NAME = f"CONTROL:{CONTROL_GROUP}"


def unique_patients_in_patient_group(df: pd.DataFrame, patient_group: str) -> int:

    patient_df = df[df["patient_group"] == patient_group]
    return len(patient_df["__id__"].unique())


def get_variant_info_dict(
    variant: str, literature_variants: List[str], boruta_variants: List[str]
) -> Dict[str, bool]:

    variant_info_dict = {}
    if variant == "TOTAL":
        variant_info_dict["IS_LITERATURE_VARIANT"] = ""
        variant_info_dict["IS_MACHINE_LEARNING_VARIANT"] = ""
    else:
        variant_info_dict["IS_LITERATURE_VARIANT"] = variant in literature_variants
        variant_info_dict["IS_MACHINE_LEARNING_VARIANT"] = variant in boruta_variants
    variant_info_dict["-"] = ""
    return variant_info_dict


def get_patient_group_sizes_dict(
    df: pd.DataFrame, patient_groups: List[str]
) -> Dict[str, int]:

    result_dict = {}
    unique_patients_control = unique_patients_in_patient_group(df, CONTROL_GROUP)
    result_dict[f"SIZE:{CONTROL_GROUP_NAME}"] = unique_patients_control
    result_dict["--"] = ""
    for patient_group in patient_groups:
        unique_patients_group = unique_patients_in_patient_group(df, patient_group)
        result_dict[f"SIZE:{patient_group}"] = unique_patients_group
    result_dict["---"] = ""

    return result_dict


def get_patient_group_percentage_dict(
    variant_sizes_dict: Dict[str, int],
    total_sizes_dict: Dict[str, int],
    patient_groups: List[str],
) -> Dict[str, float]:

    result_dict = {}

    control_success = variant_sizes_dict[f"SIZE:{CONTROL_GROUP_NAME}"]
    control_size = total_sizes_dict[f"SIZE:{CONTROL_GROUP_NAME}"]
    control_percentage = control_success / control_size
    result_dict[f"PERC_OF_TOTAL:{CONTROL_GROUP_NAME}"] = f'{"%.2f"%control_percentage}'
    result_dict["----"] = ""

    for patient_group in patient_groups:
        variant_success = variant_sizes_dict[f"SIZE:{patient_group}"]
        variant_size = total_sizes_dict[f"SIZE:{patient_group}"]
        variant_percentage = variant_success / variant_size
        gain = variant_percentage - control_percentage
        if gain > 0:
            gain = f'+{"%.2f"%gain}'
        else:
            gain = f'{"%.2f"%gain}'
        _zscore, pvalue = two_proprotions_test(
            control_success, control_size, variant_success, variant_size
        )

        result_dict[
            f"PERC_OF_TOTAL:{patient_group}"
        ] = f'{"%.2f"%variant_percentage} ({gain}) ({"%.3f"%pvalue})'

    return result_dict


def get_variant_dict(
    variant_info_dict: Dict[str, bool],
    sizes_dict: Dict[str, int],
    percentage_dict: Dict[str, float],
) -> Dict[str, Any]:

    variant_dict = {}
    variant_dict.update(variant_info_dict)
    variant_dict.update(sizes_dict)
    variant_dict.update(percentage_dict)
    return variant_dict


def get_group_proportions_df(
    df: pd.DataFrame,
    all_variants: List[str],
    literature_variants: List[str],
    boruta_variants: List[str],
) -> pd.DataFrame:

    patient_groups = sorted(list(df["patient_group"].unique()))

    results_dict = {}

    # Results for TOTAL column
    total_variant_info_dict = get_variant_info_dict(
        "TOTAL",
        literature_variants,
        boruta_variants,
    )
    total_sizes_dict = get_patient_group_sizes_dict(
        df,
        patient_groups,
    )
    total_percentage_dict = {}
    results_dict["TOTAL"] = get_variant_dict(
        total_variant_info_dict,
        total_sizes_dict,
        total_percentage_dict,
    )

    # Results for variant columns
    for variant in all_variants:
        variant_df = df[df[variant]]
        variant_variant_info_dict = get_variant_info_dict(
            variant,
            literature_variants,
            boruta_variants,
        )
        variant_sizes_dict = get_patient_group_sizes_dict(
            variant_df,
            patient_groups,
        )
        variant_percentage_dict = get_patient_group_percentage_dict(
            variant_sizes_dict,
            total_sizes_dict,
            patient_groups,
        )
        results_dict[variant] = get_variant_dict(
            variant_variant_info_dict,
            variant_sizes_dict,
            variant_percentage_dict,
        )

    result_df = pd.DataFrame.from_dict(results_dict, orient="columns")
    result_df.fillna({"TOTAL": ""}, inplace=True)
    return result_df


def get_corelations_df(
    patients_genetic_df: pd.DataFrame, all_variants: List[str]
) -> pd.DataFrame:

    patient_groups = sorted(list(patients_genetic_df["patient_group"].unique()))

    result_dict = {}

    for variant_col in all_variants:

        data1 = patients_genetic_df[variant_col]
        result_dict[variant_col] = {}

        for patient_group in patient_groups:
            data2 = patients_genetic_df["patient_group"] == patient_group

            corr_pearson, pvalue_pearson = pearsonr(data1, data2)
            _corr_spearman, _pvalue_spearman = spearmanr(data1, data2)

            result_dict[(variant_col, patient_group)] = {
                "corr_pearson": corr_pearson,
                "pvalue_pearson": pvalue_pearson,
                #'corr_spearman': corr_spearman,
                #'pvalue_spearman': pvalue_spearman,
            }
    result_df = pd.DataFrame.from_dict(result_dict, orient="index")
    result_df = result_df.rename_axis(["variant", "patient_group"]).reset_index()

    return result_df


def get_mean_and_std(df: pd.DataFrame, column: str) -> Dict[str, float]:

    return {
        f"mean({column})": df[column].mean(),
        f"std({column})": df[column].std(),
    }


def get_group_characteristics_df(data_genetic_df: pd.DataFrame) -> pd.DataFrame:

    patient_groups = sorted(list(data_genetic_df["patient_group"].unique()))

    result_dict = {}

    for patient_group in patient_groups:

        patient_group_df = data_genetic_df[
            data_genetic_df["patient_group"] == patient_group
        ]

        patient_group_dict = {}
        patient_group_dict["records"] = len(patient_group_df)
        patient_group_dict["patients"] = len(patient_group_df["__id__"].unique())
        patient_group_dict.update(get_mean_and_std(patient_group_df, "test_amh_r"))
        patient_group_dict.update(get_mean_and_std(patient_group_df, "patient_age"))
        patient_group_dict.update(
            get_mean_and_std(patient_group_df, "ds1_pech_licz_10_pon")
        )
        patient_group_dict.update(get_mean_and_std(patient_group_df, "cumulus_count"))
        patient_group_dict.update(get_mean_and_std(patient_group_df, "cumulus_denuded"))
        patient_group_dict.update(get_mean_and_std(patient_group_df, "day_0_mii"))
        result_dict[patient_group] = patient_group_dict

    result_df = pd.DataFrame.from_dict(result_dict, orient="index")

    return result_df


def output_df_with_description(
    output_filename: str, df: pd.DataFrame, description: Optional[str]
):

    filename_with_path = f"{OUTPUT_DIR}/{output_filename}"

    df.to_csv(filename_with_path, sep="\t")
    if description is not None:
        with open(filename_with_path, "a") as file:
            file.write("==============================================")
            file.write(description)


def rarest_variants_combinations(
    data_900_df,
    differences_df,
    LGB_PARAMS_BASE,
    stim=True,
    less=True,
    cutoff_diff=1.5,
    N_FOLDS=5,
    min_obs=10,
):

    if stim:
        if less:
            rarest_cols = differences_df.sort_values("diff", ascending=False)[
                (differences_df["gr1_count"] > min_obs)
                & (differences_df["gr2_count"] > min_obs)
                & (differences_df["diff"] > cutoff_diff)
                & (differences_df["stim"])
                & (differences_df["gr2_count"] < 300)
            ]["cols"].to_list()
        else:
            rarest_cols = differences_df.sort_values("diff", ascending=False)[
                (differences_df["gr1_count"] > min_obs)
                & (differences_df["gr2_count"] > min_obs)
                & (differences_df["diff"] > cutoff_diff)
                & (differences_df["stim"])
                & (differences_df["gr1_count"] < 300)
            ]["cols"].to_list()
    else:
        if less:
            rarest_cols = differences_df.sort_values("diff", ascending=False)[
                (differences_df["gr1_count"] > min_obs)
                & (differences_df["gr2_count"] > min_obs)
                & (differences_df["diff"] > cutoff_diff)
                & ~(differences_df["stim"])
                & (differences_df["gr2_count"] < 300)
            ]["cols"].to_list()
        else:
            rarest_cols = differences_df.sort_values("diff", ascending=False)[
                (differences_df["gr1_count"] > min_obs)
                & (differences_df["gr2_count"] > min_obs)
                & (differences_df["diff"] > cutoff_diff)
                & ~(differences_df["stim"])
                & (differences_df["gr1_count"] < 300)
            ]["cols"].to_list()

    rarest_cols_temp = rarest_cols.copy()
    rarest_cols_best = rarest_cols.copy()
    rarest_cols_temps = []
    rmse_vals = []

    data_900_df["rarest_cols"] = data_900_df[rarest_cols].sum(axis=1)

    DATA_SERIES_900 = {}
    DATA_SERIES_900["900_cumulus_denuded"] = prepare_data_serie(
        data_900_df, "cumulus_denuded", N_FOLDS
    )
    DATA_SERIES_900["900_day_0_mii"] = prepare_data_serie(
        data_900_df, "day_0_mii", N_FOLDS
    )

    GENE_ADD_COLS = [
        "test_amh_r",
        "ds1_pech_licz_10_pon",
        "patient_age",
        "prev_proc-cumulus_denuded",
        "prev_proc-day_0_mii",
        "cause_pco",
        "rarest_cols",
    ]
    GENE_RESULTS = train_data_series(
        LGB_PARAMS_BASE, DATA_SERIES_900, GENE_ADD_COLS, model_suffixes_filter=["l2"]
    )
    curr_rmse = list(
        GENE_RESULTS.train_infos["900_day_0_mii"].avg_errors_info.errors_info.values()
    )[0]
    rarest_cols_temps.append(rarest_cols)
    rmse_vals.append(curr_rmse)

    for column in rarest_cols:
        rarest_cols_temp.remove(column)
        data_900_df["rarest_cols"] = data_900_df[rarest_cols_temp].sum(axis=1)

        DATA_SERIES_900 = {}
        DATA_SERIES_900["900_cumulus_denuded"] = prepare_data_serie(
            data_900_df, "cumulus_denuded", N_FOLDS
        )
        DATA_SERIES_900["900_day_0_mii"] = prepare_data_serie(
            data_900_df, "day_0_mii", N_FOLDS
        )

        GENE_ADD_COLS = [
            "test_amh_r",
            "ds1_pech_licz_10_pon",
            "patient_age",
            "prev_proc-cumulus_denuded",
            "prev_proc-day_0_mii",
            "cause_pco",
            "rarest_cols",
        ]
        GENE_RESULTS = train_data_series(
            LGB_PARAMS_BASE,
            DATA_SERIES_900,
            GENE_ADD_COLS,
            model_suffixes_filter=["l2"],
        )
        curr_rmse = list(
            GENE_RESULTS.train_infos[
                "900_day_0_mii"
            ].avg_errors_info.errors_info.values()
        )[0]
        rarest_cols_temps.append(rarest_cols_temp)
        rmse_vals.append(curr_rmse)
        if curr_rmse == min(rmse_vals):
            rarest_cols_best.remove(column)
        rarest_cols_temp = rarest_cols_best.copy()
    return rarest_cols_best
