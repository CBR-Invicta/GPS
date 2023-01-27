from typing import Dict
from dataclasses import dataclass
from typing import Tuple, List, Dict
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from norms import add_norm_info
from literature_genes import get_literature_genes_dict
from wpisy_df import read_wpisy_df, get_body_df
from vcf import read_vcf_files_list, read_vcf_files_list, read_file_list
from math import log2
import os
import dotenv
from sqlalchemy import create_engine

import inspect


def test_column(test_name: str) -> str:
    if test_name in ["patient_age"]:
        return test_name
    return f"test_{test_name}_r"


def test_date_column(test_name: str) -> str:
    if test_name == "patient_age":
        return "procedure_start"
    return f"test_{test_name}_d"


def med_min_ds_column(med_name: str) -> str:
    return f"min_ds_{med_name}"


def med_max_ds_column(med_name: str) -> str:
    return f"max_ds_{med_name}"


def med_sum_dose_column(med_name: str) -> str:
    return f"sum_dose_{med_name}"


def med_days_count_column(med_name: str) -> str:
    return f"days_count_{med_name}"


def med_avg_dose_column(med_name: str) -> str:
    return f"avg_dose_{med_name}"


def gene_name_binary(gene_name: str) -> str:
    return f"{gene_name}_BIN"


def round_dose(x, prec=0, base=75):
    return round(base * round(float(x)/base), prec)


@dataclass
class TestsInfo:
    test_names: List[str]
    test_colors: Dict[str, Tuple[int]]
    test_scales: Dict[str, float]


@dataclass
class MedsInfo:
    med_names: List[str]
    med_colors: Dict[str, Tuple[int]]


@dataclass
class GenesInfo:
    gene_names: List[str]
    gene_names_binary: List[str]


@dataclass
class Data:
    pickups_df: pd.DataFrame
    meds_df: pd.DataFrame
    genes_df: pd.DataFrame
    poli_df: pd.DataFrame
    patient_genes_df: pd.DataFrame
    causes_df: pd.DataFrame
    input_df: pd.DataFrame
    tests_info: TestsInfo
    meds_info: MedsInfo
    genes_info: GenesInfo
    poli_change_names: List[str]


def read_pickups_df(directory: str) -> pd.DataFrame:

    pickups_df = pd.read_csv(
        f"{directory}/rpo_ivf_data_excluded_null_pickups_height_weight_2020-02-01.csv"
    )
    pickups_df = pickups_df.drop_duplicates()
    assert len(pickups_df) == len(pickups_df["process_number"].unique())

    # Sum _l + _r
    # If both _l and _r are NaN then result is NaN
    pickups_df["ds1_bubble_count"] = pickups_df["ds1_bubble_count_r"].fillna(
        0
    ) + pickups_df["ds1_bubble_count_l"].fillna(0)
    pickups_df.loc[
        (pickups_df["ds1_bubble_count_r"].isnull())
        & (pickups_df["ds1_bubble_count_l"].isnull()),
        "ds1_bubble_count",
    ] = np.NaN
    pickups_df["ds_bubble_count_prenatal"] = pickups_df[
        "ds_bubble_count_prenatal_r"
    ].fillna(0) + pickups_df["ds_bubble_count_prenatal_l"].fillna(0)
    pickups_df.loc[
        (pickups_df["ds_bubble_count_prenatal_r"].isnull())
        & (pickups_df["ds_bubble_count_prenatal_l"].isnull()),
        "ds_bubble_count_prenatal",
    ] = np.NaN

    # Rename columns before merge
    pickups_df.rename(
        columns={"process_type": "pickups_process_type"}, inplace=True)

    # Convert columns to datetime
    pickups_df = update_date_columns(
        pickups_df,
        ["procedure_start", "ds1_date", "ds_date", "pickup_date", "day_0_date"]
        + [column for column in pickups_df.columns if re.match(r"^test_.*_d$", column)],
        "%Y.%m.%d",
    )

    return pickups_df


def read_genes_df(directory: str) -> pd.DataFrame:
    genes_df = pd.read_csv(f"{directory}/gene_map.csv", sep="\t")

    return genes_df


def read_poli_df(directory: str) -> pd.DataFrame:
    poli_df = pd.read_csv(f"{directory}/2021-02-10_vcf_with_commercial.csv")

    return poli_df


def read_poli_df_call(directory: str) -> pd.DataFrame:

    result_allele_call = pd.read_csv(
        directory + "/result_allele_call.csv", sep=";")
    rpo_1_1_1 = pd.read_csv(directory + "/rpo_1_1_1.csv")
    mapping_archival_order_number_patientId = pd.read_csv(
        directory + "/mapping_archival_order_number_patientId.csv"
    )
    merged_result_allele_call = result_allele_call.merge(
        rpo_1_1_1, left_on="ORDER_NUMBER", right_on="project_order_number"
    )
    merged_result_allele_call = merged_result_allele_call.merge(
        mapping_archival_order_number_patientId,
        left_on="archival_order_number",
        right_on="number",
    )

    merged_result_allele_call.rename(
        columns={
            "POS": "position",
            "REF": "reference",
            "ALT": "alternative",
            "CHROME": "chromosome",
            "patientId": "patient_id",
        },
        inplace=True,
    )
    merged_result_allele_call.drop(
        columns=[
            "id",
            "lp",
            "number",
            "ASSEMBLY",
            "ORDER_NUMBER",
        ],
        inplace=True,
    )
    merged_result_allele_call.loc[:, "chromosome"] = merged_result_allele_call.loc[
        :, "chromosome"
    ].str.strip("chr")

    return merged_result_allele_call


def read_patient_groups_df(directory: str) -> pd.DataFrame:
    patient_groups_df = pd.read_csv(f"{directory}/patient_groups.csv", sep=";")

    groups_dict = {
        0: "0-some_group_no_reason",
        1: "1-non_standard_stim_response",
        2: "2-young_but_low_amh",
        3: "3-high_amh",
        4: "4-control_multiple_donors",
    }
    patient_groups_df.replace({"patient_group": groups_dict}, inplace=True)
    patient_groups_df["patient_group"] = patient_groups_df["patient_group"].astype(
        "category"
    )

    return patient_groups_df


def read_causes_df(directory: str) -> pd.DataFrame:
    causes_df = pd.read_csv(
        f"{directory}/infertility_causes_project.csv", sep=";")
    return causes_df


def read_process_list_df(directory: str) -> pd.DataFrame:

    process_list_df = pd.read_csv(
        f"{directory}/project_med_process_list_10-02-2021.csv", sep=";"
    )

    # Prepare meds_info
    meds_info = prepare_meds_info(process_list_df)

    # Update date columns
    date_columns = ["last_ds_date_ivf", "ds1_date"]
    process_list_df = update_date_columns(
        process_list_df, date_columns, "%d.%m.%Y")

    # Update dose columns
    dose_columns = (
        [med_avg_dose_column(med_name) for med_name in meds_info.med_names]
        + [med_sum_dose_column(med_name) for med_name in meds_info.med_names]
        + ["ds1_3_dawka", "ds1_7_dawka"]
    )
    for dose_column in dose_columns:
        if dose_column not in process_list_df.columns:
            continue
        if process_list_df[dose_column].dtype != np.number:
            process_list_df[dose_column] = (
                process_list_df[dose_column].str.replace(
                    ",", ".").astype(float)
            )

    # Convert columns to float
    for col in process_list_df.columns:

        need_convert_to_float = False
        if "qual_result_num" in col:
            need_convert_to_float = True

        if col == "ds_789_result_num_E2":
            need_convert_to_float = True

        if col in [
            "ds_1_result_num_AMH",
            "ds_1_result_num_E2",
            "ds_1_result_num_LH",
            "ds_1_result_num_PRG",
        ]:
            need_convert_to_float = True

        if need_convert_to_float:
            print(f"Converting {col} to float")
            process_list_df[col] = (
                process_list_df[col].astype(
                    str).str.replace(",", ".").astype(float)
            )

    # Rename columns for merge
    process_list_df.rename(
        columns={
            "ds1_date": "meds_ds1_date",
            "patient_age": "meds_patient_age",
            "pickup_date": "meds_pickup_date",
            "cumulus_count": "meds_cumulus_count",
            "cumulus_denuded": "meds_cumulus_denuded",
            "procedure_start": "meds_procedure_start",
            "process_type": "meds_process_type",
        },
        inplace=True,
    )

    process_list_df["ds4_7_dawka"] = (
        process_list_df["ds1_7_dawka"] - process_list_df["ds1_3_dawka"]
    )
    process_list_df["ds1_3_dawka_dzienna"] = (
        process_list_df["ds1_3_dawka"] / 3).apply(round_dose)
    process_list_df["ds1_7_dawka_dzienna"] = (
        process_list_df["ds1_7_dawka"] / 7).apply(round_dose)
    process_list_df["ds4_7_dawka_dzienna"] = (
        process_list_df["ds4_7_dawka"] / 4).apply(round_dose)

    return process_list_df, meds_info


def update_test_results_for_extreme_cases(df: pd.DataFrame) -> pd.DataFrame:

    df.loc[df["test_inhibina_b_r"] == "<7.2", "test_inhibina_b_r"] = 7.2
    df["test_inhibina_b_r"] = df["test_inhibina_b_r"].astype(float)

    df.loc[df["test_tsh_r"] == ">100", "test_tsh_r"] = 100.0
    df.loc[df["test_tsh_r"] == ".", "test_tsh_r"] = np.NaN
    df["test_tsh_r"] = df["test_tsh_r"].astype(float)

    df.loc[df["test_e2_r"] == "<5", "test_e2_r"] = 5.0
    df["test_e2_r"] = df["test_e2_r"].astype(float)

    df.loc[df["test_lh_r"] == "<0.100", "test_lh_r"] = 0.100
    df["test_lh_r"] = df["test_lh_r"].astype(float)

    df.loc[df["test_testosterone_r"] ==
           "< 0.087", "test_testosterone_r"] = 0.087
    df["test_testosterone_r"] = df["test_testosterone_r"].astype(float)

    df.loc[df["test_shbg_r"] == ">200", "test_shbg_r"] = 200.0
    df["test_shbg_r"] = df["test_shbg_r"].astype(float)

    return df


def prepare_tests_info(pickups_df: pd.DataFrame) -> TestsInfo:

    test_names = [
        re.sub(r"^test_(.*)_r$", "\\1", column)
        for column in pickups_df.columns
        if re.match(r"^test_.*_r$", column)
    ]
    test_names = ["patient_age"] + test_names
    test_colors = {
        test_name: plt.get_cmap("gist_rainbow")(color_number / len(test_names))
        for color_number, test_name in enumerate(test_names)
    }
    test_scales = {
        test_name: max(
            pickups_df[~pickups_df[test_column(test_name)].isnull()][
                test_column(test_name)
            ]
        )
        for test_name in test_names
    }

    return TestsInfo(
        test_names=test_names,
        test_colors=test_colors,
        test_scales=test_scales,
    )


def prepare_meds_info(meds_df: pd.DataFrame) -> MedsInfo:

    med_names = [
        column.replace("days_count_", "")
        for column in meds_df.columns
        if "days_count" in column
    ]
    med_colors = {
        med_name: plt.get_cmap("gist_rainbow")(color_number / len(med_names))
        for color_number, med_name in enumerate(med_names)
    }

    return MedsInfo(
        med_names=med_names,
        med_colors=med_colors,
    )


def prepare_genes_info(poli_df: pd.DataFrame) -> GenesInfo:

    gene_names = list(poli_df["gene"].unique())
    gene_names_binary = [gene_name_binary(
        gene_name) for gene_name in gene_names]

    return GenesInfo(
        gene_names=gene_names,
        gene_names_binary=gene_names_binary,
    )


def fill_gene_in_poli_df(poli_df: pd.DataFrame, genes_df: pd.DataFrame) -> pd.DataFrame:

    poli_df["gene"] = np.NaN
    for _index, row in genes_df.iterrows():

        poli_df.loc[
            (poli_df["chromosome"] == row["chromosome"])
            & (
                poli_df["position"].between(
                    row["from_gchr38_hg38"], row["to_gchr38_hg38"]
                )
            ),
            "gene",
        ] = f"{row['gene']}"
    poli_df.fillna({"gene": "OTHER"}, inplace=True)
    poli_df = poli_df[
        ["patient_id", "gene", "chromosome", "position", "alternative", "reference"]
    ]

    return poli_df


def fill_gene_in_poli_df_call(
    poli_df: pd.DataFrame, genes_df: pd.DataFrame
) -> pd.DataFrame:

    poli_df["gene"] = np.NaN
    for _index, row in genes_df.iterrows():

        poli_df.loc[
            (poli_df["chromosome"] == row["chromosome"])
            & (
                poli_df["position"].between(
                    row["from_gchr38_hg38"], row["to_gchr38_hg38"]
                )
            ),
            "gene",
        ] = f"{row['gene']}"
    poli_df.fillna({"gene": "OTHER"}, inplace=True)
    poli_df = poli_df[
        [
            "patient_id",
            "gene",
            "chromosome",
            "position",
            "alternative",
            "reference",
            "ALLELE_CALL",
        ]
    ]
    return poli_df


def generate_patient_genes_df(poli_df: pd.DataFrame) -> pd.DataFrame:

    patient_genes_dict = {
        patient_id: {} for patient_id in poli_df["patient_id"].unique()
    }
    for gene_name in poli_df["gene"].unique():

        for _index, row in poli_df[poli_df["gene"] == gene_name].iterrows():
            if gene_name not in patient_genes_dict[row["patient_id"]]:
                patient_genes_dict[row["patient_id"]][gene_name] = 0
            patient_genes_dict[row["patient_id"]][gene_name] += 1

    patient_genes_df = pd.DataFrame.from_dict(
        patient_genes_dict, orient="index")
    patient_genes_df.fillna(0.0, inplace=True)
    patient_genes_df.rename_axis("patient_id", inplace=True)
    patient_genes_df.reset_index(inplace=True)

    for gene_name in poli_df["gene"].unique():
        patient_genes_df[gene_name_binary(
            gene_name)] = patient_genes_df[gene_name] != 0
    return patient_genes_df


def prepare_med_one_hot_encoding(df: pd.DataFrame, meds_info: MedsInfo) -> pd.DataFrame:

    for med_name in meds_info.med_names:
        df[med_name] = ~df[med_sum_dose_column(med_name)].isnull()

    for med_name in meds_info.med_names:
        df[f"valid_{med_name}"] = ~df[f"valid_sum_dose_{med_name}"].isnull()

    return df


def prepare_poli_changes_df(poli_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

    poli_changes_dict = {
        patient_id: {} for patient_id in poli_df["patient_id"].unique()
    }

    for _index, row in poli_df.iterrows():

        feature_name = (
            f"{row['gene']}_"
            f"{row['chromosome']}_"
            f"{row['position']}_"
            f"{row['reference']}_"
            f"{row['alternative']}"
        )
        poli_changes_dict[row["patient_id"]][feature_name] = True

    poli_changes_df = pd.DataFrame.from_dict(poli_changes_dict, orient="index")
    poli_changes_df = poli_changes_df[sorted(poli_changes_df.columns)]
    poli_change_names = list(poli_changes_df.columns)
    poli_changes_df.fillna(False, inplace=True)
    poli_changes_df = poli_changes_df.reset_index()
    poli_changes_df.rename(columns={"index": "patient_id"}, inplace=True)

    return poli_changes_df, poli_change_names


def prepare_poli_changes_df_call(
    poli_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:

    poli_changes_dict = {
        patient_id: {} for patient_id in poli_df["patient_id"].unique()
    }

    for _index, row in poli_df.iterrows():

        feature_name = (
            f"{row['gene']}_"
            f"{row['chromosome']}_"
            f"{row['position']}_"
            f"{row['reference']}_"
            f"{row['alternative']}_"
            f"{row['ALLELE_CALL']}"
        )
        poli_changes_dict[row["patient_id"]][feature_name] = True

    poli_changes_df = pd.DataFrame.from_dict(poli_changes_dict, orient="index")
    poli_changes_df = poli_changes_df[sorted(poli_changes_df.columns)]
    poli_change_names = list(poli_changes_df.columns)
    poli_changes_df.fillna(False, inplace=True)
    poli_changes_df = poli_changes_df.reset_index()
    poli_changes_df.rename(columns={"index": "patient_id"}, inplace=True)

    return poli_changes_df, poli_change_names


def apply_literature_genes_dict(
    poli_changes_df: pd.DataFrame, poli_change_names: List[str]
) -> Tuple[pd.DataFrame, List[str]]:

    genes_dict = get_literature_genes_dict(poli_changes_df)
    for col, description in genes_dict.items():
        if col in poli_change_names:
            col_with_description = f"{col}___{description}"
            poli_change_names.remove(col)
            poli_change_names.append(col_with_description)
            poli_changes_df.rename(
                columns={col: col_with_description}, inplace=True)
    return poli_changes_df, poli_change_names


def update_date_columns(
    input_df: pd.DataFrame, column_names: List[str], date_format
) -> pd.DataFrame:

    for column_name in column_names:
        if column_name not in input_df.columns:
            continue
        input_df[column_name] = pd.to_datetime(
            input_df[column_name], format=date_format
        )

    return input_df


def prepare_categorical_columns(
    input_df: pd.DataFrame, column_names: List[str]
) -> pd.DataFrame:

    for column_name in column_names:
        if column_name not in input_df.columns:
            continue
        if column_name == "prot_type":
            col_map = {
                column_value: column_number
                for column_number, column_value in enumerate(
                    [
                        "prot_long",
                        "prot_short_agonist",
                        "prot_short_antagonista",
                        "prot_progesteron",
                        "prot_other",
                    ]
                )
            }
        else:
            col_map = {
                column_value: column_number
                for column_number, column_value in enumerate(
                    input_df[column_name].unique()
                )
            }
        input_df[f"{column_name}_cat"] = (
            input_df[column_name].map(col_map).astype("category")
        )
        input_df[f"{column_name}"] = input_df[f"{column_name}"].astype(
            "category")

    return input_df


def prepare_limited_test_columns(
    pickups_df: pd.DataFrame, tests_info: TestsInfo
) -> pd.DataFrame:

    for test_name in tests_info.test_names:

        pickups_df[f"test_{test_name}_months"] = (
            (pickups_df["procedure_start"] -
             pickups_df[test_date_column(test_name)])
            / np.timedelta64(1, "M")
        ).apply(np.floor)

        for months_range in [3, 6, 12, 24]:
            pickups_df.loc[
                pickups_df[f"test_{test_name}_months"] <= months_range,
                f"{test_column(test_name)}_{months_range}",
            ] = pickups_df[test_column(test_name)]

    return pickups_df


def calculate_meds_before_pickup(df: pd.DataFrame, meds_info: MedsInfo) -> pd.DataFrame:

    # NOTE:
    # Quite complicated calculations
    # "last_valid" == last day before pickup
    # we don't want to use meds after this day
    df["calc_last_valid_ds"] = (
        df["pickup_date"] - df["meds_ds1_date"]).dt.days - 1

    for med_name in meds_info.med_names:

        # Calculate potential last_valid_max_ds
        df[f"valid_max_ds_{med_name}"] = df[
            [med_max_ds_column(med_name), "calc_last_valid_ds"]
        ].min(axis=1)

        # Calculate fraction of days in valid period
        df[f"valid_days_fraction_{med_name}"] = (
            df[f"valid_max_ds_{med_name}"] -
            df[med_min_ds_column(med_name)] + 1
        ) / (df[med_max_ds_column(med_name)] - df[med_min_ds_column(med_name)] + 1)
        df.loc[
            df[f"valid_days_fraction_{med_name}"] < 0, f"valid_days_fraction_{med_name}"
        ] = 0.0

        # Calculate valid_sum_dose for valid period
        df[f"valid_sum_dose_{med_name}"] = np.NaN
        df.loc[
            df[f"valid_days_fraction_{med_name}"] > 0.0, f"valid_sum_dose_{med_name}"
        ] = (df[med_sum_dose_column(med_name)] * df[f"valid_days_fraction_{med_name}"])

        # Calculate valid_days_count for valid period
        df[f"valid_days_diff_{med_name}"] = np.NaN
        df.loc[
            df[f"valid_days_fraction_{med_name}"] > 0.0, f"valid_days_diff_{med_name}"
        ] = (df[f"valid_max_ds_{med_name}"] - df[med_min_ds_column(med_name)] + 1)

        # Calculate valid_avg_dose for valid period
        df[f"valid_avg_day_dose_{med_name}"] = np.NaN
        df.loc[
            df[f"valid_days_fraction_{med_name}"] > 0.0,
            f"valid_avg_day_dose_{med_name}",
        ] = (
            df[f"valid_sum_dose_{med_name}"] /
            df[f"valid_days_diff_{med_name}"]
        )

    return df


def add_prev_procedure_info(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(by=["__id__", "procedure_start"])

    df[f"prev_proc-__id__"] = df["__id__"].shift(1)
    for col in [
        "ds1_pech_licz_10_pon",
        "cumulus_denuded",
        "day_0_mii",
        "ds1_3_dawka_dzienna",
        "ds4_7_dawka_dzienna",
        "prot_type",
    ]:
        df[f"prev_proc-{col}"] = df[col].shift(1)
        df.loc[df["__id__"] != df["prev_proc-__id__"],
               f"prev_proc-{col}"] = np.NaN

    df.fillna({"prev_proc-prot_type": "NONE"}, inplace=True)
    df["prev_proc-prot_type"] = df["prev_proc-prot_type"].astype("category")

    df["prev_proc-denuded_per_bubbles"] = (
        df["prev_proc-cumulus_denuded"] / df["prev_proc-ds1_pech_licz_10_pon"]
    )
    df["prev_proc-mii_per_bubbles"] = (
        df["prev_proc-day_0_mii"] / df["prev_proc-ds1_pech_licz_10_pon"]
    )

    return df


def add_poseidon_group(
    df: pd.DataFrame,
    amh_col: str = "amh_qual_result_num",
    prev_proc_mii_col: str = "prev_proc-mii_cells_count",
) -> pd.DataFrame:

    df.loc[
        (df["patient_age"] < 35) & (df[amh_col] >= 1.2) & (
            df[prev_proc_mii_col] <= 3),
        "poseidon_group",
    ] = "1a_young35_highAMH1.2_prev0to3"
    df.loc[
        (df["patient_age"] < 35)
        & (df[amh_col] >= 1.2)
        & (df[prev_proc_mii_col] >= 4)
        & (df[prev_proc_mii_col] <= 9),
        "poseidon_group",
    ] = "1b_young35_highAMH1.2_prev4to9"
    df.loc[
        (df["patient_age"] >= 35) & (df[amh_col]
                                     >= 1.2) & (df[prev_proc_mii_col] <= 3),
        "poseidon_group",
    ] = "2a_old35_highAMH1.2_prev0to3"
    df.loc[
        (df["patient_age"] >= 35)
        & (df[amh_col] >= 1.2)
        & (df[prev_proc_mii_col] >= 4)
        & (df[prev_proc_mii_col] <= 9),
        "poseidon_group",
    ] = "2b_old35_highAMH1.2_prev4to9"
    df.loc[
        (df["patient_age"] < 35) & (df[amh_col] < 1.2), "poseidon_group"
    ] = "3_young35_lowAMH1.2"
    df.loc[
        (df["patient_age"] >= 35) & (df[amh_col] < 1.2), "poseidon_group"
    ] = "4_old35_lowAMH1.2"
    df.loc[
        (df["poseidon_group"].isnull()) & (
            df[amh_col].isnull()), "poseidon_group"
    ] = "0a_no_amh"
    df.loc[
        (df["poseidon_group"].isnull()) & (df[prev_proc_mii_col].isnull()),
        "poseidon_group",
    ] = "0b_first_time"
    df.loc[
        (df["poseidon_group"].isnull()) & (df[prev_proc_mii_col] >= 10),
        "poseidon_group",
    ] = "0c_prev10ormore"

    assert len(df[df["poseidon_group"].isnull()]) == 0

    return df


def add_mii_group(df: pd.DataFrame) -> pd.DataFrame:
    # df['mii_group'] = pd.qcut(df['day_0_mii'],5)
    # df['mii_group'] = pd.Series(df.loc[:,'mii_group'].astype(str), dtype = "category")
    df["mii_group"] = pd.cut(
        df["day_0_mii"],
        bins=[-1, 4, 10, 1000],
        labels=["1: 4>", "2: 5-10", "3: 10<"],
        ordered=False,
    )

    # df['mii_group'].fillna('(-0.001, 2.0]', inplace = True)
    return df


def add_pech_group(df: pd.DataFrame) -> pd.DataFrame:
    # df['mii_group'] = pd.qcut(df['day_0_mii'],5)
    # df['mii_group'] = pd.Series(df.loc[:,'mii_group'].astype(str), dtype = "category")
    df["pech_group"] = pd.cut(
        df["dslast_pech_licz"],
        bins=[-1, 6, 15, 1000],
        labels=["1: 6>", "2: 7-15", "3: 15<"],
        ordered=False,
    )

    # df['mii_group'].fillna('(-0.001, 2.0]', inplace = True)
    return df


def add_cumulus_denuded_group(df: pd.DataFrame) -> pd.DataFrame:
    df["cumulus_denuded_group"] = pd.qcut(df["cumulus_denuded"], 5)
    df["cumulus_denuded_group"] = pd.Series(
        df.loc[:, "cumulus_denuded_group"].astype(str), dtype="category"
    )
    # df['mii_group'].fillna('(-0.001, 2.0]', inplace = True)
    return df


def add_hiper(df: pd.DataFrame) -> pd.DataFrame:

    df["hiper_20"] = df["cumulus_count"] >= 20
    df["hiper_25"] = df["cumulus_count"] >= 25
    df["hiper_20"] = df["hiper_20"].astype(int)
    df["hiper_25"] = df["hiper_25"].astype(int)

    return df


def add_hiper_pech(df: pd.DataFrame) -> pd.DataFrame:

    df["hiper_20_pech"] = df["dslast_pech_licz"] >= 20
    df["hiper_25_pech"] = df["dslast_pech_licz"] >= 25
    df["hiper_20_pech"] = df["hiper_20_pech"].astype(int)
    df["hiper_25_pech"] = df["hiper_25_pech"].astype(int)

    return df


def remove_outliers_mii(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(df[df["day_0_mii"] > 20].index, inplace=True)
    return df


def remove_outliers_IF(df: pd.DataFrame, cols: List) -> pd.DataFrame:
    model_data = df.loc[:, cols].dropna(axis=1).copy()
    iforest_model = IsolationForest(
        random_state=2, n_estimators=100, max_samples=256)
    iforest_model.fit(model_data)
    score_iforest_model = -1 * iforest_model.score_samples(model_data)
    df.drop(model_data[score_iforest_model > 0.6].index, inplace=True)
    return df


def read_data(
    directory: str,
    merge_with_meds: bool = False,
    merge_with_patient_genes: bool = False,
    call: bool = False,
    vcf=False,
    remove_outliers=False,
) -> Data:

    pickups_df = read_pickups_df(directory)
    meds_df, meds_info = read_process_list_df(directory)
    genes_df = read_genes_df(directory)
    if vcf:
        poli_df = read_vcf_files_list(directory, directory + "/vcf/")
    if call:
        poli_df = read_poli_df_call(directory)
    else:
        poli_df = read_poli_df(directory)
    causes_df = read_causes_df(directory)
    patient_groups_df = read_patient_groups_df(directory)
    wpisy_df = read_wpisy_df(directory)
    body_df = get_body_df(wpisy_df)

    pickups_df = update_test_results_for_extreme_cases(pickups_df)
    if call:
        poli_df = fill_gene_in_poli_df_call(poli_df, genes_df)
        patient_genes_df = generate_patient_genes_df(poli_df)
        poli_changes_df, poli_change_names = prepare_poli_changes_df_call(
            poli_df)
        poli_changes_df, poli_change_names = apply_literature_genes_dict(
            poli_changes_df, poli_change_names
        )
    else:
        poli_df = fill_gene_in_poli_df(poli_df, genes_df)
        patient_genes_df = generate_patient_genes_df(poli_df)
        poli_changes_df, poli_change_names = prepare_poli_changes_df(poli_df)
        poli_changes_df, poli_change_names = apply_literature_genes_dict(
            poli_changes_df, poli_change_names
        )

    genes_info = prepare_genes_info(poli_df)
    tests_info = prepare_tests_info(pickups_df)

    pickups_df = prepare_limited_test_columns(pickups_df, tests_info)

    # Merge dataframes
    input_df = pickups_df
    print(f"pickups_df:                  {len(input_df)}")
    if merge_with_meds:
        input_df = pickups_df.merge(meds_df, on="process_number", how="inner")
        print(f"merged with meds_df:         {len(input_df)}")
    if merge_with_patient_genes:
        input_df = input_df.merge(
            patient_genes_df, left_on="__id__", right_on="patient_id", how="inner"
        )
        print(f"merged with patient_gens_df: {len(input_df)}")
    if merge_with_patient_genes:
        input_df = input_df.merge(
            poli_changes_df, left_on="__id__", right_on="patient_id", how="inner"
        )
        print(f"merged with poli_changes_df: {len(input_df)}")

    input_patient_ids = list(input_df["__id__"].unique())
    causes_df = causes_df[causes_df["patient_id"].isin(input_patient_ids)]
    input_df = input_df.merge(
        causes_df, left_on="__id__", right_on="patient_id", how="outer"
    )
    for col in input_df.columns:
        if "cause" in col:
            input_df.fillna({col: False}, inplace=True)
            # input_df[col] = input_df[col].astype(float)
            input_df[col] = input_df[col].astype("category")
    print(f"merged with causes_df: {len(input_df)}")
    input_df = input_df.merge(
        patient_groups_df, left_on="__id__", right_on="patient_id", how="left"
    )
    print(f"merged with patient_groups_df: {len(input_df)}")
    input_df = input_df.merge(
        body_df, left_on="__id__", right_on="person_id", how="left"
    )
    print(f"merged with body_df: {len(input_df)}")

    if merge_with_meds:
        input_df = calculate_meds_before_pickup(input_df, meds_info)
        input_df = prepare_med_one_hot_encoding(input_df, meds_info)
        input_df = add_prev_procedure_info(input_df)
        input_df = add_poseidon_group(
            input_df, amh_col="test_amh_r", prev_proc_mii_col="prev_proc-day_0_mii"
        )
        input_df = add_mii_group(input_df)
        input_df = add_pech_group(input_df)
        input_df = add_cumulus_denuded_group(input_df)
        input_df = add_norm_info(input_df)

        input_df["amh_including_ds1"] = input_df["test_amh_r"]
        input_df.loc[
            input_df["ds_1_result_num_AMH"].notnull(), "amh_including_ds1"
        ] = input_df["ds_1_result_num_AMH"]

        input_df["days_count_Antykoncepcja_limited"] = np.NaN
        input_df.loc[
            (input_df["days_count_Antykoncepcja"].notnull())
            & (input_df["days_count_Antykoncepcja"] <= 28),
            "days_count_Antykoncepcja_limited",
        ] = "antykoncepcja <= 28 days"
        input_df.loc[
            (input_df["days_count_Antykoncepcja"].notnull())
            & (input_df["days_count_Antykoncepcja"] > 28),
            "days_count_Antykoncepcja_limited",
        ] = "antykoncepcja > 28 days"
        prepare_categorical_columns(
            input_df, ["days_count_Antykoncepcja_limited"])

        input_df["ds1_dc_limited_by_4"] = np.NaN
        input_df.loc[
            (input_df["ds1_dc"].notnull()) & (input_df["ds1_dc"] <= 4),
            "ds1_dc_limited_by_4",
        ] = "dzien cyklu w ds1 <= 4"
        input_df.loc[
            (input_df["ds1_dc"].notnull()) & (input_df["ds1_dc"] > 4),
            "ds1_dc_limited_by_4",
        ] = "dzien cyklu w ds1 > 4"
        prepare_categorical_columns(input_df, ["ds1_dc_limited_by_4"])

        input_df["ds1_dc_limited_by_8"] = np.NaN
        input_df.loc[
            (input_df["ds1_dc"].notnull()) & (input_df["ds1_dc"] <= 8),
            "ds1_dc_limited_by_8",
        ] = "dzien cyklu w ds1 <= 8"
        input_df.loc[
            (input_df["ds1_dc"].notnull()) & (input_df["ds1_dc"] > 8),
            "ds1_dc_limited_by_8",
        ] = "dzien cyklu w ds1 > 8"
        prepare_categorical_columns(input_df, ["ds1_dc_limited_by_8"])

    input_df = add_hiper(input_df)
    input_df = add_hiper_pech(input_df)
    input_df = prepare_categorical_columns(
        input_df, ["prot_type", "poseidon_group"])
    input_df['ds1_7_dawka_dzienna_str'] = input_df.ds1_3_dawka_dzienna.astype(
        int).astype(str)+','+input_df.ds4_7_dawka_dzienna.astype(int).astype(str)
    input_df.rename(
        columns={
            "pickups_process_type": "process_type",
        },
        inplace=True,
    )
    if not call:
        input_df.drop(columns=["OTHER_X_140505223_AAAAAAA_-"], inplace=True)
        poli_change_names.remove("OTHER_X_140505223_AAAAAAA_-")

        input_df["one_in_every_row"] = 1

        # BASED ON INITIAL GENE RESULTS:
        input_df["gene_variants_CA"] = input_df.loc[
            :,
            [  # 'FSHR_2_49154446_C_T___5 Prime UTR Variant',
                "ESR1_6_152098960_G_A",
                "ESR1_6_152061176_G_T",
                "GDF9_5_132865538_T_C",
                "GDF9_5_132866205_T_C",
                # 'LHCGR_2_48687476_C_G','ESR2_14_64227477_C_T',
                #'AR_X_67723521_-_CACACAC', 'FSHB_11_30234435_A_G'
            ],
        ].sum(axis=1)
        input_df["gene_variants_SOM"] = input_df.loc[
            :,
            [
                "GDF9_5_132865538_T_C",
                "GDF9_5_132866205_T_C",
                "PRL_6_22292747_A_T",
                "AR_X_67723521_-_CACACAC",
                "LHCGR_2_48729336_C_T",
                "FSHB_11_30234435_A_G",
                "FSHR_2_48963902_C_T___Missense Variant",
                "ESR1_6_152098960_G_A",
                "ESR1_6_152061190_A_G",
                "PRLR_5_35069955_T_A",
                "ESR2_14_64227477_C_T",
            ],
        ].sum(axis=1)

        input_df["gene_variants_MIM"] = input_df.loc[
            :, ["SOX9_17_72125967_G_A", "ESR2_14_64227364_T_C"]
        ].sum(axis=1)

    if remove_outliers:
        input_df = remove_outliers_mii(input_df)
        input_df = remove_outliers_IF(
            input_df,
            [
                "test_amh_r",
                "patient_age",
                "cause_pco",
                "ds1_pech_licz_10_pon",
                "prev_proc-cumulus_denuded",
                "prev_proc-day_0_mii",
                "day_0_mii",
            ],
        )

    # input_df['log_test_amh_r'] = input_df['test_amh_r'].apply(log2)
    # input_df['class_test_amh_r'] = pd.cut(input_df['test_amh_r'], bins = [0,0.5,2,5,1000] ,
    # labels=[">0.5", "0.5-2", "2-5","5<"], ordered=False)

    return Data(
        pickups_df=pickups_df,
        meds_df=meds_df,
        genes_df=genes_df,
        poli_df=poli_df,
        patient_genes_df=patient_genes_df,
        causes_df=causes_df,
        input_df=input_df,
        meds_info=meds_info,
        tests_info=tests_info,
        genes_info=genes_info,
        poli_change_names=poli_change_names,
    )


dotenv.load_dotenv()
sql_path = os.getenv("sql_path")


def connect_engine(
    user: str = "USER", password: str = "PASSWORD", domain: str = "DOMAIN_replica"
):
    """connect_engine allows user to connect to selected database, based on env varaibles

    Parameters
    ----------
    user : str, optional
        name of the user in ENV, by default 'USER'
    password : str, optional
        password of the user in ENV, by default "PASSWORD"
    domain : str, optional
        database you want to connect to, by default "DOMAIN_replica"

    Returns
    -------
    conn
        sql alchemy connection
    """
    try:
        user = os.getenv(user)
        password = os.getenv(password)
        domain = os.getenv(domain)
    except:
        raise ValueError(
            f"W zmiennych środowiskowych brakuje wartości {user}/{password}/{domain}."
        )

    path = "postgresql+psycopg2://" + user + ":" + password + "@" + domain
    engine = create_engine(path, execution_options={"stream_results": True})
    conn = engine.connect()
    return conn


def get_query(name: str, **kwargs):
    """Wczytywanie danych z bazy na podstawie pliku .sql

    Parameters:
    ----------
    name : str
        Nazwa query w folderze sql, która ma posłużyć do wczytania danych z repliki

    Returns:
    -------
    data : pd.DataFrame
        Dataframe z danymi z bazy
    """
    user = kwargs.pop(
        "user",
        inspect.signature(connect_engine).parameters.get("user").default,
    )
    password = kwargs.pop(
        "password",
        inspect.signature(connect_engine).parameters.get("password").default,
    )
    domain = kwargs.pop(
        "domain",
        inspect.signature(connect_engine).parameters.get("domain").default,
    )
    with open(sql_path + "/" + name + ".sql", "r") as query:
        q = query.read()
    conn = connect_engine(user=user, password=password, domain=domain)
    data = pd.read_sql_query(q, conn)
    return data


threshold = {
    64: {"max": 7},
    92: {"min": 130, "max": 210},
    90: {"min": 40, "max": 150},
    112: {"min": 9, "max": 18},
    209: {"min": 18, "max": 100},
    210: {"min": 18, "max": 100},
    211: {"min": 18, "max": 100},
    479: {"min": 0, "max": 1},
    480: {"min": 0, "max": 5},
    482: {"min": 0, "max": 25},
}


def process_question_92(data: pd.DataFrame, question_id: int = 92) -> pd.DataFrame:
    """Przygotowanie pytania: wzrost

    Parameters
    ----------
    data : pd.DataFrame
        Zbiór danych wejściowych
    question_id : int, optional
        id pytania, by default 92
    Returns
    -------
    pd.DataFrame
        Tabela z wartościami numerycznymi jako wzrost pacjentek w cm
    """
    response = data.loc[data.id_question == question_id].odpowiedzi.apply(
        lambda x: re.sub(r"\d(.)\d\d", "", x)
    )

    nazwa_zmiennej = data.loc[data.id_question ==
                              question_id].pytanie_pl.iloc[0]
    response = pd.to_numeric(
        data.loc[data.id_question ==
                 question_id].odpowiedzi.str.replace(",", "."),
        errors="coerce",
    ).rename("92_" + nazwa_zmiennej)
    if question_id in threshold.keys():
        response.loc[
            (response > threshold[question_id]["max"])
            | (response < threshold[question_id]["min"])
        ] = np.nan

    response = pd.DataFrame(response)
    response.set_index(
        data.loc[data.id_question == question_id].wizyta_id, inplace=True
    )

    return response


def process_numeric_questions(
    data: pd.DataFrame,
    question_id: int,
    threshold: Dict[float, Dict[str, float]] = threshold,
) -> pd.Series:
    """process_numeric_question zamienia zmienną liczbową na float

    Parameters
    ----------
    data : pd.DataFrame
        Zbiór danych wejściowych
    question_id : int
        ID pytania
    threshold : Dict[float, Dict[str, float]], optional
        Próg dolny i górny zmiennej, by default None

    Returns
    -------
    pd.Series
        Zmienna w formacie float
    """

    nazwa_zmiennej = (
        str(question_id)
        + "_"
        + data.loc[data.id_question ==
                   question_id].pytanie_pl.iloc[0].replace(" ", "_")
    )
    response = pd.to_numeric(
        data.loc[data.id_question ==
                 question_id].odpowiedzi.str.replace(",", "."),
        errors="coerce",
    ).rename(nazwa_zmiennej)
    if question_id in threshold.keys():
        response.loc[
            (response > threshold[question_id]["max"])
            | (response < threshold[question_id]["min"])
        ] = np.nan
    response = pd.DataFrame(response)
    response.set_index(
        data.loc[data.id_question == question_id].wizyta_id, inplace=True
    )
    return response


def process_calculate_bmi(
    data: pd.DataFrame, question_id_waga: int = 90, question_id_wzrost: int = 92
) -> pd.DataFrame:
    """process_calculate_bmi oblicza BMi na podstawie kolumn waga oraz wzrot (transformuje wzrost na m oraz usuwa z wejściowej tabeli kolumny waga oraz wzrost)

    Parameters
    ----------
    data : pd.DataFrame
        Zbiór danych wejściowych
    question_id_waga : int, optional
        id_question pytania o wagę pacjentki, by default 90
    question_id_wzrost : int, optional
        id_question pytania o wzrost pacjentki, by default 92
    """
    bmi = (data["90_Waga"] / (data["92_Wzrost"] / 100) ** 2).round(1).rename(
        str(question_id_waga) + "_" + str(question_id_wzrost) + "_" + "BMI"
    )
    data = data.merge(bmi, how="left", left_index=True, right_index=True)
    data.drop(["90_Waga", "92_Wzrost"], axis=1, inplace=True)

    return data


def prepare_zalecenie_redukcji_masy_ciala_dataset():
    redukcja_masy_ciala = get_query(
        "redukcja_masy_ciala",
        user="USER_staging",
        password="PASSWORD_staging",
        domain="DOMAIN_staging",
    )
    data = process_question_92(redukcja_masy_ciala).merge(
        process_numeric_questions(redukcja_masy_ciala, question_id=90),
        left_index=True,
        right_index=True,
        how="outer",
    )
    data = data.loc[~data.index.duplicated(keep="first")]
    response = process_calculate_bmi(data)
    return response.reset_index().merge(
        redukcja_masy_ciala[["wizyta_id", "patient_id", "result_time"]]
        .sort_values(by="result_time", ascending=False)
        .drop_duplicates(subset=["wizyta_id", "patient_id"]),
        left_on="wizyta_id",
        right_on="wizyta_id",
        how="left",
    )
