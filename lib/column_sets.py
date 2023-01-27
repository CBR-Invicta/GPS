from typing import List
from dataclasses import dataclass
import re
import pandas as pd


from lib.read_data import Data


@dataclass
class ColumnSets:
    DS1_PECH_COLS: List[str]
    MODULATED_COLS: List[str]
    DS1_RESULT_COLS: List[str]

    VARIOUS_COLS_900: List[str]
    BODY_COLS_900: List[str]
    HORMONE_COLS_900: List[str]
    HORMONE_NORMS_COLS_900: List[str]
    HORMONE_NORMS_VALID_IN_TIME_COLS_900: List[str]
    VALID_SUM_DOSES_COLS_900: List[str]
    VALID_SUM_DOSES_AND_DAYS_DIFF_COLS_900: List[List[str]]
    CAUSE_COLS_900: List[str]

    VARIOUS_COLS_2015: List[str]
    HORMONE_COLS_2015: List[str]
    HORMONE_NORMS_COLS_2015: List[str]
    HORMONE_NORMS_VALID_IN_TIME_COLS_2015: List[str]
    FSH_NORM_CYCLED_COLS_2015: List[str]
    CAUSE_COLS_2015: List[str]


def prepare_column_sets(data_900: Data, data_2015_df: pd.DataFrame) -> ColumnSets:

    data_900_df = data_900.input_df

    DS1_PECH_COLS = [
        "ds1_pech_licz_14_pow",
        "ds1_pech_licz_18_pow",
        "ds1_pech_licz_11_pow",
        "ds1_pech_licz_3_8",
        "ds1_pech_licz_2_8",
        "ds1_pech_licz_16_22",
    ]

    MODULATED_COLS = [
        "ds1_3_dawka_dzienna",
        "ds4_7_dawka_dzienna",
        "prot_type",
    ]

    DS1_RESULT_COLS = [
        "ds_1_result_num_AMH",
        "ds_1_result_num_E2",
        "ds_1_result_num_LH",
        "ds_1_result_num_PRG",
    ]

    VARIOUS_COLS_900 = [
        "pickup_no_4m_prev",
        "poseidon_group",
        "patient_group",
        "prev_proc-denuded_per_bubbles",
        "prev_proc-mii_per_bubbles",
        "prev_proc-ds1_3_dawka_dzienna",
        "prev_proc-ds4_7_dawka_dzienna",
    ]

    BODY_COLS_900 = [
        "weight",
        "height",
        "bmi",
    ]

    HORMONE_COLS_900 = []
    for col in data_900_df.columns:
        if not re.match(r"test_.*_r$", col):
            continue
        if col == "test_amh_r":
            continue
        HORMONE_COLS_900.append(col)

    HORMONE_NORMS_COLS_900 = []
    for col in data_900_df.columns:
        if not re.match(r"norm_.*$", col):
            continue
        HORMONE_NORMS_COLS_900.append(col)

    HORMONE_NORMS_VALID_IN_TIME_COLS_900 = []
    for col in data_900_df.columns:
        if not re.match(r"valid_norm_.*$", col):
            continue
        HORMONE_NORMS_VALID_IN_TIME_COLS_900.append(col)

    VALID_SUM_DOSES_COLS_900 = []
    for col in data_900_df.columns:
        if not re.match(r"valid_sum_dose.*$", col):
            continue
        if col == "valid_sum_dose_Gonadotropiny":
            continue
        VALID_SUM_DOSES_COLS_900.append(col)

    VALID_SUM_DOSES_AND_DAYS_DIFF_COLS_900 = []
    for col in data_900_df.columns:
        if not re.match(r"valid_sum_dose.*$", col):
            continue
        days_col = re.sub(r"sum_dose", "days_diff", col)
        VALID_SUM_DOSES_AND_DAYS_DIFF_COLS_900.append([col, days_col])

    CAUSE_COLS_900 = []
    for col in data_900_df.columns:
        if not re.match(r"cause_.*$", col):
            continue
        CAUSE_COLS_900.append(col)
    CAUSE_COLS_900.remove("cause_male_factor")
    CAUSE_COLS_900.remove("cause_genetic_male")

    VARIOUS_COLS_2015 = VARIOUS_COLS_900
    VARIOUS_COLS_2015.remove("patient_group")

    HORMONE_COLS_2015 = []
    for col in data_2015_df.columns:
        if not re.match(r"^qual_result_num_.*$", col):
            continue
        HORMONE_COLS_2015.append(col)

    HORMONE_NORMS_COLS_2015 = []
    for col in data_2015_df.columns:
        if not re.match(r"norm_.*$", col):
            continue
        HORMONE_NORMS_COLS_2015.append(col)

    HORMONE_NORMS_VALID_IN_TIME_COLS_2015 = []
    for col in data_2015_df.columns:
        if not re.match(r"valid_norm_.*$", col):
            continue
        HORMONE_NORMS_VALID_IN_TIME_COLS_2015.append(col)

    FSH_NORM_CYCLED_COLS_2015 = ["norm_cycled_FSH"]

    CAUSE_COLS_2015 = []
    for col in data_2015_df.columns:
        if not re.match(r"cause_.*$", col):
            continue
        CAUSE_COLS_2015.append(col)
    CAUSE_COLS_2015.remove("cause_male_factor")
    CAUSE_COLS_2015.remove("cause_genetic_male")

    return ColumnSets(
        DS1_PECH_COLS=DS1_PECH_COLS,
        MODULATED_COLS=MODULATED_COLS,
        DS1_RESULT_COLS=DS1_RESULT_COLS,
        VARIOUS_COLS_900=VARIOUS_COLS_900,
        BODY_COLS_900=BODY_COLS_900,
        HORMONE_COLS_900=HORMONE_COLS_900,
        HORMONE_NORMS_COLS_900=HORMONE_NORMS_COLS_900,
        HORMONE_NORMS_VALID_IN_TIME_COLS_900=HORMONE_NORMS_VALID_IN_TIME_COLS_900,
        VALID_SUM_DOSES_COLS_900=VALID_SUM_DOSES_COLS_900,
        VALID_SUM_DOSES_AND_DAYS_DIFF_COLS_900=VALID_SUM_DOSES_AND_DAYS_DIFF_COLS_900,
        CAUSE_COLS_900=CAUSE_COLS_900,
        VARIOUS_COLS_2015=VARIOUS_COLS_2015,
        HORMONE_COLS_2015=HORMONE_COLS_2015,
        HORMONE_NORMS_COLS_2015=HORMONE_NORMS_COLS_2015,
        HORMONE_NORMS_VALID_IN_TIME_COLS_2015=HORMONE_NORMS_VALID_IN_TIME_COLS_2015,
        FSH_NORM_CYCLED_COLS_2015=FSH_NORM_CYCLED_COLS_2015,
        CAUSE_COLS_2015=CAUSE_COLS_2015,
    )
