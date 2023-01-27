import pandas as pd
import numpy as np
from lib.read_data import (
    add_poseidon_group,
    update_date_columns,
    prepare_categorical_columns,
    add_hiper,
    add_mii_group,
    add_cumulus_denuded_group,
    add_pech_group,
    add_hiper_pech,
    round_dose
)
from lib.norms import add_norm_fsh_in_cycle, add_norm_info_2015


def read_data_2015(data_dir: str) -> pd.DataFrame:

    data_2015_df = pd.read_csv(
        f"{data_dir}/process_ivf_stim_2015.csv", sep=";")
    data_2015_df.rename(columns={"patient_id": "__id__"}, inplace=True)
    print(
        f"Data len - before drop_duplicates(process_number): {len(data_2015_df)}")
    data_2015_df = data_2015_df.drop_duplicates(subset="process_number")
    print(
        f"Data len - after drop_duplicates(process_number): {len(data_2015_df)}")

    causes_df = pd.read_csv(f"{data_dir}/infertility_causes_2015.csv", sep=";")

    print(f"Data len: {len(data_2015_df)}")
    data_2015_df = data_2015_df.merge(
        causes_df, left_on="__id__", right_on="patient_id", how="outer"
    )
    for col in data_2015_df.columns:
        if "cause" in col:
            data_2015_df.fillna({col: False}, inplace=True)
            # data_2015_df[col] = data_2015_df[col].astype(float)
            data_2015_df[col] = data_2015_df[col].astype("category")
    print(f"Data len: {len(data_2015_df)} after merge with causes")

    # Convert columns to float
    for col in data_2015_df.columns:

        need_convert_to_float = False
        if "qual_result_num" in col:
            need_convert_to_float = True
        if col in [
            "ds1_3_dawka",
            "ds1_7_dawka",
            "ds_1_result_num_AMH",
            "ds_1_result_num_E2",
            "ds_1_result_num_LH",
            "ds_1_result_num_PRG",
        ]:
            need_convert_to_float = True

        if need_convert_to_float:
            print(f"Converting {col} to float")
            data_2015_df[col] = data_2015_df[col].str.replace(
                ",", ".").astype(float)

    # Calculate 'ds1_3_dawka_dzienna', 'ds4_7_dawka_dzienna'
    data_2015_df["ds4_7_dawka"] = (
        data_2015_df["ds1_7_dawka"] - data_2015_df["ds1_3_dawka"]
    )
    data_2015_df["ds1_3_dawka_dzienna"] = (
        data_2015_df["ds1_3_dawka"] / 3).apply(round_dose)
    data_2015_df["ds1_7_dawka_dzienna"] = (
        data_2015_df["ds1_7_dawka"] / 7).apply(round_dose)
    data_2015_df["ds4_7_dawka_dzienna"] = (
        data_2015_df["ds4_7_dawka"] / 4).apply(round_dose)

    # Calculate 'prev_proc-' columns
    data_2015_df = data_2015_df.sort_values(by=["__id__", "procedure_start"])
    data_2015_df[f"prev_proc-__id__"] = data_2015_df["__id__"].shift(1)
    for col in [
        "ds1_pech_licz_10_pon",
        "cumulus_denuded",
        "mii_cells_count",
        "ds1_3_dawka_dzienna",
        "ds4_7_dawka_dzienna",
        "prot_type",
    ]:

        data_2015_df[f"prev_proc-{col}"] = data_2015_df[col].shift(1)
        data_2015_df.loc[
            data_2015_df["__id__"] != data_2015_df["prev_proc-__id__"],
            f"prev_proc-{col}",
        ] = np.NaN

    data_2015_df.fillna({"prev_proc-prot_type": "NONE"}, inplace=True)
    data_2015_df["prev_proc-prot_type"] = data_2015_df["prev_proc-prot_type"].astype(
        "category"
    )

    data_2015_df["prev_proc-denuded_per_bubbles"] = (
        data_2015_df["prev_proc-cumulus_denuded"]
        / data_2015_df["prev_proc-ds1_pech_licz_10_pon"]
    )
    data_2015_df["prev_proc-mii_per_bubbles"] = (
        data_2015_df["prev_proc-mii_cells_count"]
        / data_2015_df["prev_proc-ds1_pech_licz_10_pon"]
    )

    for dose_column in data_2015_df.columns:
        if "sum_dose_" not in dose_column and "avg_dose" not in dose_column:
            continue
        if data_2015_df[dose_column].dtype != np.number:
            data_2015_df[dose_column] = (
                data_2015_df[dose_column]
                .astype(str)
                .str.replace(",", ".")
                .astype(float)
            )

    data_2015_df = add_norm_info_2015(data_2015_df)
    data_2015_df = add_norm_fsh_in_cycle(data_2015_df)
    data_2015_df = update_date_columns(
        data_2015_df, ["process_start"], "%Y-%m-%d")
    data_2015_df = add_poseidon_group(data_2015_df)

    data_2015_df = prepare_categorical_columns(
        data_2015_df, ["prot_type", "poseidon_group"]
    )

    data_2015_df.rename(
        columns={
            "amh_qual_result_num": "test_amh_r",
            "mii_cells_count": "day_0_mii",
            "prev_proc-mii_cells_count": "prev_proc-day_0_mii",
        },
        inplace=True,
    )

    data_2015_df = add_hiper(data_2015_df)
    data_2015_df = add_hiper_pech(data_2015_df)

    data_2015_df["amh_including_ds1"] = data_2015_df["test_amh_r"]
    data_2015_df.loc[
        data_2015_df["ds_1_result_num_AMH"].notnull(), "amh_including_ds1"
    ] = data_2015_df["ds_1_result_num_AMH"]

    data_2015_df["days_count_Antykoncepcja_round_10"] = data_2015_df[
        "days_count_Antykoncepcja"
    ].round(-1)

    data_2015_df["one_in_every_row"] = 1

    data_2015_df["days_count_Antykoncepcja_limited"] = np.NaN
    data_2015_df.loc[
        (data_2015_df["days_count_Antykoncepcja"].notnull())
        & (data_2015_df["days_count_Antykoncepcja"] <= 28),
        "days_count_Antykoncepcja_limited",
    ] = "antykoncepcja <= 28 days"
    data_2015_df.loc[
        (data_2015_df["days_count_Antykoncepcja"].notnull())
        & (data_2015_df["days_count_Antykoncepcja"] > 28),
        "days_count_Antykoncepcja_limited",
    ] = "antykoncepcja > 28 days"
    prepare_categorical_columns(
        data_2015_df, ["days_count_Antykoncepcja_limited"])

    data_2015_df["ds1_dc_limited_by_4"] = np.NaN
    data_2015_df.loc[
        (data_2015_df["ds1_dc"].notnull()) & (data_2015_df["ds1_dc"] <= 4),
        "ds1_dc_limited_by_4",
    ] = "dzien cyklu w ds1 <= 4"
    data_2015_df.loc[
        (data_2015_df["ds1_dc"].notnull()) & (data_2015_df["ds1_dc"] > 4),
        "ds1_dc_limited_by_4",
    ] = "dzien cyklu w ds1 > 4"
    prepare_categorical_columns(data_2015_df, ["ds1_dc_limited_by_4"])

    data_2015_df["ds1_dc_limited_by_8"] = np.NaN
    data_2015_df.loc[
        (data_2015_df["ds1_dc"].notnull()) & (data_2015_df["ds1_dc"] <= 8),
        "ds1_dc_limited_by_8",
    ] = "dzien cyklu w ds1 <= 8"
    data_2015_df.loc[
        (data_2015_df["ds1_dc"].notnull()) & (data_2015_df["ds1_dc"] > 8),
        "ds1_dc_limited_by_8",
    ] = "dzien cyklu w ds1 > 8"
    prepare_categorical_columns(data_2015_df, ["ds1_dc_limited_by_8"])

    data_2015_df = add_mii_group(data_2015_df)
    data_2015_df = add_cumulus_denuded_group(data_2015_df)
    data_2015_df = add_pech_group(data_2015_df)
    data_2015_df['ds1_7_dawka_dzienna_str'] = data_2015_df.ds1_3_dawka_dzienna.astype(
        int).astype(str)+','+data_2015_df.ds4_7_dawka_dzienna.astype(int).astype(str)

    return data_2015_df
