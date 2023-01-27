import pandas as pd
import numpy as np
from lib.filter_data import filter_data


def add_genetic(data_900_df):
    data_900_df["chr2_4GAM_block4_haplotypes_variants"] = (
        (data_900_df["genome_012_chr2_48962782_C_T"] > 0)
        & (data_900_df["genome_012_chr2_48962060_A_G"] == 0)
    ) * 1
    data_900_df["chr5_4GAM_block2_haplotypes_variants"] = (
        data_900_df[
            [
                "genome_012_chr5_35063190_A_T",
                "genome_012_chr5_35064922_C_G",
                "genome_012_chr5_35068146_G_C",
                "genome_012_chr5_35061629_T_C",
                "genome_012_chr5_35069864_G_A",
                "genome_012_chr5_35064413_C_A",
                "genome_012_chr5_35062516_C_T",
                "genome_012_chr5_35065548_C_T",
            ]
        ].sum(axis=1)
        == 0
    ) * 1
    data_900_df["SOM_cols"] = data_900_df[
        [
            "GDF9_5_132865538_T_C",
            "GDF9_5_132866205_T_C",
            "LHCGR_2_48729336_C_T",
            "FSHB_11_30234435_A_G",
            "ESR1_6_152061190_A_G",
            "ESR2_14_64227477_C_T",
        ]
    ].sum(axis=1)
    return data_900_df


def alter_prot(data_900_df, NN_COLS, add_genetic_cols=True):
    if add_genetic_cols:
        data_900_df = add_genetic(data_900_df)
    data_900_df = data_900_df.loc[
        data_900_df.test_amh_r.notna() & data_900_df.day_0_mii.notna()
    ]
    if not pd.api.types.is_integer_dtype(data_900_df.cause_pco.dtype):
        data_900_df.cause_pco = data_900_df.cause_pco.cat.codes
    data_900_df = filter_data(
        data_900_df, ~data_900_df["process_type"].isin(
            ["DAWKJ", "BIOKJ", "DD", "DS"])
    )
    data_900_df = filter_data(
        data_900_df, ~data_900_df["lek_Gonadotropiny"].str.contains("Elonva")
    )
    data_900_df = filter_data(
        data_900_df, data_900_df["ds1_3_dawka_dzienna"] < 1250)
    data_900_df = filter_data(data_900_df, data_900_df["test_amh_r"] < 15.0)
    data_900_df.reset_index(inplace=True, drop=True)
    data_900_df = (
        data_900_df[NN_COLS + ["day_0_mii", "hiper_20"]
                    ].dropna(axis=0).reset_index(drop=True)
    )
    return data_900_df


def alter_stim(data_900_df, NN_COLS, add_genetic_cols=True):
    data_900_df = filter_data(
        data_900_df, ~data_900_df["prot_type"].str.contains("prot_other")
    )
    data_900_df.reset_index(inplace=True, drop=True)
    data_900_df = pd.concat(
        [
            data_900_df,
            pd.get_dummies(
                data_900_df.prot_type.cat.remove_unused_categories(), drop_first=False
            ),
        ],
        axis=1,
    )
    data_900_df = alter_prot(data_900_df, NN_COLS, add_genetic_cols)
    return data_900_df
