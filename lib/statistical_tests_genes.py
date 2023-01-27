from typing import List, Dict, Any, Optional
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kruskal, mannwhitneyu, kstest
from lib.data_series import DataSerie, prepare_data_series, prepare_data_serie
import numpy as np


def get_statistical_significance(data_900_df, GENE_COL_SETS, min_obs=20):
    data_900_df_mii = data_900_df.loc[pd.notna(data_900_df["mii_group"])].copy()
    data_900_df_mii.reset_index(drop=True, inplace=True)
    gene_column_list = GENE_COL_SETS.ALL_GENES_COLS_900
    gene_columns_test = []
    for gene_column in gene_column_list:
        if len(data_900_df_mii[gene_column].unique()) > 1:
            mwu_less = mannwhitneyu(
                data_900_df_mii.loc[data_900_df_mii[gene_column] == True, "day_0_mii"],
                data_900_df_mii.loc[data_900_df_mii[gene_column] == False, "day_0_mii"],
                alternative="greater",
            )
            mwu_greater = mannwhitneyu(
                data_900_df_mii.loc[data_900_df_mii[gene_column] == True, "day_0_mii"],
                data_900_df_mii.loc[data_900_df_mii[gene_column] == False, "day_0_mii"],
                alternative="less",
            )
            ks_less = kstest(
                data_900_df_mii.loc[data_900_df_mii[gene_column] == True, "day_0_mii"],
                data_900_df_mii.loc[data_900_df_mii[gene_column] == False, "day_0_mii"],
                alternative="less",
            )
            ks_greater = kstest(
                data_900_df_mii.loc[data_900_df_mii[gene_column] == True, "day_0_mii"],
                data_900_df_mii.loc[data_900_df_mii[gene_column] == False, "day_0_mii"],
                alternative="greater",
            )
            gene_columns_test.append(
                [
                    gene_column,
                    mwu_less[0],
                    mwu_less[1],
                    mwu_greater[0],
                    mwu_greater[1],
                    ks_less[0],
                    ks_less[1],
                    ks_greater[0],
                    ks_greater[1],
                    data_900_df.groupby(gene_column)[gene_column].count().min(),
                ]
            )
        else:
            gene_columns_test.append(
                [
                    gene_column,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
    gene_columns_test = pd.DataFrame(
        gene_columns_test,
        columns=[
            "column_name",
            "mwu_less_stat",
            "mwu_less_p",
            "mwu_greater_stat",
            "mwu_greater_p",
            "ks_less_stat",
            "ks_less_p",
            "ks_greater_stat",
            "ks_greater_p",
            "count",
        ],
    )

    mwu_less = gene_columns_test.loc[
        (gene_columns_test["mwu_less_p"] < 0.05)
        & (gene_columns_test["count"] > min_obs),
        "column_name",
    ].to_list()

    mwu_greater = gene_columns_test.loc[
        (gene_columns_test["mwu_greater_p"] < 0.05)
        & (gene_columns_test["count"] > min_obs),
        "column_name",
    ].to_list()

    ks_less = gene_columns_test.loc[
        (gene_columns_test["ks_less_p"] < 0.05)
        & (gene_columns_test["count"] > min_obs),
        "column_name",
    ].to_list()

    ks_greater = gene_columns_test.loc[
        (gene_columns_test["ks_greater_p"] < 0.05)
        & (gene_columns_test["count"] > min_obs),
        "column_name",
    ].to_list()

    tests_greater = gene_columns_test.loc[
        (gene_columns_test["mwu_greater_p"] < 0.05)
        & (gene_columns_test["ks_greater_p"] < 0.05)
        & (gene_columns_test["count"] > min_obs),
        "column_name",
    ].to_list()

    tests_less = gene_columns_test.loc[
        (gene_columns_test["ks_less_p"] < 0.05)
        & (gene_columns_test["mwu_less_p"] < 0.05)
        & (gene_columns_test["count"] > min_obs),
        "column_name",
    ].to_list()

    return (
        gene_columns_test,
        mwu_less,
        mwu_greater,
        ks_less,
        ks_greater,
        tests_less,
        tests_greater,
    )
