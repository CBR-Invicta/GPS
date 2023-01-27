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
import numpy as np


def get_differences_df(data_900_df, GENE_COL_SETS):
    differences = []
    for col in GENE_COL_SETS.ALL_GENES_COLS_900:
        if len(data_900_df[col].unique()) > 1:
            temp = data_900_df.groupby(col)["day_0_mii"].agg(["mean", "count"])
            differences.append(temp.values.reshape(4))
        else:
            differences.append(np.zeros(4))
    differences_df = pd.DataFrame(
        differences, columns=["gr1_mean", "gr1_count", "gr2_mean", "gr2_count"]
    )
    differences_df["diff"] = abs(
        differences_df["gr1_mean"] - differences_df["gr2_mean"]
    )
    differences_df["stim"] = (
        differences_df["gr1_mean"] - differences_df["gr2_mean"]
    ) < 0
    differences_df["cols"] = GENE_COL_SETS.ALL_GENES_COLS_900
    return differences_df


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
        GENE_RESULTS.train_infos[
            "900_day_0_mii"
        ].avg_test_errors_info.errors_info.values()
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
            ].avg_test_errors_info.errors_info.values()
        )[0]
        rarest_cols_temps.append(rarest_cols_temp)
        rmse_vals.append(curr_rmse)
        if curr_rmse == min(rmse_vals):
            rarest_cols_best.remove(column)
        rarest_cols_temp = rarest_cols_best.copy()
    return rarest_cols_best
