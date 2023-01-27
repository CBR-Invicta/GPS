from typing import Tuple, Any, Dict
import matplotlib as plt


def get_consts() -> Tuple[Any]:

    LGB_PARAMS_BASE = {
        "metric": ["rmse", "mae", "mse", "mape"],
        "boosting": "gbdt",
        "learning_rate": 0.05,
        "verbose": -1,
        "num_leaves": 5,
        "max_depth": 16,
        "max_bin": 63,
        "seed": 42,
        "num_threads": 10,
        "num_boost_round": 100,
        "early_stopping_round": 10,
    }

    N_FOLDS = 5
    ALL_MODEL_SUFFIXES = ["l1", "l2", "log_l2",
                          "mape", "upp", "low", "h20", "h25"]
    BASE_COLS_1 = [
        "test_amh_r",
    ]
    BASE_COLS_2 = [
        "test_amh_r",
        "patient_age",
        "ds1_pech_licz_10_pon",
        "prev_proc-cumulus_denuded",
        "prev_proc-day_0_mii",
        "prot_type"
    ]
    ranking_cols = [
        "GDF9_5_132865538_T_C",
        "FSHB_11_30234435_A_G",
        "AR_X_67723521_-_ACACAC",
        "GDF9_5_132866205_T_C",
        "ESR1_6_152061176_G_T",
        "PRLR_5_35064922_C_G",
        "ESR1_6_152061247_G_A",
        "LHCGR_2_48729336_C_T",
        "PRL_6_22292324_G_A",
        "PRLR_5_35072278_T_G",
        "FSHR_2_49154446_C_T___5 Prime UTR Variant",
        "ESR2_14_64227477_C_T",
        "PRLR_5_35069955_T_A",
        "PRL_6_22292747_A_T",
        "ESR1_6_152061190_A_G",
        "ESR1_6_152098960_G_A",
        "FSHR_2_48963902_C_T___Missense Variant",
        "AR_X_67723521_-_CACACAC",
        "LHCGR_2_48687476_C_G",
        "PRLR_5_35056614_C_T",
    ]

    mwu_less_no_outliers = [
        "AMH_19_2250529_G_A",
        "AR_X_67711195_A_T",
        "AR_X_67721755_T_G",
        "AR_X_67722634_A_G",
        "AR_X_67725898_T_C",
        "AR_X_67730181_AAG_-",
        "ESR1_6_151808453_C_T",
        "ESR1_6_151842149_A_G",
        "ESR1_6_151880740_T_C",
        "ESR1_6_152061190_A_G",
        "ESR1_6_152101200_C_T",
        "ESR1_6_152102770_T_A",
        "ESR2_14_64226599_T_C",
        "ESR2_14_64227364_T_C",
        "ESR2_14_64228031_T_G",
        "FSHB_11_30234748_T_G",
        "FSHR_2_48989145_C_T",
        "GDF9_5_132865378_C_T",
        "GDF9_5_132865538_T_C",
        "LHCGR_2_48709018_G_A",
        "OTHER_5_132861019_C_T",
        "OTHER_5_132861108_C_T",
        "PRLR_5_35064922_C_G",
        "PRLR_5_35069955_T_A",
        "SOX9_17_72126087_TTTTTTT_-",
        "ESR1_6_151842246_A_G___Intron Variant",
        "FSHR_2_49154446_C_T___5 Prime UTR Variant",
    ]
    stim_cols_less_selected_no_outliers = [
        "LHCGR_2_48694236_T_C",
        "ESR2_14_64226699_-_TTTTTT",
        "OTHER_6_22287169_A_T",
        "ESR1_6_151808173_G_-",
    ]

    destim_cols_less_selected_no_outliers = [
        "PRLR_5_35072610_T_G",
        "AR_X_67722634_A_G",
        "FSHR_2_48989016_C_T",
        "ESR2_14_64235053_A_G",
        "SOX9_17_72124635_A_G",
        "PRLR_5_35055852_GC_TT",
        "PRLR_5_35064922_C_G",
        "ESR1_6_151808453_C_T",
        "SOX9_17_72126087_TTTTTTT_-",
        "ESR1_6_151842149_A_G",
        "ESR1_6_152102770_T_A",
        "ESR1_6_152011609_G_A",
        "ESR2_14_64227364_T_C",
        "ESR2_14_64228031_T_G",
        "AR_X_67726853_G_C",
    ]
    ks_less_no_outliers = [
        "AR_X_67545395_C_-",
        "ESR1_6_151808173_G_C",
        "ESR1_6_152098584_C_T",
        "OTHER_6_22287169_A_T",
        "SOX9_17_72123628_C_-",
    ]
    ks_greater_no_outliers = [
        "AMH_19_2250529_G_A",
        "AR_X_67711195_A_T",
        "AR_X_67722634_A_G",
        "AR_X_67730181_AAG_-",
        "ESR1_6_151842149_A_G",
        "ESR1_6_151880740_T_C",
        "ESR1_6_152101200_C_T",
        "ESR2_14_64226599_T_C",
        "ESR2_14_64227364_T_C",
        "ESR2_14_64228031_T_G",
        "LHCGR_2_48709018_G_A",
        "FSHR_2_49154446_C_T___5 Prime UTR Variant",
    ]
    mwu_greater_no_outliers = [
        "AMHR2_12_53431536_T_C",
        "AR_X_67545395_C_-",
        "ESR1_6_151805689_T_G",
        "ESR1_6_151808173_G_-",
        "ESR1_6_151808173_G_C",
        "ESR1_6_152098584_C_T",
        "GDF9_5_132866707_T_C",
        "GDF9_5_132866719_C_G",
        "LHB_19_49016740_A_G",
        "LHCGR_2_48698552_C_G",
        "LHCGR_2_48729336_C_T",
        "OTHER_6_22287169_A_T",
        "PRLR_5_35229973_C_A",
        "PRLR_5_35230018_C_A",
        "PRLR_5_35230124_G_A",
        "PRLR_5_35230200_G_C",
        "PRLR_5_35230396_G_A",
        "PRL_6_22292324_G_A",
        "SOX3_X_140504452_A_G",
        "SOX9_17_72123628_C_-",
    ]

    return (
        LGB_PARAMS_BASE,
        N_FOLDS,
        ALL_MODEL_SUFFIXES,
        BASE_COLS_1,
        BASE_COLS_2,
        ranking_cols,
    )


def get_color(n: int):

    if n < len(list(plt.rcParams["axes.prop_cycle"])):
        return list(plt.rcParams["axes.prop_cycle"])[n]["color"]
    else:
        return "black"


def get_lgb_params_base() -> Dict[str, Any]:
    return {
        "metric": ["rmse", "mae", "mse", "mape"],
        "boosting": "gbdt",
        "learning_rate": 0.05,
        "verbose": -1,
        "num_leaves": 5,
        "max_depth": 16,
        "max_bin": 63,
        "seed": 42,
        "num_threads": 10,
        "num_boost_round": 100,
        "early_stopping_round": 10,
    }


def get_n_folds():
    return 5


nn_data_prot_subset_columns = [
    "test_amh_r",
    "prev_proc-cumulus_denuded",
    "prev_proc-day_0_mii",
    "patient_age",
    "cause_pco",
    "predictions_nn",
    "prot_type_cat",
]
nn_data_prot_subset_columns_marcin = [
    "test_amh_r",
    "prev_proc-cumulus_denuded",
    "prev_proc-day_0_mii",
    "patient_age",
    "cause_pco",
    "predictions_nn",
    "protokol",
]
