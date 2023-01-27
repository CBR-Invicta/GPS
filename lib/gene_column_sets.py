from typing import List, Dict, Any
from dataclasses import dataclass
import random
import re


from lib.read_data import Data
from lib.literature_genes import get_literature_genes_cols
from lib.data_series import DataSerie
from lib.boruta_selector import boruta_select_longlist, boruta_select_shortlist
from lib.shap_utils import shap_get_important_cols


@dataclass
class GeneColumnSets:
    ALL_GENES_COLS_900: List[str]
    LITERATURE_GENES_COLS_900: List[str]
    SHAP_IMPORTANT_GENES_COLS_900: List[str]
    BORUTA_GENES_LONGLIST_WITH_AMH_900: List[str]
    BORUTA_GENES_LONGLIST_WITHOUT_AMH_900: List[str]
    SHORTLIST_RANDOM_6_FROM_BORUTA_GENES_LONGLIST_WITH_AMH_900: List[str]
    BORUTA_GENES_SHORTLIST_WITH_AMH_900: List[str]
    BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900: List[str]
    SHORTLIST_CONST_900: List[str]
    SHORTLIST_CONST_MOTIVATING_900: List[str]
    SHORTLIST_CONST_DEMOTIVATING_900: List[str]
    CHROMOSOME_COLS_DICT: Dict[str, List[str]]


@dataclass
class GeneColumnSetsCall:
    ALL_GENES_COLS_900: List[str]
    LITERATURE_GENES_COLS_900: List[str]
    SHAP_IMPORTANT_GENES_COLS_900: List[str]
    BORUTA_GENES_LONGLIST_WITH_AMH_900: List[str]
    BORUTA_GENES_LONGLIST_WITHOUT_AMH_900: List[str]
    SHORTLIST_RANDOM_6_FROM_BORUTA_GENES_LONGLIST_WITH_AMH_900: List[str]
    BORUTA_GENES_SHORTLIST_WITH_AMH_900: List[str]
    BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900: List[str]


def prepare_gene_column_sets(
    data_900: Data,
    LGB_PARAMS_BASE: Dict[str, Any],
    BASE_COLS_2: List[str],
    DATA_SERIES: Dict[str, DataSerie],
    model_suffixes_filter: List[str],
    call: bool = False,
) -> GeneColumnSets:

    data_900_df = data_900.input_df

    ALL_GENES_COLS_900 = data_900.poli_change_names

    LITERATURE_GENES_COLS_900 = get_literature_genes_cols(data_900_df)

    print("Calculating SHAP_IMPORTANT_COLS_900")
    SHAP_IMPORTANT_GENES_COLS_900 = shap_get_important_cols(
        LGB_PARAMS_BASE,
        DATA_SERIES,
        ALL_GENES_COLS_900,
        BASE_COLS_2,
        model_suffixes_filter,
    )

    print("Calculating BORUTA_GENES_LONGLIST_WITH_AMH_900")
    BORUTA_GENES_LONGLIST_WITH_AMH_900 = boruta_select_longlist(
        DATA_SERIES, ALL_GENES_COLS_900, ["test_amh_r"]
    )

    print("Calculating BORUTA_GENES_LONGLIST_WITHOUT_AMH_900")
    BORUTA_GENES_LONGLIST_WITHOUT_AMH_900 = boruta_select_longlist(
        DATA_SERIES, ALL_GENES_COLS_900, []
    )

    SHORTLIST_RANDOM_6_FROM_BORUTA_GENES_LONGLIST_WITH_AMH_900 = random.sample(
        BORUTA_GENES_LONGLIST_WITH_AMH_900, 6
    )

    print("Calculating BORUTA_GENES_SHORTLIST_WITH_AMH_900")
    BORUTA_GENES_SHORTLIST_WITH_AMH_900 = boruta_select_shortlist(
        DATA_SERIES, BORUTA_GENES_LONGLIST_WITH_AMH_900, ["test_amh_r"], 90
    )

    print("Calculating BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900")
    BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900 = boruta_select_shortlist(
        DATA_SERIES, BORUTA_GENES_LONGLIST_WITHOUT_AMH_900, [], 90
    )

    if not call:
        SHORTLIST_CONST_900 = [
            "ESR2_14_64227364_T_C",
            "OTHER_X_50910111_C_T___2KB Upstream Variant",
            #'OTHER_X_140505223_AAAAAAA_-', - ARTEFAKT TECHNIKI
            "SOX9_17_72125967_G_A",
            "AR_X_67723521_-_ACACAC",
            "GDF9_5_132866205_T_C",
            "LHCGR_2_48729278_C_T",
        ]
        SHORTLIST_CONST_MOTIVATING_900 = [
            "OTHER_X_50910111_C_T___2KB Upstream Variant",
            "AR_X_67723521_-_ACACAC",
            "LHCGR_2_48729278_C_T",
        ]
        SHORTLIST_CONST_DEMOTIVATING_900 = [
            "GDF9_5_132866205_T_C",
            "ESR2_14_64227364_T_C",
            # "OTHER_X_140505223_AAAAAAA_-", - ARTEFAKT TECHNIKI
            "SOX9_17_72125967_G_A",
        ]

        CHROMOSOME_COLS_DICT = {}
        for col in ALL_GENES_COLS_900:
            search_results = re.search(r"^[A-Z0-9]*\_([A-Z0-9]*)\_.*$", col)
            assert search_results is not None, f"Invalid gene name: {col}"
            chromosome = search_results[1]
            if chromosome not in CHROMOSOME_COLS_DICT:
                CHROMOSOME_COLS_DICT[chromosome] = []
            CHROMOSOME_COLS_DICT[chromosome] += [col]

        return GeneColumnSets(
            ALL_GENES_COLS_900=ALL_GENES_COLS_900,
            LITERATURE_GENES_COLS_900=LITERATURE_GENES_COLS_900,
            SHAP_IMPORTANT_GENES_COLS_900=SHAP_IMPORTANT_GENES_COLS_900,
            BORUTA_GENES_LONGLIST_WITH_AMH_900=BORUTA_GENES_LONGLIST_WITH_AMH_900,
            BORUTA_GENES_LONGLIST_WITHOUT_AMH_900=BORUTA_GENES_LONGLIST_WITHOUT_AMH_900,
            SHORTLIST_RANDOM_6_FROM_BORUTA_GENES_LONGLIST_WITH_AMH_900=SHORTLIST_RANDOM_6_FROM_BORUTA_GENES_LONGLIST_WITH_AMH_900,
            BORUTA_GENES_SHORTLIST_WITH_AMH_900=BORUTA_GENES_SHORTLIST_WITH_AMH_900,
            BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900=BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900,
            SHORTLIST_CONST_900=SHORTLIST_CONST_900,
            SHORTLIST_CONST_MOTIVATING_900=SHORTLIST_CONST_MOTIVATING_900,
            SHORTLIST_CONST_DEMOTIVATING_900=SHORTLIST_CONST_DEMOTIVATING_900,
            CHROMOSOME_COLS_DICT=CHROMOSOME_COLS_DICT,
        )
    return GeneColumnSetsCall(
        ALL_GENES_COLS_900=ALL_GENES_COLS_900,
        LITERATURE_GENES_COLS_900=LITERATURE_GENES_COLS_900,
        SHAP_IMPORTANT_GENES_COLS_900=SHAP_IMPORTANT_GENES_COLS_900,
        BORUTA_GENES_LONGLIST_WITH_AMH_900=BORUTA_GENES_LONGLIST_WITH_AMH_900,
        BORUTA_GENES_LONGLIST_WITHOUT_AMH_900=BORUTA_GENES_LONGLIST_WITHOUT_AMH_900,
        SHORTLIST_RANDOM_6_FROM_BORUTA_GENES_LONGLIST_WITH_AMH_900=SHORTLIST_RANDOM_6_FROM_BORUTA_GENES_LONGLIST_WITH_AMH_900,
        BORUTA_GENES_SHORTLIST_WITH_AMH_900=BORUTA_GENES_SHORTLIST_WITH_AMH_900,
        BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900=BORUTA_GENES_SHORTLIST_WITHOUT_AMH_900,
    )
