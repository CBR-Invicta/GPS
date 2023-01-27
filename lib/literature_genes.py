from typing import Dict, List
import pandas as pd


def get_literature_genes_dict(df: pd.DataFrame) -> Dict[str, str]:
    genes_dict = {}
    genes_dict["AMH_19_2249478_G_T"] = "Missense Variant"
    genes_dict["ESR1_6_151842200_T_C"] = "Intron Variant"
    genes_dict["ESR1_6_151842246_A_G"] = "Intron Variant"
    genes_dict[
        "ESR2_14_64233098_C_T"
    ] = "Non Coding Transcript Variant 3 Prime downstream"
    genes_dict["ESR2_14_64257333_C_T"] = "Synonymous Variant"
    genes_dict["FSHR_2_49154446_C_T"] = "5 Prime UTR Variant"
    genes_dict["FSHR_2_48963902_C_T"] = "Missense Variant"
    genes_dict["FSHR_2_48962782_C_T"] = "Missense Variant"
    genes_dict["LHB_19_49016648_A_G"] = "Missense Variant"
    genes_dict["LHB_19_49016626_A_G"] = "Missense Variant"
    genes_dict["LHCGR_2_48755483_C_G"] = "Intron Variant"
    genes_dict["BMP15_X_50910775_C_G"] = "5 Prime UTR Variant"
    genes_dict["BMP15_X_50912016_A_G"] = "Intron Variant"
    genes_dict["BMP15_X_50911091_A_G"] = "Missense Variant"
    genes_dict["GDF9_5_132862408_C_T"] = "Synonymous Variant"
    # 'AMHR2_12_53423453_A_G'
    genes_dict["OTHER_12_53423453_A_G"] = "2KB Upstream Variant"
    # 'FSHB_11_30230805_G_T'
    genes_dict["OTHER_11_30230805_G_T"] = "2KB Upstream Variant"
    # 'BMP15_X_50910111_C_T'
    genes_dict["OTHER_X_50910111_C_T"] = "2KB Upstream Variant"

    for col in df.columns:
        if "ESR1_6_151806527" in col:
            genes_dict[col] = "Intron Tandem Repreat"

    return genes_dict


def get_literature_genes_cols(df: pd.DataFrame) -> List[str]:

    genes_dict = get_literature_genes_dict(df)

    return [f"{col}___{description}" for col, description in genes_dict.items()]
