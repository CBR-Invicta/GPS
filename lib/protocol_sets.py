from typing import List
from dataclasses import dataclass
import pandas as pd

from lib.data_filter import DataFilter


@dataclass
class ProtocolSets:
    PROT_TYPE: List[DataFilter]
    DS1_3_DAWKA_DZIENNA: List[DataFilter]
    DS4_7_DAWKA_DZIENNA: List[DataFilter]
    DOSING_PROTOCOLS_LONGLIST: List[DataFilter]
    DOSING_PROTOCOLS_SHORTLIST: List[DataFilter]
    DOSING_PROTOCOLS_VERYSHORTLIST: List[DataFilter]


def prepare_protocol_sets(data_2015_df: pd.DataFrame):

    PROT_TYPE = [
        DataFilter([("prot_type", "prot_long")]),
        DataFilter([("prot_type", "prot_short_antagonista")]),
        DataFilter([("prot_type", "prot_short_agonist")]),
        DataFilter([("prot_type", "prot_progesteron")]),
    ]

    DS1_3_DAWKA_DZIENNA = [
        DataFilter([("ds1_3_dawka_dzienna", 300)]),
        DataFilter([("ds1_3_dawka_dzienna", 225)]),
        DataFilter([("ds1_3_dawka_dzienna", 150)]),
    ]

    DS4_7_DAWKA_DZIENNA = [
        DataFilter([("ds4_7_dawka_dzienna", 300)]),
        DataFilter([("ds4_7_dawka_dzienna", 225)]),
        DataFilter([("ds4_7_dawka_dzienna", 150)]),
    ]

    DOSING_PROTOCOLS_LONGLIST = []
    DOSING_PROTOCOLS_SHORTLIST = []

    major_protocoles_df = (
        data_2015_df[["ds1_3_dawka_dzienna", "ds4_7_dawka_dzienna", "process_number"]]
        .groupby(["ds1_3_dawka_dzienna", "ds4_7_dawka_dzienna"])
        .count()
        .sort_index()
        .reset_index()
    )
    major_protocoles_df["percentage"] = major_protocoles_df["process_number"] / sum(
        major_protocoles_df["process_number"]
    )

    for _index, row in major_protocoles_df.iterrows():
        protocol_filter = DataFilter(
            [
                ("ds1_3_dawka_dzienna", row["ds1_3_dawka_dzienna"]),
                ("ds4_7_dawka_dzienna", row["ds4_7_dawka_dzienna"]),
            ]
        )
        if row["percentage"] > 0.01:
            DOSING_PROTOCOLS_LONGLIST += [protocol_filter]
        if row["percentage"] > 0.05:
            DOSING_PROTOCOLS_SHORTLIST += [protocol_filter]

    DOSING_PROTOCOLS_VERYSHORTLIST = []
    DOSING_PROTOCOLS_VERYSHORTLIST += [
        DataFilter([("ds1_3_dawka_dzienna", 150), ("ds4_7_dawka_dzienna", 150)])
    ]
    # DOSING_PROTOCOLS_VERYSHORTLIST += [
    #     DataFilter([("ds1_3_dawka_dzienna", 200), ("ds4_7_dawka_dzienna", 150)])
    # ]
    DOSING_PROTOCOLS_VERYSHORTLIST += [
        DataFilter([("ds1_3_dawka_dzienna", 225), ("ds4_7_dawka_dzienna", 150)])
    ]
    DOSING_PROTOCOLS_VERYSHORTLIST += [
        DataFilter([("ds1_3_dawka_dzienna", 225), ("ds4_7_dawka_dzienna", 225)])
    ]
    DOSING_PROTOCOLS_VERYSHORTLIST += [
        DataFilter([("ds1_3_dawka_dzienna", 300), ("ds4_7_dawka_dzienna", 225)])
    ]
    DOSING_PROTOCOLS_VERYSHORTLIST += [
        DataFilter([("ds1_3_dawka_dzienna", 300), ("ds4_7_dawka_dzienna", 300)])
    ]

    return ProtocolSets(
        PROT_TYPE=PROT_TYPE,
        DS1_3_DAWKA_DZIENNA=DS1_3_DAWKA_DZIENNA,
        DS4_7_DAWKA_DZIENNA=DS4_7_DAWKA_DZIENNA,
        DOSING_PROTOCOLS_LONGLIST=DOSING_PROTOCOLS_LONGLIST,
        DOSING_PROTOCOLS_SHORTLIST=DOSING_PROTOCOLS_SHORTLIST,
        DOSING_PROTOCOLS_VERYSHORTLIST=DOSING_PROTOCOLS_VERYSHORTLIST,
    )
