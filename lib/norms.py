import datetime
import pandas as pd
import numpy as np


NORMS = {}
NORMS["dhea_s"] = [
    75,
    370,
    "ug/dl",
    "",
    "https://www.google.com/search?client=ubuntu&channel=fs&ei=NeY4YMHnAsyNwPAPytyy6A0&q=dhea+normy&oq=dhea+normy&gs_lcp=Cgdnd3Mtd2l6EAMyBggAEAcQHjICCAAyBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeOgkIABCwAxAIEB46BwgAELADEB5Q8p0DWNCkA2C8pgNoAXAAeACAAX-IAd0FkgEDMy40mAEAoAEBqgEHZ3dzLXdpesgBBMABAQ&sclient=gws-wiz&ved=0ahUKEwiB1ru1w4fvAhXMBhAIHUquDN0Q4dUDCAw&uact=5",
]
NORMS["shbg"] = [
    20,
    110,
    "nmol/l",
    "",
    "https://www.google.com/search?client=ubuntu&hs=dK0&channel=fs&ei=Buc4YPTmHunIrgTxtoT4BA&q=shbg+normy&oq=shbg+normy&gs_lcp=Cgdnd3Mtd2l6EAMyAggAMgYIABAHEB4yBggAEAcQHjIECAAQHjIECAAQHjIGCAAQBRAeMgYIABAHEB4yBggAEAUQHjoHCAAQRxCwA1DZuiBY2bogYIDGIGgBcAJ4AIABggGIAeABkgEDMS4xmAEAoAECoAEBqgEHZ3dzLXdpesgBCMABAQ&sclient=gws-wiz&ved=0ahUKEwj0gayZxIfvAhVppIsKHXEbAU8Q4dUDCAw&uact=5",
]
NORMS["testosterone"] = [
    1,
    2.5,
    "nmol/l",
    "",
    "https://stronazdrowia.pl/poziom-testosteronu-badanie-normy-wyniki-i-ich-interpretacja-czym-sie-rozni-testosteron-wolny-od-calkowitego/ar/c14-14204543",
]
NORMS["fsh"] = [
    3,
    9,
    "mIU/ml",
    "?",
    "https://www.medme.pl/artykuly/fsh-badanie-hormonu-norma-przebieg-i-interpretacja-wynikow,67690.html",
]
NORMS["tsh"] = [
    2.0,
    4.2,
    "uU/ml",
    "",
    "https://aleksandrarodziewicz.pl/jak-interpretowac-poziom-tsh/",
]

NORMS["lh"] = [
    1.4,
    9.6,
    "mIU/ml",
    "faza folikularna",
    "https://www.google.com/search?client=ubuntu&hs=P20&channel=fs&ei=oPE4YMzoLc6QrgT1tK2ABQ&q=lh+normy+miu%2Fml&oq=normy+lh+mi&gs_lcp=Cgdnd3Mtd2l6EAEYADIGCAAQFhAeMggIABAWEAoQHjIICAAQCBANEB4yCAgAEAgQDRAeMggIABAIEA0QHjIICAAQCBANEB4yCAgAEAgQDRAeMgUIABCGAzIFCAAQhgMyBQgAEIYDOgkIABCwAxAIEB46AggAOgQIABANOggIABANEAUQHlCKsjlYxsM5YLLVOWgBcAB4AIABb4gBiQWSAQM1LjKYAQCgAQGqAQdnd3Mtd2l6yAEDwAEB&sclient=gws-wiz",
]
NORMS["e2"] = [
    84,
    970,
    "pg/ml",
    "faza folikularna",
    "https://www.google.com/search?client=ubuntu&hs=O7I&channel=fs&ei=14xIYMPVMuGtrgTuvZK4Aw&q=normy+e2+faza+folikularna&oq=normy+e2+faza+fol&gs_lcp=Cgdnd3Mtd2l6EAEYADIICCEQFhAdEB46BwgAEEcQsAM6BggAEBYQHjoFCCEQoAFQxXJYxn5g8IsBaAFwAngAgAGSAYgBmQiSAQM2LjSYAQCgAQGqAQdnd3Mtd2l6yAEIwAEB&sclient=gws-wiz",
]
NORMS["inhibina_b"] = [
    30,
    90,
    "pg/ml",
    "faza folikularna",
    "https://wylecz.to/badania-laboratoryjne/inhibina-b-badanie/",
]

NORMS_2015 = {}
NORMS_2015["FSH"] = NORMS["fsh"]
NORMS_2015["DHEAS"] = NORMS["dhea_s"]
NORMS_2015["Inh_B"] = NORMS["inhibina_b"]
NORMS_2015["TST"] = NORMS["testosterone"]
NORMS_2015["TSH"] = NORMS["tsh"]
NORMS_2015["SHBG"] = NORMS["shbg"]
# NORMS_2015['anty_TPO'] = brak


# https://wylecz.to/badania-laboratoryjne/hormon-lh-i-hormon-fsh-normy-i-wyniki-badania-lh-i-fsh/
NORMS_FSH = {}
NORMS_FSH["faza folikularna"] = [2.8, 11.3, "mlU/ml"]
NORMS_FSH["owulacja"] = [5.8, 21.0, "mlU/ml"]
NORMS_FSH["faza lutealna"] = [1.2, 9.0, "mlU/ml"]
NORMS_FSH["menopauza"] = [21.7, 153, "mlU/ml"]


# https://www.medonet.pl/zdrowie,cykl-miesiaczkowy---dlugosc--fazy--zaburzenia-cyklu,artykul,1721973.html
CYKL = {}
CYKL["menstruacja"] = [0, 5]
CYKL["faza folikularna"] = [6, 12]
CYKL["owulacja"] = [13, 15]
CYKL["faza lutealna"] = [16, 28]


def add_norm_fsh_in_cycle(df: pd.DataFrame) -> pd.DataFrame:

    print("-")
    for phase, norm in NORMS_FSH.items():

        if phase not in CYKL:
            continue
        min_day, max_day = CYKL[phase]

        df.loc[
            (df[f"qual_cycle_day_FSH"] >= min_day)
            & (df[f"qual_cycle_day_FSH"] <= max_day)
            & (df[f"qual_result_num_FSH"] < norm[0]),
            f"norm_cycled_FSH",
        ] = 0
        df.loc[
            (df[f"qual_cycle_day_FSH"] >= min_day)
            & (df[f"qual_cycle_day_FSH"] <= max_day)
            & (df[f"qual_result_num_FSH"] >= norm[0])
            & (df[f"qual_result_num_FSH"] <= norm[1]),
            f"norm_cycled_FSH",
        ] = 1
        df.loc[
            (df[f"qual_cycle_day_FSH"] >= min_day)
            & (df[f"qual_cycle_day_FSH"] <= max_day)
            & (df[f"qual_result_num_FSH"] > norm[1]),
            f"norm_cycled_FSH",
        ] = 2

    low = len(df[df[f"norm_cycled_FSH"] == 0])
    normal = len(df[df[f"norm_cycled_FSH"] == 1])
    high = len(df[df[f"norm_cycled_FSH"] == 2])
    filled = len(df[~df[f"norm_cycled_FSH"].isnull()])

    print(
        f"cycle_day_FSH: ".ljust(17, " ")
        + f"low: {round(100*low/filled)}%, ".ljust(10, " ")
        + f"normal: {round(100*normal/filled)}%, ".ljust(10, " ")
        + f"high: {round(100*high/filled)}%, ".ljust(10, " ")
        + f"filled: {round(100*filled/len(df))}%, ".ljust(10, " ")
    )

    df[f"norm_cycled_FSH"] = df[f"norm_cycled_FSH"].astype("category")

    return df


def add_norm_info(df: pd.DataFrame) -> pd.DataFrame:

    print("-")
    for hormone, norm in NORMS.items():

        if f"test_{hormone}_r" not in df.columns:
            continue

        units = list(
            df[~df[f"test_{hormone}_u"].isnull()][f"test_{hormone}_u"].unique()
        )
        assert len(units) == 1
        assert (
            units[0] == norm[2]
        ), f"Invalid units - hormone: {hormone}, {units[0]} != {norm[2]}"

        df.loc[df[f"test_{hormone}_r"] < norm[0], f"norm_{hormone}"] = 0
        df.loc[
            (df[f"test_{hormone}_r"] >= norm[0]) & (df[f"test_{hormone}_r"] <= norm[1]),
            f"norm_{hormone}",
        ] = 1
        df.loc[df[f"test_{hormone}_r"] > norm[1], f"norm_{hormone}"] = 2

        low = len(df[df[f"norm_{hormone}"] == 0])
        normal = len(df[df[f"norm_{hormone}"] == 1])
        high = len(df[df[f"norm_{hormone}"] == 2])
        filled = len(df[~df[f"norm_{hormone}"].isnull()])

        print(
            f"{hormone}: ".ljust(17, " ")
            + f"low: {round(100*low/filled)}%, ".ljust(10, " ")
            + f"normal: {round(100*normal/filled)}%, ".ljust(10, " ")
            + f"high: {round(100*high/filled)}%, ".ljust(10, " ")
            + f"filled: {round(100*filled/len(df))}%, ".ljust(10, " ")
            + f"({norm[3]})"
        )

        df[f"norm_{hormone}"] = df[f"norm_{hormone}"].astype("category")

    print("-")
    for hormone, norm in NORMS.items():

        df[f"valid_norm_{hormone}"] = df[f"norm_{hormone}"]
        df.loc[
            (df["patient_age"] < 37)
            & (
                df["procedure_start"] - df[f"test_{hormone}_d"]
                > datetime.timedelta(days=365)
            ),
            f"valid_norm_{hormone}",
        ] = np.NaN
        df.loc[
            (df["patient_age"] >= 37)
            & (
                df["procedure_start"] - df[f"test_{hormone}_d"]
                > datetime.timedelta(days=182)
            ),
            f"valid_norm_{hormone}",
        ] = np.NaN

        low = len(df[df[f"valid_norm_{hormone}"] == 0])
        normal = len(df[df[f"valid_norm_{hormone}"] == 1])
        high = len(df[df[f"valid_norm_{hormone}"] == 2])
        filled = len(df[~df[f"valid_norm_{hormone}"].isnull()])

        print(
            f"valid_{hormone}: ".ljust(17, " ")
            + f"low: {round(100*low/filled)}%, ".ljust(10, " ")
            + f"normal: {round(100*normal/filled)}%, ".ljust(10, " ")
            + f"high: {round(100*high/filled)}%, ".ljust(10, " ")
            + f"filled: {round(100*filled/len(df))}%, ".ljust(10, " ")
            + f"({norm[3]})"
        )

    return df


def add_norm_info_2015(df: pd.DataFrame) -> pd.DataFrame:

    print("-")
    for hormone, norm in NORMS_2015.items():

        if f"qual_result_num_{hormone}" not in df.columns:
            continue

        df.loc[df[f"qual_result_num_{hormone}"] < norm[0], f"norm_{hormone}"] = 0
        df.loc[
            (df[f"qual_result_num_{hormone}"] >= norm[0])
            & (df[f"qual_result_num_{hormone}"] <= norm[1]),
            f"norm_{hormone}",
        ] = 1
        df.loc[df[f"qual_result_num_{hormone}"] > norm[1], f"norm_{hormone}"] = 2

        low = len(df[df[f"norm_{hormone}"] == 0])
        normal = len(df[df[f"norm_{hormone}"] == 1])
        high = len(df[df[f"norm_{hormone}"] == 2])
        filled = len(df[~df[f"norm_{hormone}"].isnull()])

        print(
            f"{hormone}: ".ljust(17, " ")
            + f"low: {round(100*low/filled)}%, ".ljust(10, " ")
            + f"normal: {round(100*normal/filled)}%, ".ljust(10, " ")
            + f"high: {round(100*high/filled)}%, ".ljust(10, " ")
            + f"filled: {round(100*filled/len(df))}%, ".ljust(10, " ")
            + f"({norm[3]})"
        )

        df[f"norm_{hormone}"] = df[f"norm_{hormone}"].astype("category")

    print("-")
    for hormone, norm in NORMS_2015.items():

        df[f"valid_norm_{hormone}"] = df[f"norm_{hormone}"]
        df.loc[
            (df["patient_age"] < 37) & (df[f"qual_ds_{hormone}"] < -365),
            f"valid_norm_{hormone}",
        ] = np.NaN
        df.loc[
            (df["patient_age"] >= 37) & (df[f"qual_ds_{hormone}"] < -182),
            f"valid_norm_{hormone}",
        ] = np.NaN

        low = len(df[df[f"valid_norm_{hormone}"] == 0])
        normal = len(df[df[f"valid_norm_{hormone}"] == 1])
        high = len(df[df[f"valid_norm_{hormone}"] == 2])
        filled = len(df[~df[f"valid_norm_{hormone}"].isnull()])

        print(
            f"valid_{hormone}: ".ljust(17, " ")
            + f"low: {round(100*low/filled)}%, ".ljust(10, " ")
            + f"normal: {round(100*normal/filled)}%, ".ljust(10, " ")
            + f"high: {round(100*high/filled)}%, ".ljust(10, " ")
            + f"filled: {round(100*filled/len(df))}%, ".ljust(10, " ")
            + f"({norm[3]})"
        )

    return df
