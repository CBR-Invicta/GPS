from typing import List, Tuple
import re
import pandas as pd
import numpy as np
from IPython.display import display


def read_wpisy_df(directory: str) -> pd.DataFrame:
    wpisy_df = pd.read_csv(f"{directory}/wpisy_kompendium_projekt.csv", sep=";")
    return wpisy_df


def get_weight_regexes() -> List[Tuple[str, int]]:
    regexes = []
    regexes += [
        (
            r"[Ww][Aa][Gg][Aa]"
            r"\s*"
            r"([Cc][Ii][Aa][Łł][Aa]){0,1}"
            r"\s*"
            r"-{0,1}"
            r":{0,1}"
            r"\s*"
            r"(ok){0,1}"
            r"\s*"
            r"(\d+)",
            3,
        )
    ]
    return regexes


def get_height_regexes() -> List[Tuple[str, int]]:
    regexes = []
    regexes += [
        (
            r"[Ww][Zz][Rr][Oo][Ss][Tt]"
            r"\s*"
            r"-{0,1}"
            r":{0,1}"
            r"\s*"
            r"(ok){0,1}"
            r"\s*"
            r"(\d+)",
            2,
        )
    ]
    return regexes


def _apply_regexes(row, regexes: List[Tuple[str, int]], min_value: int, max_value: int):

    text = row["form_description"]
    result = np.NaN
    for regex, position in regexes:
        search = re.search(regex, text)
        if not search:
            continue
        search_result = float(search[position].replace(",", "."))
        if search_result < min_value:
            continue
        if search_result > max_value:
            continue
        result = search_result
        break

    return result


def get_body_df(wpisy_df: pd.DataFrame) -> pd.DataFrame:

    wpisy_df["weight"] = wpisy_df.apply(
        _apply_regexes,
        regexes=get_weight_regexes(),
        min_value=40,
        max_value=130,
        axis="columns",
    )
    wpisy_df["height"] = wpisy_df.apply(
        _apply_regexes,
        regexes=get_height_regexes(),
        min_value=140,
        max_value=220,
        axis="columns",
    )

    body_df = wpisy_df[["person_id", "weight", "height"]].groupby(by="person_id").min()
    body_df["bmi"] = body_df["weight"] / (body_df["height"] / 100) ** 2

    print(body_df.notnull().mean())
    # display(body_df['weight'].value_counts(dropna=False).sort_index().to_frame())
    # display(body_df['heigth'].value_counts(dropna=False).sort_index().to_frame())

    return body_df
