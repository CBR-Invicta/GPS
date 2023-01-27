from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from lib.read_data import (
    Data,
    test_column,
    test_date_column,
    med_avg_dose_column,
    med_days_count_column,
    med_max_ds_column,
    med_min_ds_column,
    med_sum_dose_column,
)


def plot_tests(data: Data, title: str, tests_df: pd.DataFrame) -> plt.plot:

    _fig, ax = plt.subplots(figsize=(15, round(len(tests_df) / 2) + 2))

    colors = [data.tests_info.test_colors[test_name] for test_name in tests_df.index]
    ax.barh(tests_df.index, tests_df["value_scaled"], color=colors)

    test_number = 0
    for test_name, row in tests_df.iterrows():
        value_precision = "%.2f" % row["value"]
        ax.annotate(
            f'{value_precision} ({row["meds_ds1_offset"]} days)',
            xy=(row["value_scaled"], test_number),
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="black",
            weight="bold",
            clip_on=True,
        )
        ax.annotate(
            "%.2f" % data.tests_info.test_scales[test_name],
            xy=(100, test_number),
            xytext=(0, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color="black",
            weight="bold",
            clip_on=True,
        )
        test_number += 1

    plt.xlim(0, 100)
    ax.invert_yaxis()
    plt.xlabel("percent of max")
    plt.title(f"Tests: {title}")
    plt.show()


def plot_bubble(offset: int, y: int, value: float, marker_scale: int):
    marker = "o"
    color = "blue"
    if value == 0:
        marker = "x"
        color = "blue"
        value = 5
    if np.isnan(value):
        marker = "x"
        color = "black"
        value = 5
    plt.plot(offset, y, marker=marker, color=color, markersize=value * marker_scale)


def plot_meds_and_bubbles(
    data: Data,
    title: str,
    meds_df: pd.DataFrame,
    valid_meds_df: pd.DataFrame,
    bubbles_dict: Dict[str, Dict[str, Any]],
) -> plt.plot:

    show_valid_meds = False

    _fig, ax = plt.subplots(figsize=(15, round(len(meds_df) / 2) + 4))

    # Plot meds
    med_number = 0
    for med_name, row in meds_df.iterrows():
        for day_number in range(int(row["min_ds"]), int(row["max_ds"]) + 1):
            plt.plot(
                day_number,
                med_number,
                marker=".",
                color=data.meds_info.med_colors[med_name],
            )
        med_number += 1

    # Plot valid_meds
    valid_med_number = 0
    if show_valid_meds:
        for med_name, row in valid_meds_df.iterrows():
            for day_number in range(int(row["min_ds"]), int(row["valid_max_ds"]) + 1):
                plt.plot(
                    day_number,
                    med_number + valid_med_number,
                    marker=".",
                    color=data.meds_info.med_colors[med_name],
                )
            valid_med_number += 1

    # Plot bubbles
    marker_scale = 2

    plot_bubble(
        bubbles_dict["ds1"]["meds_ds1_offset"],
        len(meds_df) + valid_med_number,
        bubbles_dict["ds1"]["count_r"],
        marker_scale,
    )
    plot_bubble(
        bubbles_dict["ds1"]["meds_ds1_offset"],
        len(meds_df) + valid_med_number + 1,
        bubbles_dict["ds1"]["count_l"],
        marker_scale,
    )

    plot_bubble(
        bubbles_dict["ds"]["meds_ds1_offset"],
        len(meds_df) + valid_med_number,
        bubbles_dict["ds"]["count_r"],
        marker_scale,
    )
    plot_bubble(
        bubbles_dict["ds"]["meds_ds1_offset"],
        len(meds_df) + valid_med_number + 1,
        bubbles_dict["ds"]["count_l"],
        marker_scale,
    )

    plot_bubble(
        bubbles_dict["pickup"]["meds_ds1_offset"],
        len(meds_df) + valid_med_number + 2,
        bubbles_dict["pickup"]["cumulus_count"],
        marker_scale,
    )
    plot_bubble(
        bubbles_dict["pickup"]["meds_ds1_offset"],
        len(meds_df) + valid_med_number + 3,
        bubbles_dict["pickup"]["cumulus_denuded"],
        marker_scale,
    )

    plot_bubble(
        bubbles_dict["day_0"]["meds_ds1_offset"],
        len(meds_df) + valid_med_number + 4,
        bubbles_dict["day_0"]["mii_count"],
        marker_scale,
    )

    plt.title(title)

    y_ticks = []
    for med_name, row in meds_df.iterrows():
        y_ticks.append(
            f'{med_name} ({row["avg_dose"]} = {row["sum_dose"]} / {row["days_count"]})'
        )
    if show_valid_meds:
        for med_name, row in valid_meds_df.iterrows():
            y_ticks.append(
                f'{med_name} ({row["avg_dose"]} = {row["sum_dose"]} / {row["days_count"]})'
            )
    y_ticks += [
        "count_right",
        "count_left",
        "cumulus_count",
        "cumulus_denuded",
        "mii_count",
    ]
    plt.yticks(range(0, len(y_ticks)), y_ticks)

    plt.vlines(0, 0, len(meds_df) + valid_med_number + 4, color="grey")
    plt.vlines(
        bubbles_dict["pickup"]["meds_ds1_offset"],
        0,
        len(meds_df) + valid_med_number + 4,
        color="grey",
    )
    x_min, x_max = ax.get_xlim()
    plt.hlines(len(meds_df) - 0.5, x_min, x_max, color="grey")
    if show_valid_meds:
        plt.hlines(len(meds_df) + valid_med_number - 0.5, x_min, x_max, color="grey")

    plt.xlabel("days since start of stimulation")
    ax.invert_yaxis()
    plt.show()


def calculate_meds_df(
    data: Data, process_dict: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:

    meds_dict = {}
    for med_name in data.meds_info.med_names:
        if np.isnan(process_dict[f"days_count_{med_name}"]):
            continue
        sum_dose = float(
            str(process_dict[med_sum_dose_column(med_name)]).replace(",", ".")
        )
        days_count = process_dict[med_days_count_column(med_name)]

        if med_avg_dose_column(med_name) in process_dict:
            avg_dose = float(
                str(process_dict[med_avg_dose_column(med_name)]).replace(",", ".")
            )
        else:
            avg_dose = float(sum_dose / days_count)
        meds_dict[med_name] = {
            "min_ds": process_dict[med_min_ds_column(med_name)],
            "max_ds": process_dict[med_max_ds_column(med_name)],
            "sum_dose": sum_dose,
            "days_count": days_count,
            "avg_dose": "%.2f" % avg_dose,
        }

    return meds_dict


def calculate_valid_meds_df(
    data: Data, process_dict: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:

    valid_meds_dict = {}
    for med_name in data.meds_info.med_names:
        if np.isnan(process_dict[f"valid_sum_dose_{med_name}"]):
            continue

        valid_meds_dict[med_name] = {
            "min_ds": process_dict[med_min_ds_column(med_name)],
            "valid_max_ds": process_dict[f"valid_max_ds_{med_name}"],
            "sum_dose": "%.2f" % process_dict[f"valid_sum_dose_{med_name}"],
            "days_count": process_dict[f"valid_days_diff_{med_name}"],
            "avg_dose": "%.2f" % process_dict[f"valid_avg_day_dose_{med_name}"],
        }

    return valid_meds_dict


def calculate_tests_df(
    data: Data, process_dict: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:

    test_dict = {}

    meds_ds1_date = process_dict["meds_ds1_date"]

    for test_name in data.tests_info.test_names:
        test_date = process_dict[test_date_column(test_name)]
        value = process_dict[test_column(test_name)]
        if str(test_date) == "NaT":
            continue
        test_dict[test_name] = {
            "date": test_date,
            "meds_ds1_offset": (test_date - meds_ds1_date).days,
            "value": value,
            "value_scaled": (100 * value / data.tests_info.test_scales[test_name]),
        }

    return test_dict


def calculate_bubbles_df(process_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:

    bubbles_dict = {}

    meds_ds1_date = process_dict["meds_ds1_date"]
    ds_date = process_dict["ds_date"]
    pickup_date = process_dict["pickup_date"]
    day_0_date = process_dict["day_0_date"]

    bubbles_dict["ds1"] = {
        "date": meds_ds1_date,
        "meds_ds1_offset": (meds_ds1_date - meds_ds1_date).days,
        "count_r": process_dict["ds1_bubble_count_r"],
        "count_l": process_dict["ds1_bubble_count_l"],
    }

    bubbles_dict["ds"] = {
        "date": ds_date,
        "meds_ds1_offset": (ds_date - meds_ds1_date).days,
        "count_r": process_dict["ds_bubble_count_prenatal_r"],
        "count_l": process_dict["ds_bubble_count_prenatal_l"],
    }

    bubbles_dict["pickup"] = {
        "date": pickup_date,
        "meds_ds1_offset": (pickup_date - meds_ds1_date).days,
        "cumulus_count": process_dict["cumulus_count"],
        "cumulus_denuded": process_dict["cumulus_denuded"],
    }

    bubbles_dict["day_0"] = {
        "date": day_0_date,
        "meds_ds1_offset": (day_0_date - meds_ds1_date).days,
        "mii_count": process_dict["day_0_mii"],
    }

    return bubbles_dict


def show_process(data: Data, process_number: str, display_frames: bool = False):

    process_df = data.input_df[data.input_df["process_number"] == process_number]
    process_dict = process_df.to_dict(orient="records")[0]
    patient_id = process_dict["__id__"]
    pickups_process_type = process_dict["process_type"]
    prot_type = process_dict["prot_type"]

    title = (
        f"Patient: {patient_id}, process: {process_number}, "
        f"meds_process_type: {pickups_process_type}, prot_type: {prot_type}"
    )
    print(title)

    meds_dict = calculate_meds_df(data, process_dict)
    valid_meds_dict = calculate_valid_meds_df(data, process_dict)
    tests_dict = calculate_tests_df(data, process_dict)
    bubbles_dict = calculate_bubbles_df(process_dict)

    meds_df = pd.DataFrame.from_dict(meds_dict, orient="index")
    if len(meds_df) != 0:
        meds_df = meds_df.sort_values(by="min_ds")
    valid_meds_df = pd.DataFrame.from_dict(valid_meds_dict, orient="index")
    if len(valid_meds_df) != 0:
        valid_meds_df = valid_meds_df.sort_values(by="min_ds")
    tests_df = pd.DataFrame.from_dict(tests_dict, orient="index")
    tests_df = tests_df[
        [
            "date",
            "meds_ds1_offset",
            "value",
            "value_scaled",
        ]
    ]
    bubbles_df = pd.DataFrame.from_dict(bubbles_dict, orient="index")
    bubbles_df = bubbles_df[
        [
            "date",
            "meds_ds1_offset",
            "count_r",
            "count_l",
            "cumulus_count",
            "cumulus_denuded",
            "mii_count",
        ]
    ]

    plot_tests(data, title, tests_df)

    plot_meds_and_bubbles(data, title, meds_df, valid_meds_df, bubbles_dict)

    # current_patient_genes_df = data.patient_genes_df[
    #    data.patient_genes_df['patient_id'] == patient_id].copy()
    # current_patient_genes_df.drop(columns='patient_id', inplace=True)
    # current_patient_genes_df = current_patient_genes_df.transpose()
    # current_patient_genes_df = current_patient_genes_df.set_axis(
    #    ["count"], axis=1, inplace=False).rename_axis('gene')
    # current_patient_poli_df = data.poli_df[data.poli_df['patient_id'] == patient_id]

    if display_frames:
        display(tests_df)
        display(bubbles_df)
        display(meds_df)
        display(valid_meds_df)
        # display(current_patient_genes_df)
        # display(current_patient_poli_df)


def show_patient(data: Data, patient_id: int, display_frames: bool = False):

    patient_data = data.input_df[data.input_df["patient_id"] == patient_id]
    patient_data = patient_data.sort_values(by="procedure_start")
    for process_number in patient_data["process_number"]:
        show_process(data, process_number, display_frames)
