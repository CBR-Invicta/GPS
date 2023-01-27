from typing import Tuple, Optional, List, Any, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display
from lib.p_value import calculate_bootstrap_p_value
from lib.utils import dataframe_to_latex


class LabelsTranslator:
    def __init__(
        self,
        labels_dict: Optional[Dict[str, str]] = None,
    ):
        if labels_dict is None:
            self.labels_dict = {}
        else:
            self.labels_dict = labels_dict

    def add_translation(self, key: str, value: str):
        self.labels_dict[key] = value

    def translate_label(self, label: str):
        if label in self.labels_dict:
            return self.labels_dict[label]
        else:
            return label


class Target:
    def __init__(
        self,
        col: str,
        aggregate_function: str,
    ):
        assert aggregate_function in ["avg", "sum", "count"]
        self.col = col
        self.aggregate_function = aggregate_function

    def get_col(self) -> str:
        return self.col

    def get_aggregate_function(self):
        return self.aggregate_function

    def get_target_name(self):
        return f"{self.get_aggregate_function()}({self.get_col()})"


class Segmentation:
    def __init__(
        self,
        col: str,
    ):
        self.col = col
        self.segments = None
        self.segment_percentage = None

    def get_col(self) -> str:
        return self.col

    def get_segments(self) -> List[Any]:
        assert self.segments is not None, "Segments not initialized"
        return self.segments

    def get_segment_percentage(self, segment: Any) -> int:
        assert self.segment_percentage is not None, "Segment percentage not initialized"
        assert (
            segment in self.segment_percentage.keys()
        ), f"Missing segment percentage: {segment}"
        return self.segment_percentage[segment]

    def calculate_segment_percentage(self, df: pd.DataFrame):
        self.segment_percentage = {}
        for segment in self.get_segments():
            self.segment_percentage[segment] = round(
                100 * len(df[df[self.get_col()] == segment]) / len(df)
            )


class SegmentationTopSegments(Segmentation):
    def __init__(
        self,
        col: str,
        top_segments: int,
    ):

        super().__init__(col)
        self.top_segments = top_segments

    def prepare_segments(self, df: pd.DataFrame):
        self.segments = list(df[self.col].value_counts().head(self.top_segments).index)
        self.calculate_segment_percentage(df)


class SegmentationSegments(Segmentation):
    def __init__(
        self,
        col: str,
        segments: List[str],
    ):
        super().__init__(col)
        self.segments = segments

    def prepare_segments(self, df: pd.DataFrame):
        self.calculate_segment_percentage(df)


def prepare_segmentation(
    col: str,
    top_segments: Optional[int] = None,
    segments: Optional[List[Any]] = None,
):
    valid_parameters = "(top_segments) | (segments)"
    if top_segments is not None:
        if segments is not None:
            raise ValueError("Invalid parameters. Must pass {valid_parameters}")
        return SegmentationTopSegments(col, top_segments)
    if segments is not None:
        if top_segments is not None:
            raise ValueError("Invalid parameters. Must pass {valid_parameters}")
        return SegmentationSegments(col, segments)
    raise ValueError("Invalid parameters. Must pass {valid_parameters}")


class Grouping:
    def __init__(self, col: str):
        self.col = col
        self.groups = None

    def get_col(self) -> str:
        return self.col

    def get_group(self, _value: Any) -> Any:
        raise NotImplementedError

    def get_groups(self):
        assert self.groups is not None, "Groups not initialized"
        return self.groups

    def prepare_groups(self, df: pd.DataFrame):
        groups = set()
        for value in list(df[self.col].unique()):
            group = self.get_group(value)
            if group is not None:
                groups.add(group)
        self.groups = list(sorted(groups))

    def filter_rows_in_any_group(self, row: pd.core.series.Series) -> bool:
        return self.get_group(row[self.get_col()]) is not None


class GroupingByRoundedValue(Grouping):
    def __init__(
        self,
        col: str,
        decimals: int,
        min_value: float,
        max_value: float,
    ):

        super().__init__(col)
        self.decimals = decimals
        self.min_value = min_value
        self.max_value = max_value

    def get_group(self, value: Any) -> Optional[float]:
        if np.isnan(value):
            return None
        group = round(value, self.decimals)
        if group < self.min_value:
            return None
        if group > self.max_value:
            return None
        return group


class GroupingByIntervals(Grouping):
    def __init__(
        self,
        col: str,
        intervals: Dict[str, Tuple[Optional[float], Optional[float]]],
    ):
        super().__init__(col)
        # TODO - VALIDATE INTERVALS (THEY MUST NOT INTERSECT AND SO ON)
        self.intervals = intervals

    def get_group(self, value: Any) -> Optional[float]:
        if np.isnan(value):
            return None
        for group, (min_value, max_value) in self.intervals.items():
            if min_value is not None and not value >= min_value:
                continue
            if max_value is not None and not value < max_value:
                continue
            return group
        return None


def prepare_grouping(
    col: str,
    decimals: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    intervals: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
):

    valid_parameters = "(decimals & min_value & max_value) | (intervals)"
    if decimals is not None and min_value is not None and max_value is not None:
        if intervals is not None:
            raise ValueError("Invalid parameters. Must pass {valid_parameters}")
        return GroupingByRoundedValue(col, decimals, min_value, max_value)
    if intervals is not None:
        if decimals is not None or min_value is not None or max_value is not None:
            raise ValueError("Invalid parameters. Must pass {valid_parameters}")
        return GroupingByIntervals(col, intervals)
    raise ValueError("Invalid parameters. Must pass {valid_parameters}")


def calculate_counterfactual_data(
    df: pd.DataFrame,
    target: Target,
    segmentation: Segmentation,
    grouping: Grouping,
    bootstrap_p_value_sample_count: int = 1000,
    skip_threshold_for_count: Optional[int] = None,
    show_p_values_on_plot: bool = False,
    show_p_values_cross_tables: bool = False,
) -> Tuple[Dict[Tuple[Any, Any], Dict[str, Any]], Dict[Tuple[Any, Any, Any], float],]:

    # Prepare groups
    grouping.prepare_groups(df)

    # Filter out data not in any group
    df = df[df.apply(grouping.filter_rows_in_any_group, axis=1)]

    # Prepare segmentation
    segmentation.prepare_segments(df)

    # Init res_dict
    res_dict = {}
    for group in grouping.get_groups():
        for segment in segmentation.get_segments():
            res_dict[group, segment] = {
                "count": 0,
                "sum": 0,
                "values_list": [],
            }

    # Iterate rows
    # Calculate count, sum, values_list
    for _index, row in df.iterrows():

        group = grouping.get_group(row[grouping.get_col()])
        if group is None:
            continue

        segment = row[segmentation.get_col()]
        if segment not in segmentation.get_segments():
            continue

        value = row[target.get_col()]
        if np.isnan(value):
            continue

        res_dict[group, segment]["count"] += 1
        res_dict[group, segment]["sum"] += value
        res_dict[group, segment]["values_list"] += [value]

    # Calculate avg
    for group in grouping.get_groups():
        for segment in segmentation.get_segments():

            if res_dict[group, segment]["count"] == 0:
                avg_value = np.NaN
            else:
                avg_value = (
                    res_dict[group, segment]["sum"] / res_dict[group, segment]["count"]
                )
            res_dict[group, segment]["avg"] = avg_value

    # Apply skip_threshold_for_count
    if skip_threshold_for_count is not None:
        for group in grouping.get_groups():
            for segment in segmentation.get_segments():
                if res_dict[group, segment]["count"] < skip_threshold_for_count:
                    res_dict[group, segment]["count"] += np.NaN
                    res_dict[group, segment]["sum"] += np.NaN
                    res_dict[group, segment]["values_list"] += []
                    res_dict[group, segment]["avg"] += np.NaN

    # Due to computational complexity
    # we calculate p_values only for needed segment combinations
    # Prepare p_value_segments
    p_value_segments = []
    if show_p_values_on_plot:
        if show_p_values_cross_tables:
            raise ValueError(
                "Both show_p_values_on_plot and show_p_values_cross_tables are True"
            )
        if len(segmentation.get_segments()) != 2:
            raise ValueError(
                f"Segmentation len must be 2 "
                f"(found {len(segmentation.get_segments())}) "
                f"with show_p_values_on_plot"
            )
        p_value_segments = [
            (segmentation.get_segments()[0], segmentation.get_segments()[1])
        ]
    if show_p_values_cross_tables:
        if show_p_values_on_plot:
            raise ValueError(
                "Both show_p_values_on_plot and show_p_values_cross_tables are True"
            )
        for segment1 in segmentation.get_segments():
            for segment2 in segmentation.get_segments():
                p_value_segments += [(segment1, segment2)]

    # Calculate p_value_dict
    p_value_dict = {}
    for group in grouping.get_groups():
        for segment1, segment2 in p_value_segments:
            x = res_dict[group, segment1]["values_list"]
            y = res_dict[group, segment2]["values_list"]
            if len(x) > 0 and len(y) > 0:
                p_value = 1 - calculate_bootstrap_p_value(
                    x, y, bootstrap_p_value_sample_count
                )
            else:
                p_value = np.NaN
            p_value_dict[group, segment1, segment2] = p_value

    return res_dict, p_value_dict


def display_p_values_cross_table(
    segmentation: Segmentation,
    grouping: Grouping,
    p_value_dict: Dict[Tuple[Any, Any, Any], float],
):

    for group in grouping.get_groups():
        group_p_values_dict = {}
        for segment1 in segmentation.get_segments():
            group_p_values_dict[segment1] = {}
            for segment2 in segmentation.get_segments():
                group_p_values_dict[segment1][segment2] = p_value_dict[
                    group, segment1, segment2
                ]
        group_p_values_df = pd.DataFrame.from_dict(group_p_values_dict, orient="index")

        print("=========================================")
        print(f"Group: {group} (grouping by {grouping.get_col()})")
        print("=========================================")
        dataframe_to_latex(group_p_values_df)
        display(group_p_values_df)


def display_segmentation_df(
    target: Target,
    segmentation: Segmentation,
    grouping: Grouping,
    cross_dict: Dict[Tuple[Any, Any], Dict[str, Any]],
    labels_translator: LabelsTranslator = LabelsTranslator(),
):

    segmentation_dict = {}
    for segment in segmentation.get_segments():
        segment_dict = {}
        for group in grouping.get_groups():
            segment_dict[f"count({group})"] = cross_dict[group, segment]["count"]
            target_name = labels_translator.translate_label(target.get_target_name())
            segment_dict[f"{target_name})({group})"] = cross_dict[group, segment][
                target.get_aggregate_function()
            ]
        segmentation_dict[segment] = segment_dict
    segmentation_df = pd.DataFrame.from_dict(segmentation_dict, orient="index")
    dataframe_to_latex(segmentation_df)
    display(segmentation_df)


def display_counterfactual_plot(
    target: Target,
    segmentation: Segmentation,
    grouping: Grouping,
    cross_dict: Dict[Tuple[Any, Any], Dict[str, Any]],
    p_value_dict: Dict[Tuple[Any, Any, Any], float],
    show_p_values_on_plot: bool = False,
    title: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    rotation: Optional[int] = 0,
    legend_location: Optional[str] = "upper left",
    labels_translator: LabelsTranslator = LabelsTranslator(),
    plot_directory: Optional[str] = None,
    filename: Optional[str] = None,
):

    # Display bars
    _fig, ax = plt.subplots(figsize=(15, 6))
    segment_bars = {}
    x_offsets = np.arange(len(grouping.get_groups()))
    bar_width = 1 / (len(segmentation.get_segments()) + 1)
    for segment_number, segment in enumerate(segmentation.get_segments()):
        bar_heights = [
            cross_dict[group, segment][target.get_aggregate_function()]
            for group in grouping.get_groups()
        ]
        segment_bars[segment] = ax.bar(
            x_offsets + bar_width * segment_number - bar_width / 2,
            bar_heights,
            bar_width,
            label=f"{segment} ({segmentation.get_segment_percentage(segment)}%)",
        )

    # Display count for each bar
    for segment, group_bars in segment_bars.items():
        for group, group_bar in zip(grouping.get_groups(), group_bars):
            count = cross_dict[group, segment]["count"]
            height = group_bar.get_height()
            ax.annotate(
                f"{count}",
                xy=(group_bar.get_x() + group_bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Display p_values on plot
    if show_p_values_on_plot and len(segmentation.get_segments()) == 2:

        for group_number, group in enumerate(grouping.get_groups()):

            p_value = p_value_dict[
                group, segmentation.get_segments()[0], segmentation.get_segments()[1]
            ]
            if np.isnan(p_value):
                continue

            height = max(
                [
                    cross_dict[group, segment][target.get_aggregate_function()]
                    for segment in segmentation.get_segments()
                ]
            )
            ax.annotate(
                f'{"%.2f"%p_value}',
                xy=(group_number, height),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Display xticks
    ax.set_xticks(x_offsets)
    ax.set_xticklabels(grouping.get_groups(), rotation=rotation)
    if len(grouping.get_groups()) == 1:
        plt.xticks([], [])

    # Display axis labels
    ylabel = labels_translator.translate_label(target.get_target_name())
    xlabel = labels_translator.translate_label(grouping.get_col())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Display title, legend, set limit to y axis
    if title is not None:
        ax.set_title(title)
    if legend_location is not None:
        ax.legend(loc=legend_location)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Save figure
    if plot_directory is not None and filename is not None:
        plt.savefig(f"{plot_directory}/plot_{filename}.png")


def counterfactual_plot(
    df: pd.DataFrame,
    target: Target,
    segmentation: Segmentation,
    grouping: Grouping,
    bootstrap_p_value_sample_count: int = 1000,
    skip_threshold_for_count: Optional[int] = None,
    show_p_values_on_plot: bool = False,
    show_p_values_cross_tables: bool = False,
    show_segmentation_df: bool = False,
    show_plot: bool = True,
    title: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    rotation: Optional[int] = 0,
    legend_location: Optional[str] = "upper left",
    labels_translator: LabelsTranslator = LabelsTranslator(),
    plot_directory: Optional[str] = None,
    filename: Optional[str] = None,
):

    cross_dict, p_value_dict = calculate_counterfactual_data(
        df=df,
        target=target,
        segmentation=segmentation,
        grouping=grouping,
        bootstrap_p_value_sample_count=bootstrap_p_value_sample_count,
        skip_threshold_for_count=skip_threshold_for_count,
        show_p_values_on_plot=show_p_values_on_plot,
        show_p_values_cross_tables=show_p_values_cross_tables,
    )

    if show_p_values_cross_tables:
        display_p_values_cross_table(
            segmentation=segmentation,
            grouping=grouping,
            p_value_dict=p_value_dict,
        )

    if show_segmentation_df:
        display_segmentation_df(
            target=target,
            segmentation=segmentation,
            grouping=grouping,
            cross_dict=cross_dict,
            labels_translator=labels_translator,
        )

    if show_plot:
        display_counterfactual_plot(
            target=target,
            segmentation=segmentation,
            grouping=grouping,
            cross_dict=cross_dict,
            p_value_dict=p_value_dict,
            show_p_values_on_plot=show_p_values_on_plot,
            title=title,
            y_lim=y_lim,
            rotation=rotation,
            legend_location=legend_location,
            labels_translator=labels_translator,
            plot_directory=plot_directory,
            filename=filename,
        )
