from typing import Dict, Tuple, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.neighbors import KernelDensity

from lib.data_filter import DataFilter
from lib.kernels import TherapyPredictorFactory
from lib.consts import get_color
from lib.split_utils import split_train_test
from lib.metrics import calculate_metric
from lib.train import TrainResults
from lib.shap_utils import shap_waterfall_plots
from lib.simulate import highlight_rows, format_float
from lib.protocol_sets import ProtocolSets
from lib.kernels import TherapyPredictor


def plot_all_models(
    x_col: str,
    y_col: str,
    therapy_predictor_factory: TherapyPredictorFactory,
    predictor_class_names: List[str],
    df: pd.DataFrame,
    data_filter: DataFilter,
):

    _fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    plt.scatter(df[x_col], df[y_col], marker="o",
                color="lightgrey", label="data")

    X_plot = np.linspace(0, max(df[x_col]), 1000)[:, None]
    for predictor_class_name in predictor_class_names:

        predictor = therapy_predictor_factory.make_therapy_predictor(
            predictor_class_name
        )
        filtered_df, _filtered_percentage = data_filter.filter_data(df)
        predictor.fit(filtered_df)
        color = predictor.get_color()
        if not predictor.has_sigma():
            y_plot = predictor.predict(X_plot)
            plt.plot(X_plot, y_plot, "-", color=color,
                     label=predictor.get_name())
        else:
            y_plot, sigma = predictor.predict_with_std(X_plot)
            plt.plot(X_plot, y_plot, "-", color=color,
                     label=predictor.get_name())
            ax.plot(X_plot, y_plot + sigma * 1.96,
                    "--", color=color, label="q0.95")
            ax.plot(X_plot, y_plot - sigma * 1.96,
                    "--", color=color, label="q0.95")

    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.legend()
    plt.show()


def plot_protocols(
    x_col: str,
    y_col: str,
    therapy_predictor_factory: TherapyPredictorFactory,
    predictor_class_name: str,
    df: pd.DataFrame,
    protocol_list: List[DataFilter],
    plot_dots: bool = False,
    kde_threshold: float = 0.025,
    hiper: bool = False
):

    _fig, _ax = plt.subplots(1, 1, figsize=(12, 10))

    X_plot = np.linspace(min(df[x_col]), max(df[x_col]), 1000)[:, None]
    Y_plot = np.linspace(min(df[x_col]), max(df[x_col]), 1000)[:, None]
    if hiper:
        Y_plot = np.linspace(min(df[y_col]), max(df[y_col]), 1000)[:, None]
    plt.plot(X_plot, Y_plot, "-", color="grey", linewidth=2, label="avg")
    for protocol_number, protocol_data_filter in enumerate(protocol_list):

        filter_name = protocol_data_filter.get_name()
        filtered_df, filtered_percentage = protocol_data_filter.filter_data(df)
        color = get_color(protocol_number)

        if plot_dots:
            plt.scatter(
                filtered_df[x_col],
                filtered_df[y_col],
                marker=".",
                color=color,
                # label=f'{filter_name}: {filtered_percentage}%',
                alpha=0.15,
            )

        kde = KernelDensity(kernel="gaussian", bandwidth=0.20).fit(
            filtered_df[[x_col]])
        kde_plot = np.exp(kde.score_samples(X_plot))

        predictor = therapy_predictor_factory.make_therapy_predictor(
            predictor_class_name, x_col, y_col
        )
        predictor.fit(filtered_df)
        y_plot = predictor.predict(X_plot)

        X_plot_kde = []
        Y_plot_kde = []
        for x, y, kde in zip(X_plot, y_plot, kde_plot):
            if kde > kde_threshold:
                X_plot_kde += [x]
                Y_plot_kde += [y]
        plt.plot(X_plot_kde, Y_plot_kde, "-", color=color, linewidth=3)
        plt.plot(
            X_plot,
            y_plot,
            "--",
            color=color,
            label=f"{filter_name}: {filtered_percentage}%",
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{predictor_class_name}")

    plt.legend()
    plt.show()


def plot_kde_density_for_protocols(
    x_col: str,
    df: pd.DataFrame,
    protocol_list: List[DataFilter],
    kde_threshold: float = 0.025,
    kde_bandwidth: float = 0.2,
):

    _fig, _ax = plt.subplots(1, 1, figsize=(12, 10))

    X_plot = np.linspace(0, max(df[x_col]), 1000)[:, None]
    for protocol_number, protocol_data_filter in enumerate(protocol_list):

        filter_name = protocol_data_filter.get_name()
        filtered_df, filtered_percentage = protocol_data_filter.filter_data(df)
        color = get_color(protocol_number)

        kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(
            filtered_df[[x_col]])
        kde_plot = kde.score_samples(X_plot)

        plt.plot(
            X_plot,
            np.exp(kde_plot),
            "-",
            color=color,
            label=f"{filter_name}: {filtered_percentage}%",
        )

    plt.hlines(
        kde_threshold,
        0,
        max(df[x_col]),
        color="grey",
        label=f"kernel density = {kde_threshold}",
    )

    plt.xlabel(x_col)
    plt.ylabel("kernel density (gaussian)")
    plt.title(f"kernel density")

    plt.legend()
    plt.show()


def calculate_rmse_for_protocols(
    x_col: str,
    y_col: str,
    therapy_predictor_factory: TherapyPredictorFactory,
    predictor_class_names: List[str],
    input_df: pd.DataFrame,
    protocol_list: List[DataFilter],
    n_folds: int,
    error_metric: str,
):

    rmse_dict = {}
    summary_rmse_dict = {}

    for predictor_class_name in predictor_class_names:

        all_folds_test_dfs = []
        splits = split_train_test(input_df, n_folds)
        for fold_number, split in enumerate(splits):
            train_df = split.train_df
            test_df = split.test_df

            fold_rmse_dict = {}
            protocol_test_dfs = []
            for protocol_data_filter in protocol_list:

                filter_name = protocol_data_filter.get_name()
                protocol_train_df, _percentage = protocol_data_filter.filter_data(
                    train_df
                )
                protocol_test_df, _percentage = protocol_data_filter.filter_data(
                    test_df
                )

                predictor = therapy_predictor_factory.make_therapy_predictor(
                    predictor_class_name, x_col, y_col
                )
                predictor.fit(protocol_train_df)

                protocol_test_df[
                    f"prediction_{predictor_class_name}"
                ] = predictor.predict(protocol_test_df[[x_col]])
                fold_rmse_dict[f"{filter_name}"] = "%.2f" % calculate_metric(
                    error_metric,
                    protocol_test_df[f"prediction_{predictor_class_name}"],
                    protocol_test_df[y_col],
                )
                protocol_test_dfs += [protocol_test_df]

            all_protocols_test_df = pd.concat(protocol_test_dfs)
            fold_rmse_dict[f"all_protocols"] = "%.2f" % calculate_metric(
                error_metric,
                all_protocols_test_df[f"prediction_{predictor_class_name}"],
                all_protocols_test_df[y_col],
            )
            all_folds_test_dfs += [all_protocols_test_df]
            rmse_dict[f"{predictor_class_name} : fold_{fold_number}"] = fold_rmse_dict

        all_folds_test_df = pd.concat(all_folds_test_dfs)
        all_folds_rmse_dict = {}
        for protocol_data_filter in protocol_list:
            filter_name = protocol_data_filter.get_name()
            filtered_all_folds_test_df, _percentage = protocol_data_filter.filter_data(
                all_folds_test_df
            )
            all_folds_rmse_dict[f"{filter_name}"] = "%.2f" % calculate_metric(
                error_metric,
                filtered_all_folds_test_df[f"prediction_{predictor_class_name}"],
                filtered_all_folds_test_df[y_col],
            )
        all_folds_rmse_dict[f"all_protocols"] = "%.2f" % calculate_metric(
            error_metric,
            all_folds_test_df[f"prediction_{predictor_class_name}"],
            all_folds_test_df[y_col],
        )
        summary_rmse_dict[f"{predictor_class_name} : all_folds"] = all_folds_rmse_dict

    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient="index")
    display(rmse_df)

    summary_rmse_df = pd.DataFrame.from_dict(summary_rmse_dict, orient="index")
    display(summary_rmse_df)


class VisitExplainer:
    def __init__(
        self,
        name: str,
        protocol_data_filters: List[DataFilter],
        kernel_models: Dict[str, Tuple[Any, Any]],
        train_results: TrainResults,
        train_cols: List[str],
        merged_test_df: pd.DataFrame,
        data_serie_name: str,
        x_col: str,
        y_col: str,
        kde_threshold: float,
    ):

        self.name = name
        self.protocol_data_filters = protocol_data_filters
        self.kernel_models = kernel_models
        self.train_results = train_results
        self.train_cols = train_cols
        self.merged_test_df = merged_test_df
        self.data_serie_name = data_serie_name
        self.x_col = x_col
        self.y_col = y_col
        self.kde_threshold = kde_threshold

    def explain_visit(self, process_number: str):

        display(Markdown(f"**=================================================**"))
        display(Markdown(f"**{self.name}**"))
        display(Markdown(f"**=================================================**"))

        process_df = self.merged_test_df[
            self.merged_test_df["process_number"] == process_number
        ].copy()

        show_cols = self.train_cols.copy()
        if "prot_type_cat" in show_cols:
            show_cols = ["prot_type"] + show_cols
            show_cols.remove("prot_type_cat")

        process_df = process_df[
            ["process_number"]
            + show_cols
            + [
                "prediction_low",
                "prediction_l2",
                "prediction_upp",
                "prediction_h20",
                "prediction_h25",
            ]
            + ["cumulus_count", "cumulus_denuded",
                "day_0_mii", "hiper_20", "hiper_25"]
        ]
        x_col_value = process_df[self.x_col].values
        assert len(x_col_value) == 1
        x_col_value = x_col_value[0]
        process_df.rename(
            columns={
                "test_amh_r": "Poziom AMH",
                "patient_age": "Wiek",
                "prev_proc-cumulus_denuded": "Liczba cumulus_denuded w poprzednim procesie",
                "prev_proc-day_0_mii": "Liczba mii w poprzednim procesie",
                "ds1_pech_licz_10_pon": "ds1_pech_licz_10_pon",
                "cause_pco": "Czy występuje zespół jajników policystycznych",
                "prediction_low": f"Predykcja {self.y_col} - dolna granica przedziału 95% ufności",
                "prediction_l2": f"Predykcja {self.y_col}",
                "prediction_upp": f"Predykcja {self.y_col} - górna granica przedziału 95% ufności",
                "prediction_h20": "Predykcja prawdopodobieństwa hiperstymulacji (cumulus > 20)",
                "prediction_h25": "Predykcja prawdopodobieństwa hiperstymulacji (cumulus > 25)",
                "cumulus_count": "Rzeczywista wartość cumulus",
                "cumulus_denuded": "Rzeczywista wartość cumulus_denuded",
                "day_0_mii": "Rzeczywista wartość mii",
                "hiper_20": "Czy wystąpiła hiperstymulacja z granicą (cumulus > 20)",
                "hiper_25": "Czy wystąpiła hiperstymulacja z granicą (cumulus > 25)",
            },
            inplace=True,
        )
        process_df = process_df.T

        display(
            process_df.style.apply(
                highlight_rows,
                min_value=1,
                max_value=1 + len(show_cols),
                color="lightblue",
                axis=0,
            )
            .apply(
                highlight_rows,
                min_value=len(show_cols) + 1,
                max_value=len(show_cols) + 6,
                color="orange",
                axis=0,
            )
            .apply(
                highlight_rows,
                min_value=len(show_cols) + 6,
                max_value=len(show_cols) + 11,
                color="lightgreen",
                axis=0,
            )
            .format(format_float)
        )

        shap_waterfall_plots(
            self.train_results,
            self.data_serie_name,
            folds=range(0, 5),
            model_suffix="l2",
            number_of_cases=1,
            filter_tuple=("process_number", process_number),
        )

        protocol_dict = {}
        for protocol_data_filter in self.protocol_data_filters:
            filter_name = protocol_data_filter.get_name()

            kde_model = self.kernel_models[filter_name][0]
            krr_model = self.kernel_models[filter_name][1]
            kde_value = np.exp(kde_model.score_samples([[x_col_value]]))[0]
            krr_value = krr_model.predict([[x_col_value]])[0]

            gain = krr_value - x_col_value
            if gain < 0:
                gain = "%.2f" % gain
            else:
                gain = "+%.2f" % gain
            if kde_value > self.kde_threshold:
                is_valid = f'wystarczająca ilość danych: (kde={"%.3f"%kde_value})'
            else:
                is_valid = f'niewystarczająca ilość danych: (kde={"%.3f"%kde_value})'

            protocol_dict[filter_name] = {
                f"predition_{self.y_col}": krr_value,
                f"gain": gain,
                f"is_valid": is_valid,
            }
        protocol_df = pd.DataFrame.from_dict(protocol_dict, orient="index")
        display(protocol_df.style.format(format_float))


class VisitExplainers:
    def __init__(self, visit_explainers: List[VisitExplainer]):
        self.visit_explainers = visit_explainers

    def explain_visits(self, process_number: str):

        for visit_explainer in self.visit_explainers:
            visit_explainer.explain_visit(process_number)
