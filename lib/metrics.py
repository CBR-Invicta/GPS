from typing import Generator, Iterable, Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import pandas as pd
import statistics


MODEL_SUFFIXES = [
    "upp",
    "l2",
    "l1",
    "log_l2",
    "mape",
    "low",
    "h20",
    "h25",
    "classification",
    "custom",
]
ERROR_METRICS = ["RMSE", "MAE", "MAPE", "LIKELIHOOD", "MEDIAN_ABSOLUTE_ERROR"]
CLASS_METRICS = ["auc_mu", "multi_logloss", "ACC"]


def ACC(y_true, y_pred):
    y_pred = [np.argmax(line) for line in y_pred]
    return accuracy_score(y_true, y_pred)


def multi_logloss(y_true, y_pred):
    return log_loss(y_true, y_pred)


def auc_mu(y_true, y_score, A=None, W=None):
    """
    Compute the multi-class measure AUC Mu from prediction scores and labels.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        The true class labels in the range [0, n_samples-1]

    y_score : array, shape = [n_samples, n_classes]
        Target scores, where each row is a categorical distribution over the
        n_classes.

    A : array, shape = [n_classes, n_classes], optional
        The partition (or misclassification cost) matrix. If ``None`` A is the
        argmax partition matrix. Entry A_{i,j} is the cost of classifying an
        instance as class i when the true class is j. It is expected that
        diagonal entries in A are zero and off-diagonal entries are positive.

    W : array, shape = [n_classes, n_classes], optional
        The weight matrix for incorporating class skew into AUC Mu. If ``None``,
        the standard AUC Mu is calculated. If W is specified, it is expected to
        be a lower triangular matrix where entrix W_{i,j} is a positive float
        from 0 to 1 for the partial score between classes i and j. Entries not
        in the lower triangular portion of W must be 0 and the sum of all
        entries in W must be 1.

    Returns
    -------
    auc_mu : float

    References
    ----------
    .. [1] Kleiman, R., Page, D. ``AUC Mu: A Performance Metric for Multi-Class
           Machine Learning Models``, Proceedings of the 2019 International
           Conference on Machine Learning (ICML).

    """

    # Validate input arguments
    if not isinstance(y_score, np.ndarray):
        raise TypeError(
            "Expected y_score to be np.ndarray, got: %s" % type(y_score))
    if not y_score.ndim == 2:
        raise ValueError(
            "Expected y_score to be 2 dimensional, got: %s" % y_score.ndim)
    n_samples, n_classes = y_score.shape

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if not isinstance(y_true, np.ndarray):
        raise TypeError(
            "Expected y_true to be np.ndarray, got: %s" % type(y_true))
    if not y_true.ndim == 1:
        raise ValueError(
            "Expected y_true to be 1 dimensional, got: %s" % y_true.ndim)
    if not y_true.shape[0] == n_samples:
        raise ValueError(
            "Expected y_true to be shape %s, got: %s"
            % (str(y_score.shape), str(y_true.shape))
        )
    unique_labels = np.unique(y_true)
    if not np.all(unique_labels == np.arange(n_classes)):
        raise ValueError(
            "Expected y_true values in range 0..%i, got: %s"
            % (n_classes - 1, str(unique_labels))
        )

    if A is None:
        A = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    if not isinstance(A, np.ndarray):
        raise TypeError("Expected A to be np.ndarray, got: %s" % type(A))
    if not A.ndim == 2:
        raise ValueError("Expected A to be 2 dimensional, got: %s" % A.ndim)
    if not A.shape == (n_classes, n_classes):
        raise ValueError(
            "Expected A to be shape (%i, %i), got: %s"
            % (n_classes, n_classes, str(A.shape))
        )
    if not np.all(A.diagonal() == np.zeros(n_classes)):
        raise ValueError("Expected A to be zero on the diagonals")
    if not np.all(A >= 0):
        raise ValueError("Expected A to be non-negative")

    if W is None:
        W = np.tri(n_classes, k=-1)
        W /= W.sum()
    if not isinstance(W, np.ndarray):
        raise TypeError("Expected W to be np.ndarray, got: %s" % type(W))
    if not W.ndim == 2:
        raise ValueError("Expected W to be 2 dimensional, got: %s" % W.ndim)
    if not W.shape == (n_classes, n_classes):
        raise ValueError(
            "Expected W to be shape (%i, %i), got: %s"
            % (n_classes, n_classes, str(W.shape))
        )

    auc_total = 0.0

    for class_i in range(n_classes):
        preds_i = y_score[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):

            preds_j = y_score[y_true == class_j]
            temp_preds = np.vstack((preds_i, preds_j))
            n_j = preds_j.shape[0]
            n = n_i + n_j

            temp_labels = np.zeros((n), dtype=int)
            temp_labels[n_i:n] = 1

            v = A[class_i, :] - A[class_j, :]
            scores = np.dot(temp_preds, v)

            score_i_j = roc_auc_score(temp_labels, scores)
            auc_total += W[class_i, class_j] * score_i_j

    return auc_total


def LIKELIHOOD(predictions: Iterable[float], y: Iterable[float]):
    EPS = 0.00001
    predictions = np.maximum(EPS, predictions)
    predictions = np.minimum(1 - EPS, predictions)
    log_likelihood = np.sum(
        np.log(y * predictions + (1 - y) * (1 - predictions))
    ) / len(predictions)
    return np.exp(log_likelihood)


def RMSE(predictions: Iterable[float], y: Iterable[float]) -> float:
    return np.sqrt(np.mean((predictions - y) ** 2))


def MEDIAN_ABSOLUTE_ERROR(predictions: Iterable[float], y: Iterable[float]) -> float:
    return np.median(np.abs((predictions - y)))


def MAE(predictions: Iterable[float], y: Iterable[float]) -> float:

    return metrics.mean_absolute_error(predictions, y)


def MAPE(predictions: Iterable[float], y: Iterable[float]) -> float:

    # Handling zeros from: https://stackoverflow.com/questions/47648133/mape-calculation-in-python
    mean_actual = np.mean(y)
    if mean_actual == 0:
        mean_actual = 0.00000001

    errors = []
    for prediction, actual in zip(predictions, y):
        if actual != 0:
            error = abs((actual - prediction) / actual)
        else:
            error = abs((actual - prediction) / mean_actual)
        errors.append(error)

    return sum(errors) / len(errors)


def errors_info_keys_generator() -> Generator:

    for model_suffix in MODEL_SUFFIXES:
        for error_metric in ERROR_METRICS:
            yield (model_suffix, error_metric)


def class_errors_info_keys_generator() -> Generator:

    for model_suffix in MODEL_SUFFIXES:
        for error_metric in CLASS_METRICS:
            yield (model_suffix, error_metric)


def calculate_metric(
    error_metric: str, predictions: Iterable[float], y: Iterable[float]
) -> float:

    if error_metric == "RMSE":
        return RMSE(predictions, y)
    if error_metric == "MAE":
        return MAE(predictions, y)
    if error_metric == "MAPE":
        return MAPE(predictions, y)
    if error_metric == "LIKELIHOOD":
        return LIKELIHOOD(predictions, y)
    if error_metric == "auc_mu":
        return auc_mu(y, predictions)
    if error_metric == "multi_logloss":
        return multi_logloss(y, predictions)
    if error_metric == "ACC":
        return ACC(y, predictions)
    if error_metric == "MEDIAN_ABSOLUTE_ERROR":
        return MEDIAN_ABSOLUTE_ERROR(y, predictions)

    return np.NaN


def calculate_error_metric(
    df: pd.DataFrame,
    model_suffix: str,
    error_metric: str,
    target_col: str,
    prediction_classification=np.nan,
) -> float:

    predictions = prediction_classification
    if model_suffix != "classification":
        predictions = df[f"prediction_{model_suffix}"]

    # NOTE:
    # In most cases we compare predictions with target_col
    # In case of hiper, we use hiper_col
    y = df[target_col]
    if model_suffix == "h20":
        y = df["hiper_20"]
    if model_suffix == "h25":
        y = df["hiper_25"]

    return calculate_metric(error_metric, predictions, y)


@dataclass
class ErrorsInfo:
    count: int
    target_avg: float
    errors_info: Dict[Tuple[str, str], float]

    def __init__(self, count=None, target_avg=None, errors_info=None):

        self.count = count
        self.target_avg = target_avg
        self.errors_info = errors_info

    def calculate_errors(
        self, test_df: pd.DataFrame, target_col: str, model_suffixes_filter: List[str]
    ):

        self.count = len(test_df)
        self.target_avg = np.mean(test_df[target_col])
        self.errors_info = {}
        for model_suffix, error_metric in errors_info_keys_generator():
            if model_suffix not in model_suffixes_filter:
                continue
            self.errors_info[(error_metric, model_suffix)] = calculate_error_metric(
                test_df, model_suffix, error_metric, target_col
            )

        return self

    def calculate_avg_test_errors(
        self,
        test_dfs: List[pd.DataFrame],
        target_col: str,
        model_suffixes_filter: List[str],
    ):

        concated_test_df = pd.concat(test_dfs)

        self.count = len(concated_test_df)
        self.target_avg = np.mean(concated_test_df[target_col])
        self.errors_info = {}
        for model_suffix, error_metric in errors_info_keys_generator():

            if model_suffix not in model_suffixes_filter:
                continue

            if model_suffix in ["h20", "h25"] and error_metric != "LIKELIHOOD":
                self.errors_info[(error_metric, model_suffix)] = np.NaN
                continue

            self.errors_info[(error_metric, model_suffix)] = calculate_error_metric(
                concated_test_df, model_suffix, error_metric, target_col
            )

        return self

    def calculate_avg_train_errors(
        self,
        folds_train_error_info: List["ErrorsInfo"],
        model_suffixes_filter: List[str],
    ):

        self.count = None
        self.target_avg = None
        self.errors_info = {}
        for model_suffix, error_metric in errors_info_keys_generator():

            if model_suffix not in model_suffixes_filter:
                continue

            self.errors_info[(error_metric, model_suffix)] = statistics.mean(
                [
                    train_errors_info.errors_info[(error_metric, model_suffix)]
                    for train_errors_info in folds_train_error_info
                ]
            )
        return self

    def calculate_avg_errors(
        self,
        test_dfs: List[pd.DataFrame],
        target_col: str,
        model_suffixes_filter: List[str],
        filter_tuple: Optional[Tuple[str, Any]] = None,
    ):

        filtered_test_dfs = []
        for test_df in test_dfs:
            if filter_tuple is not None:
                if filter_tuple[1] is None:
                    test_df = test_df[test_df[filter_tuple[0]].isnull()]
                else:
                    test_df = test_df[test_df[filter_tuple[0]]
                                      == filter_tuple[1]]

            filtered_test_dfs += [
                test_df[
                    [
                        f"prediction_{model_suffix}"
                        for model_suffix in model_suffixes_filter
                    ]
                    + [target_col, "hiper_20", "hiper_25"]
                ]
            ]
        filtered_test_df = pd.concat(filtered_test_dfs)

        self.count = len(filtered_test_df)
        self.target_avg = np.mean(filtered_test_df[target_col])
        self.errors_info = {}
        for model_suffix, error_metric in errors_info_keys_generator():

            if model_suffix not in model_suffixes_filter:
                continue

            self.errors_info[(error_metric, model_suffix)] = calculate_error_metric(
                filtered_test_df, model_suffix, error_metric, target_col
            )

        return self

    def print_error(
        self,
        title_data_serie_name: str,
        title_fold_name: str,
        model_suffix: str,
        error_metric: str,
        base_errors_infos: Optional[List["ErrorsInfo"]] = None,
    ) -> None:

        if (error_metric, model_suffix) not in self.errors_info:
            return

        value = self.errors_info[(error_metric, model_suffix)]
        if model_suffix == "l2" and error_metric == "RMSE":
            print("\033[44m", end="")
        print(
            f"{error_metric}"
            f"[{title_fold_name}]"
            f"[{model_suffix}]"
            f"[{title_data_serie_name}]:".ljust(46, " ") + f"\033[0m"
            f' count:{str(self.count).ljust(5, " ")}',
            f' avg:{str("%.2f"%self.target_avg).ljust(5, " ")}',
            f' {"%.2f"%value}',
            end="",
        )

        if base_errors_infos is not None:
            gains = ""
            for base_errors_info in base_errors_infos:
                if (error_metric, model_suffix) in base_errors_info.errors_info:
                    base_value = base_errors_info.errors_info[
                        (error_metric, model_suffix)
                    ]
                    gain = value - base_value
                else:
                    gain = np.NaN

                if gain < 0:
                    beg_color = "\033[43m"
                    end_color = "\033[0m"
                else:
                    beg_color = ""
                    end_color = ""

                gains += str(
                    f"{beg_color}" + f'({"%.2f"%gain})' +
                    f"{end_color}" + f"    "
                )

            print(f"    {gains}")
        else:
            print("")


@dataclass
class ClassErrorsInfo:
    count: int
    target_avg: float
    errors_info: Dict[Tuple[str, str], float]

    def __init__(self, count=None, target_avg=None, errors_info=None):

        self.count = count
        self.target_avg = target_avg
        self.errors_info = errors_info

    def class_calculate_errors(
        self,
        test_df: pd.DataFrame,
        target_col: str,
        model_suffixes_filter: List[str],
        prediction_classification,
    ):

        self.count = len(test_df)
        self.target_avg = np.mean(test_df[target_col])
        self.errors_info = {}
        for model_suffix, error_metric in class_errors_info_keys_generator():
            if model_suffix not in model_suffixes_filter:
                continue
            self.errors_info[(error_metric, model_suffix)] = calculate_error_metric(
                test_df,
                model_suffix,
                error_metric,
                target_col,
                prediction_classification,
            )

        return self

    def calculate_avg_errors(
        self,
        test_dfs: List[pd.DataFrame],
        target_col: str,
        model_suffixes_filter: List[str],
        prediction_classification,
        filter_tuple: Optional[Tuple[str, Any]] = None,
    ):

        filtered_test_dfs = []
        for test_df in test_dfs:
            if filter_tuple is not None:
                if filter_tuple[1] is None:
                    test_df = test_df[test_df[filter_tuple[0]].isnull()]
                else:
                    test_df = test_df[test_df[filter_tuple[0]]
                                      == filter_tuple[1]]

            filtered_test_dfs += [
                test_df[
                    [
                        f"prediction_{model_suffix}"
                        for model_suffix in model_suffixes_filter
                    ]
                    + [target_col, "hiper_20", "hiper_25"]
                ]
            ]
        filtered_test_df = pd.concat(filtered_test_dfs)
        prediction_classification = np.concatenate(
            prediction_classification, axis=0)
        # print(prediction_classification[1])

        self.count = len(filtered_test_df)
        self.target_avg = np.mean(filtered_test_df[target_col])
        self.errors_info = {}
        for model_suffix, error_metric in class_errors_info_keys_generator():

            if model_suffix not in model_suffixes_filter:
                continue

            self.errors_info[(error_metric, model_suffix)] = calculate_error_metric(
                filtered_test_df,
                model_suffix,
                error_metric,
                target_col,
                prediction_classification,
            )

        return self

    def print_error(
        self,
        title_data_serie_name: str,
        title_fold_name: str,
        model_suffix: str,
        error_metric: str,
        base_errors_infos: Optional[List["ClassErrorsInfo"]] = None,
    ) -> None:

        if (error_metric, model_suffix) not in self.errors_info:
            return

        value = self.errors_info[(error_metric, model_suffix)]
        if model_suffix == "l2" and error_metric == "RMSE":
            print("\033[44m", end="")
        print(
            f"{error_metric}"
            f"[{title_fold_name}]"
            f"[{model_suffix}]"
            f"[{title_data_serie_name}]:".ljust(46, " ") + f"\033[0m"
            f' count:{str(self.count).ljust(5, " ")}',
            f' avg:{str("%.2f"%self.target_avg).ljust(5, " ")}',
            f' {"%.2f"%value}',
            end="",
        )

        if base_errors_infos is not None:
            gains = ""
            for base_errors_info in base_errors_infos:
                if (error_metric, model_suffix) in base_errors_info.errors_info:
                    base_value = base_errors_info.errors_info[
                        (error_metric, model_suffix)
                    ]
                    gain = value - base_value
                else:
                    gain = np.NaN

                if gain < 0:
                    beg_color = "\033[43m"
                    end_color = "\033[0m"
                else:
                    beg_color = ""
                    end_color = ""

                gains += str(
                    f"{beg_color}" + f'({"%.2f"%gain})' +
                    f"{end_color}" + f"    "
                )

            print(f"    {gains}")
        else:
            print("")
