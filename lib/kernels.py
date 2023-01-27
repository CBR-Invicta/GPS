from pyexpat import model
from typing import Optional, Tuple, Iterable, Callable, Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import lightgbm as lgb


from scipy.stats import gaussian_kde
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from warnings import simplefilter

from lib.data_filter import DataFilter


def log_x_plus_1(it: Iterable) -> Iterable:
    return np.log(it + 1)


def exp_x_minus_1(it: Iterable) -> Iterable:
    return np.exp(it) - 1


def get_therapy_predictor_class_names() -> List[str]:
    return [
        "KernelRidgePredictor",
        "KernelRidgeLogPredictor",
        "GausianProcessRegressorPredictor",
        "GausianProcessRegressorLogPredictor",
        "LGBML2Predictor",
        "LGBML2LogPredictor",
        "LGBML2MonotonicPredictor",
        "LGBML2MonotonicLogPredictor",
        "LGBMBinaryPredictor",
        "LGBMBinaryLogPredictor",
        "LGBMBinaryMonotonicPredictor",
        "LGBMBinaryMonotonicLogPredictor",
    ]


class TherapyPredictor:
    def __init__(
        self,
        name: str,
        x_col: str,
        y_col: str,
        transformations: Optional[Tuple[Callable, Callable]],
        color: str,
        model=None,
    ):

        self.name = name
        self.x_col = x_col
        self.y_col = y_col
        self.transformations = transformations
        self.color = color
        self.model = model

    def apply_transformations(self, df: pd.DataFrame):

        if self.transformations is None:
            X = df[[self.x_col]]
            y = df[self.y_col]
            return X, y
        else:
            X = pd.DataFrame()
            X[
                f"{self.transformations[0].__name__}({self.x_col})"
            ] = self.transformations[0](df[self.x_col])
            y = self.transformations[0](df[self.y_col])
            return X, y

    def fit(self, train_df: pd.DataFrame):
        raise NotImplementedError()

    def predict(self, it: Iterable):
        if self.transformations is None:
            return self.model.predict(it)
        else:
            it = self.transformations[0](it)
            return self.transformations[1](self.model.predict(it))

    def predict_with_std(self, it: Iterable):
        if self.transformations is None:
            return self.model.predict(it, return_std=True)
        else:
            it = self.transformations[0](it)
            predictions, sigma = self.model.predict(it, return_std=True)
            predictions = self.transformations[1](predictions)
            sigma = self.transformations[1](sigma)
            return predictions, sigma

    def get_color(self):
        return self.color

    def get_name(self):
        return self.name

    def has_sigma(self):
        return False


class KernelTherapyPredictor(TherapyPredictor):
    def __init__(
        self,
        name: str,
        x_col: str,
        y_col: str,
        transformations: Optional[Tuple[Callable, Callable]],
        color: str,
        model,
    ):

        super().__init__(name, x_col, y_col, transformations, color, model)

    def fit(self, train_df: pd.DataFrame):

        print(f"Fitting {type(self).__name__}: {len(train_df)}: ", end="")
        start_time = datetime.now()
        train_X, train_y = self.apply_transformations(train_df)
        self.model.fit(train_X, train_y)
        print(f"{format(datetime.now() - start_time)}")


class KernelRidgePredictor(KernelTherapyPredictor):
    def __init__(
        self,
        name: str,
        x_col: str,
        y_col: str,
        transformations: Optional[Tuple[Callable, Callable]],
        color: str,
        kernel_params: Dict[str, Any] = None
    ):
        if kernel_params is None:
            kernel_params = {
                # "alpha": [1e0, 0.1, 1e-2, 1e-3],
                # "gamma": np.logspace(-2, 2, 5),
                "alpha": [5e-2],
                "gamma": [0.1],
            }

        super().__init__(
            name,
            x_col,
            y_col,
            transformations,
            color,
            GridSearchCV(
                KernelRidge(kernel="rbf", gamma=0.1),
                param_grid=kernel_params,
            ),
        )


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


class PolyPredictor(KernelTherapyPredictor):
    def __init__(
        self,
        name: str,
        x_col: str,
        y_col: str,
        color: str,
        kernel_params: Dict[str, Any] = None
    ):
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)
        if kernel_params is None:
            kernel_params = {'polynomialfeatures__degree': np.arange(4), 'linearregression__fit_intercept': [
                True, False], 'linearregression__normalize': [True, False]}

        super().__init__(
            name,
            x_col,
            y_col,
            None,
            color,
            GridSearchCV(PolynomialRegression(), kernel_params,
                         cv=10,
                         scoring='neg_root_mean_squared_error',
                         verbose=0),
        )


class GausianProcessRegressorPredictor(KernelTherapyPredictor):
    def __init__(
        self,
        name: str,
        x_col: str,
        y_col: str,
        transformations: Optional[Tuple[Callable, Callable]],
        color: str,
    ):

        super().__init__(
            name,
            x_col,
            y_col,
            transformations,
            color,
            GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) *
                RBF(10, (1e-2, 1e2)) + WhiteKernel(),
                n_restarts_optimizer=9,
            ),
        )

    def has_sigma(self):
        return True


class LGBMPredictor(TherapyPredictor):
    def __init__(
        self,
        name: str,
        x_col: str,
        y_col: str,
        transformations: Optional[Tuple[Callable, Callable]],
        monotonic: bool,
        objective: str,
        color: str,
        lgb_params_base: Dict[str, Any],
    ):

        super().__init__(name, x_col, y_col, transformations, color, model=None)
        self.lgb_params_base = lgb_params_base
        self.monotonic = monotonic
        self.objective = objective

    def fit(self, train_df: pd.DataFrame):

        print(f"Fitting {type(self).__name__}: {len(train_df)}: ", end="")
        start_time = datetime.now()
        train_X, train_y = self.apply_transformations(train_df)
        train_dataset = lgb.Dataset(
            train_X,
            label=train_y,
        )
        lgb_params = self.lgb_params_base.copy()
        lgb_params["objective"] = self.objective
        del lgb_params["num_boost_round"]
        del lgb_params["early_stopping_round"]
        if self.monotonic:
            lgb_params["monotone_constraints_method"] = "advanced"
            lgb_params["monotone_constraints"] = [1]
        self.model = lgb.train(
            lgb_params,
            train_set=train_dataset,
            valid_sets=[train_dataset],
            verbose_eval=False,
            num_boost_round=self.lgb_params_base["num_boost_round"],
            early_stopping_rounds=self.lgb_params_base["early_stopping_round"],
        )
        print(f"{format(datetime.now() - start_time)}")


class TherapyPredictorFactory:
    def __init__(self, lgb_params_base: Dict[str, Any], kernel_params: Dict[str, Any] = None):

        self.lgb_params_base = lgb_params_base
        self.kernel_params = kernel_params

    def make_therapy_predictor(self, predictor_class_name: str, x_col: str, y_col: str):

        if predictor_class_name == "KernelRidgePredictor":
            return KernelRidgePredictor(
                predictor_class_name, x_col, y_col, None, "green", self.kernel_params
            )

        if predictor_class_name == "KernelRidgeLogPredictor":
            return KernelRidgePredictor(
                predictor_class_name,
                x_col,
                y_col,
                (log_x_plus_1, exp_x_minus_1),
                "lime",
                self.kernel_params,
            )

        if predictor_class_name == "GausianProcessRegressorPredictor":
            return GausianProcessRegressorPredictor(
                predictor_class_name, x_col, y_col, None, "red"
            )

        if predictor_class_name == "GausianProcessRegressorLogPredictor":
            return GausianProcessRegressorPredictor(
                predictor_class_name,
                x_col,
                y_col,
                (log_x_plus_1, exp_x_minus_1),
                "orange",
            )

        if predictor_class_name == "LGBML2Predictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=None,
                monotonic=False,
                objective="regression_l2",
                color="blue",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "LGBML2LogPredictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=(log_x_plus_1, exp_x_minus_1),
                monotonic=False,
                objective="regression_l2",
                color="darkviolet",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "LGBML2MonotonicPredictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=None,
                monotonic=True,
                objective="regression_l2",
                color="blue",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "LGBML2MonotonicLogPredictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=(log_x_plus_1, exp_x_minus_1),
                monotonic=True,
                objective="regression_l2",
                color="darkviolet",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "LGBMBinaryPredictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=None,
                monotonic=False,
                objective="binary",
                color="blue",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "LGBMBinaryLogPredictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=(log_x_plus_1, exp_x_minus_1),
                monotonic=False,
                objective="binary",
                color="darkviolet",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "LGBMBinaryMonotonicPredictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=None,
                monotonic=True,
                objective="binary",
                color="blue",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "LGBMBinaryMonotonicLogPredictor":
            return LGBMPredictor(
                predictor_class_name,
                x_col,
                y_col,
                transformations=(log_x_plus_1, exp_x_minus_1),
                monotonic=True,
                objective="binary",
                color="darkviolet",
                lgb_params_base=self.lgb_params_base,
            )

        if predictor_class_name == "PolyPredictor":
            return PolyPredictor(
                predictor_class_name,
                x_col,
                y_col,
                color="black",
                kernel_params=self.kernel_params
            )

        raise ValueError(
            f"Invalid predictor_class_name: {predictor_class_name}")


class kde_therapy_predictor:
    def __init__(self):

        self.protocol_data_filters: List[str] = None
        self.model = None
        self.predictions: dict = None

    def fit_kde_predictor(
        self, dataset: pd.DataFrame, x_col: str, y_col: str
    ) -> gaussian_kde:
        if not isinstance(dataset[y_col].dtype, pd.api.types.CategoricalDtype):
            raise TypeError('Inappropriate type: {} for x whereas a float \
            or int is expected'.format(dataset[y_col].dtype))
        dataset['y_col_codes'] = dataset[y_col].cat.codes
        model = gaussian_kde(
            dataset=dataset[[x_col, 'y_col_codes']].T, bw_method='scott')
        self.model = model
        self.protocol_data_filters = dataset[y_col].cat.categories.tolist()

    def predict_kde(self, prediction_mii: float, therapy: str):
        return self.model.pdf(
            [prediction_mii, self.protocol_data_filters.index(therapy)]
        )[0]

    def predict_normalized(self, prediction_mii):
        predictions = list()
        for therapy in self.protocol_data_filters:
            predictions.append(
                self.predict_kde(prediction_mii, therapy)
            )
        predictions = (predictions) / (sum(predictions))
        self.predictions = dict(zip(self.protocol_data_filters, predictions))


def prepare_kde_therapy_model(
    merged_test_df: pd.DataFrame,
    protocol_data_filters: List[DataFilter],
    x_col: str,
    y_col: str,
) -> kde_therapy_predictor:
    kde_model = kde_therapy_predictor()
    kde_model.fit_kde_predictor(
        dataset=merged_test_df, x_col=x_col, y_col=y_col)
    return kde_model


def prepare_therapy_models(
    merged_test_df: pd.DataFrame,
    protocol_data_filters: List[DataFilter],
    therapy_predictor_factory: TherapyPredictorFactory,
    predictor_class_name: str,
    x_col: str,
    y_col: str,
) -> Dict[DataFilter, TherapyPredictor]:

    models = {}

    for protocol_data_filter in protocol_data_filters:

        filtered_df, _filtered_percentage = protocol_data_filter.filter_data(
            merged_test_df
        )

        krr_model = therapy_predictor_factory.make_therapy_predictor(
            predictor_class_name, x_col, y_col
        )
        krr_model.fit(filtered_df)

        models[protocol_data_filter] = krr_model

    return models
