import numpy as np
import os
import pandas as pd
import random
import time

from datetime import date, datetime
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Activation, Dense

from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

from tensorflow import keras
from tensorflow.keras import backend as K, metrics, layers, losses, optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.activations import (
    elu,
    exponential,
    hard_sigmoid,
    linear,
    relu,
    sigmoid,
    softmax,
    tanh,
    swish,
    softplus,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, LeakyReLU
from tensorflow.keras.losses import (
    binary_crossentropy,
    categorical_crossentropy,
    logcosh,
    mean_squared_error,
    poisson,
    mean_absolute_error,
)
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop, SGD, Ftrl
from tensorflow.keras.metrics import AUC, RootMeanSquaredError
from sklearn.model_selection import KFold


def full_search(
    train_data: pd.DataFrame,
    train_labels: list,
    target_dim: int,
    val_data: pd.DataFrame,
    val_labels: list,
    test_data: pd.DataFrame,
    test_labels: list,
    classification: bool,
    train_weights: list = None,
    max_iter=100,
    max_hours=5,
    target_metric_threshold=0.6,
    max_overfit=0.03,
    prefix="",
    verbose=True,
) -> list:
    """Performs initial search for promising model types (neural networks)
    that are later to be optimized. Modelling is based on a basis of
    random search - random model parameters are choosen.
    Models with the best target metric value and acceptable overfit are saved and stored
    Models are fitted in a while loop until maximum number of iterations or
    set elapse time is reached.
    After that results are exported to desired location.


    Parameters
    ----------
    train_data : pd.DataFrame
        tabular data with data to fit model on
    train_labels : list
        array-like values of output variable
    target_dim : int
        number of neurons in last layer
    val_data : pd.DataFrame
        tabular data with data to evaluate model performance on
    val_labels : list (of str values)
        array-like values of output variable
    test_data : pd.DataFrame
        tabular data with data to evaluate model performance on
    test_labels : list (of str values)
        array-like values of output variable
    classification : bool
        for classification, method uses different activation functions, metrics and losses
    max_iter : int
        stopping criterium in while loop - number of iterations for which
        different model classes will be generated
    max_hours : float
        stopping criterium in while loop (models will generated for that
        number of hours)
    target : str
        label - output variable
    target_metric_threshold : float
        minimum target metric value with which model will be accepted
    max_overfit : float
        maximum allowed overfit between train and test target metric value
    prefix : str
        guid with which to export results (best models) to pointed location

    Returns
    -------
    models_list : list
        list of models that met the criteria of target metric value and overfit (tensorflow
        objects)
    result_df : pd.DataFrame
        tabular results of all generated models with their parameters, metrics
    """
    # create output export directory if it doesn't exist
    if not os.path.exists(os.getcwd() + "\\data\\model_results"):
        os.makedirs(os.getcwd() + "\\data\\model_results")
    export_path = os.getcwd() + "\\data\\model_results\\"

    # declare possible parameter values for random search
    lr = np.arange(0.0005, 0.015, 0.001)
    batch_size = np.arange(60, 200, 10)
    units = [16, 32, 64, 128, 256, 512]
    num_hidden_layers = [1, 2, 3]  # , 4, 5, 6]
    epochs = [100, 150, 200]
    dropout = np.arange(0.1, 0.3, 0.05)
    optimizers = [Adam, RMSprop]  # Nadam, RMSprop, SGD
    # , LeakyReLU(), exponential, relu
    activation = [elu, swish, softplus, relu]
    # activation = [softplus]  # ,exponential, relu

    # adjust class weights to make sure 0 and 1 classes are treated equally as
    # usually there is a scarcity of ones
    if classification:
        monitor_metric = "auc"

        def mode(x):
            return max(x)

        def compare(x, y):
            return x > y

        early_stop = EarlyStopping(
            monitor="val_auc",
            min_delta=0.05,
            patience=10,
            verbose=0,
            mode="max",
            restore_best_weights=True,
        )
        losses = ["binary_crossentropy"]
        last_activations = ["sigmoid"]
        class_weights = dict(
            enumerate(
                class_weight.compute_class_weight(
                    "balanced", np.unique(test_labels), test_labels.values
                )
            )
        )
        if len(test_labels.unique()) > 2:
            print("Solving multiclass classification problem")
            test_labels = pd.get_dummies(test_labels)
            train_labels = pd.get_dummies(train_labels)
            val_labels = pd.get_dummies(val_labels)
            losses = ["categorical_crossentropy"]
            last_activations = ["softmax"]
        else:
            print("Solving classification problem")
        if verbose:
            print("----------------------")
            print(
                f"Check input data. Train shape {train_data.shape}, test shape {test_data.shape}, val shape {val_data.shape}"
            )
            print(
                f"Missing data: {pd.isna(train_data).sum().sum()+pd.isna(test_data).sum().sum()+pd.isna(val_data).sum().sum()}"
            )
            print(
                f"Missing labels: {pd.isna(train_labels).sum().sum()+pd.isna(test_labels).sum().sum()+pd.isna(val_labels).sum().sum()}"
            )
            print(f"Label's values distribution:")
            print("----------------------")
    else:
        monitor_metric = "rmse"

        def mode(x):
            return min(x)

        def compare(x, y):
            return x < y

        print("Solving regression problem")
        last_activations = [linear]
        early_stop = EarlyStopping(
            monitor="val_root_mean_squared_error",
            min_delta=0.001,
            patience=30,
            verbose=0,
            mode="min",
            restore_best_weights=True,
        )
        losses = [
            "mean_squared_error"
        ]  # 'logcosh','mean_absolute_error' ,'poisson','mean_squared_error', 'binary_crossentropy', 'categorical_crossentropy'
        # create baselinemodel with all columns
        if verbose:

            print("----------------------")
            print(
                f"Check input data. Train shape {train_data.shape}, val shape {val_data.shape}, test shape {test_data.shape}"
            )
            print(
                f"Missing data: {pd.isna(train_data).sum().sum()+pd.isna(test_data).sum().sum()+pd.isna(val_data).sum().sum()}"
            )
            print(
                f"Missing labels: {pd.isna(train_labels).sum().sum()+pd.isna(test_labels).sum().sum()+pd.isna(val_labels).sum().sum()}"
            )
            print("----------------------")
            print(f"Label's values distribution:")
            print(
                f"Train label mean: {train_labels.mean()}, val label mean: {val_labels.mean()}, test label mean: {test_labels.mean()}"
            )
            print(
                f"Train label median: {np.median(train_labels)}, val label median: {np.median(val_labels)}, test label median: {np.median(test_labels)}"
            )
            print("----------------------")

    # initialize lists to append with model results
    val_data = (
        val_data.values,
        val_labels.values,
    )
    results = []
    models_list = []
    start_time = datetime.now()
    iteration_start_time = time.time()
    execution_time = 0
    i = 0
    while (execution_time < max_hours) and i != max_iter:
        # define parameters range to perform random search
        curr_num_hidden_layers = num_hidden_layers[
            random.randrange(0, len(num_hidden_layers), 1)
        ]
        last_activation = last_activations[
            random.randrange(0, len(last_activations), 1)
        ]
        loss_function = losses[random.randrange(0, len(losses), 1)]
        optimizer_function = optimizers[random.randrange(
            0, len(optimizers), 1)]
        current_lr = lr[random.randrange(0, len(lr), 1)]
        current_epoch = epochs[random.randrange(0, len(epochs), 1)]
        current_batch_size = batch_size[random.randrange(
            0, len(batch_size), 1)]
        current_dropout = dropout[random.randrange(0, len(dropout), 1)]

        random.seed()
        model = Sequential()
        model.add(
            Dense(
                units[random.randrange(0, len(units), 1)],
                input_dim=train_data.shape[1],
                activation=activation[random.randrange(0, len(activation), 1)],
                kernel_regularizer=regularizers.l2(0.01),
            )
        )
        for j in range(curr_num_hidden_layers):
            model.add(Dropout(current_dropout))
            model.add(
                Dense(
                    units[random.randrange(0, len(units), 1)],
                    activation=activation[random.randrange(
                        0, len(activation), 1)],
                    kernel_regularizer=regularizers.l2(0.01),
                )
            )

        model.add(
            Dense(target_dim, activation=last_activation,
                  kernel_initializer="normal")
        )

        if classification:
            model.compile(
                loss=loss_function,
                optimizer=optimizer_function(lr=current_lr),
                metrics=["acc", "AUC"],
            )  # 'acc','AUC'
            model.fit(
                train_data,
                train_labels,
                epochs=current_epoch,
                batch_size=current_batch_size,
                validation_data=val_data,
                verbose=0,
                callbacks=[early_stop],
                # class_weight=class_weights,
            )
        else:
            model.compile(
                loss=loss_function,
                optimizer=optimizer_function(lr=current_lr),
                metrics=["mean_absolute_percentage_error",
                         "RootMeanSquaredError"],
            )

            if train_weights is not None:
                train_weights = np.ones(len(train_labels))

            model.fit(
                train_data,
                train_labels,
                sample_weight=train_weights,
                epochs=current_epoch,
                batch_size=current_batch_size,
                validation_data=val_data,
                verbose=0,
                callbacks=[early_stop],
            )
        scores_test = model.evaluate(test_data, test_labels, verbose=0)
        scores_val = model.evaluate(val_data[0], val_data[1], verbose=0)
        scores_train = model.evaluate(train_data, train_labels, verbose=0)

        monitor_metric_test = scores_test[2]
        monitor_metric_val = scores_val[2]
        monitor_metric_train = scores_train[2]
        if verbose:
            print(
                f"Evaluation results: test: {round(monitor_metric_test,3)}, val: {round(monitor_metric_val,3)}, train: {round(monitor_metric_train,3)}"
            )
        model_params = {
            "lr": current_lr,
            "batch_size": current_batch_size,
            "model_layers": model.get_config(),
            "num_hidden_layers": curr_num_hidden_layers,
            "epochs": current_epoch,
            "dropout": current_dropout,
            "optimizer": str(optimizer_function),
            "losses": loss_function,
            monitor_metric + "_test": monitor_metric_test,
            monitor_metric + "_val": monitor_metric_val,
            monitor_metric + "_train": monitor_metric_train,
            "columns_used": test_data.columns.to_list(),
        }
        # append results table with the latest model parameters and metrics
        results.append(model_params)

        if (
            compare(monitor_metric_test, target_metric_threshold)
            and abs(monitor_metric_val - monitor_metric_test) < max_overfit
        ):
            models_list.append(model)
            # best_models_list.append(model_params)
        else:
            models_list.append(np.nan)
        K.clear_session()
        i = i + 1
        execution_time = (time.time() - iteration_start_time) / 3600
        curr_time = datetime.now() - start_time
        if verbose:
            print("Iteration: {}. Time: {}".format(str(i), curr_time))

    results_df = pd.DataFrame(results)

    results_valid = np.where(
        (
            results_df[monitor_metric + "_test"]
            == mode(
                results_df[monitor_metric + "_test"][
                    abs(
                        results_df[monitor_metric + "_test"]
                        - results_df[monitor_metric + "_val"]
                    )
                    < max_overfit
                ]
            )
        )
        & (compare(results_df[monitor_metric + "_test"], target_metric_threshold))
    )[0]
    best_test_metric = 0
    if len(results_valid) > 0:
        best_test_metric = results_df.iloc[results_valid[0]
                                           ][monitor_metric + "_test"]
        models_list[results_valid[0]].save(
            export_path
            + str(prefix)
            + "_"
            + str(date.today())
            + "_"
            + str(datetime.now().time())[:8].replace(":", "_")
            + ".h5"
        )

    results_df.to_csv(
        export_path
        + str(prefix)
        + "_"
        + str(date.today())
        + "_"
        + str(datetime.now().time())[:8].replace(":", "_")
        + ".csv"
    )
    return models_list, results_df, best_test_metric


def narrow_search(
    train_data: pd.DataFrame,
    train_labels: list,
    target_dim: int,
    val_data: pd.DataFrame,
    val_labels: list,
    test_data: pd.DataFrame,
    test_labels: list,
    classification: bool,
    best_test_metric: float,
    columns_to_drop=None,
    max_iter=100,
    max_hours=5,
    target_metric_threshold=0.6,
    max_overfit=0.03,
    prefix="",
) -> list:
    """Performs second search in order to optimize the features of the
    dataset and find the list of columns for which results are the best.
    Feature are dropped from dataset in a loop.
    Modelling is based on a basis of random search - random model parameters
    are choosen.
    Models with the best AUC metric and acceptable overfit are saved and stored
    Models are fitted in a while loop until maximum number of iterations or
    set elapse time is reached.
    After that results are exported to desired location.


    Parameters
    ----------
    train_data : pd.DataFrame
        tabular data with data to fit model on
    train_labels : list
        list of features (independent variables) to be included in the model
        bar from label (output variable)
    test_data : pd.DataFrame
        tabular data with data to evaluate model performance on
    test_labels : list (of str values)
        array-like values of output variable
    columns_to_drop : list
        list of features to remove from dataset (selected features may
        give better results than complete list)
    max_iter : int
        stopping criterium in while loop - number of iterations for which
        different model classes will be generated
    max_hours : float
        stopping criterium in while loop (models will generated for that
        number of hours)
    target : str
        label - output variable
    min_auc : float
        minimum AUC with which model will be accepted
    max_overfit_auc : float
        maximum allowed overfit between train and test AUC
    prefix : str
        guid with which to export results (best models) to pointed location

    Returns
    -------
    models_list : list
        list of models that met the criteria of AUC and overfit (tensorflow
        objects)
    best_models_list : list
        list of model parametrs that met the criteria of AUC and overfit
    result_df : pd.DataFrame
        tabular results of all generated models with their parameters, metrics
    """
    # create output export directory if it doesn't exist
    start_time = datetime.now()
    if not os.getcwd() + "\\data\\model_results":
        os.getcwd() + "\\data\\model_results_publication"
    export_path = os.getcwd() + "\\data\\model_results"

    if columns_to_drop is None:
        columns_to_drop = train_data.columns.to_list()

    best_test_metric = best_test_metric
    all_results = []
    for c in range(len(columns_to_drop)):
        print(f"Iteration {c+1}.")
        temp_train_data = train_data.drop([columns_to_drop[c]], axis=1)
        temp_test_data = test_data.drop([columns_to_drop[c]], axis=1)
        temp_val_data = val_data.drop([columns_to_drop[c]], axis=1)

        models_list, trimmed_results, best_test_metric_iter = full_search(
            train_data=temp_train_data,
            train_labels=train_labels,
            target_dim=target_dim,
            val_data=temp_val_data,
            val_labels=val_labels,
            test_data=temp_test_data,
            test_labels=test_labels,
            classification=classification,
            max_iter=max_iter,
            max_hours=max_hours,
            target_metric_threshold=target_metric_threshold,
            max_overfit=max_overfit,
            prefix="Removed_cols_" + str(c),
            verbose=False,
        )

        all_results.append(trimmed_results)
        # if results list is not empty export results
        if best_test_metric <= best_test_metric_iter:
            print(f"Found column to drop! Removing {columns_to_drop[c]}")
            test_data = temp_test_data
            train_data = temp_train_data
            val_data = temp_val_data
            best_test_metric = best_test_metric_iter

    best_models_results_df = pd.concat(all_results)
    best_models_results_df.reset_index(drop=True, inplace=True)
    best_models_results_df.to_csv(
        export_path
        + str(prefix)
        + "_"
        + str(date.today())
        + "_final_results_of_all_iterations.csv"
    )

    return best_models_results_df


def create_nn_classification_model(X, y, optimizer):
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation=elu))

    model.add(Dense(128, activation=elu, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=elu, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=elu, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation=elu, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(16, activation=elu))
    model.add(Dropout(0.2))

    model.add(Dense(y.shape[1], activation=softmax,
              kernel_initializer="normal"))

    # define avaluation and optimization criteria
    model.compile(
        loss=categorical_crossentropy, optimizer=optimizer, metrics=[
            "acc", "AUC"]
    )
    return model


def kfold_train(dataset, labels, model, class_weights, early_stop):
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []
    auc_per_fold = []
    # K-fold Cross Validation model evaluation
    for train, test in kfold.split(dataset, labels):
        model.fit(
            dataset.loc[train],
            labels.loc[train],
            epochs=50,
            batch_size=10,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0,
        )
        scores = model.evaluate(dataset.loc[test], labels.loc[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)
        auc_per_fold.append(scores[2] * 100)
        loss_per_fold.append(scores[0])
    return (np.mean(acc_per_fold), np.mean(auc_per_fold))
