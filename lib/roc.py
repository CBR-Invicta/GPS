import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from lib.train import TrainResults


# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/


def show_roc_curve(
    BASE_RESULT: TrainResults,
    title: str,
    data_serie_name: str,
    target_col: str,
    prediction_col: str,
):

    MERGED_BASE2_TEST_DF = BASE_RESULT.get_merged_test_dfs_from_folds(data_serie_name)

    fpr, tpr, _thresholds = metrics.roc_curve(
        MERGED_BASE2_TEST_DF[target_col], MERGED_BASE2_TEST_DF[prediction_col]
    )

    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=title
    )
    display.plot()

    # gmeans = np.sqrt(tpr * (1-fpr))
    # ix = np.argmax(gmeans)
    # print(f'Best Threshold={"%.3f"%thresholds[ix]}, G-Mean={"%.3f"%gmeans[ix]}')
    # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

    plt.show()
