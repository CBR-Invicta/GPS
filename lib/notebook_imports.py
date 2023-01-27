from typing import Optional, List, Tuple, Any, Iterable, Callable
import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from datetime import datetime
from dataclasses import asdict
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import pearsonr, spearmanr
from sklearn import datasets, metrics, model_selection, svm
from tqdm import tqdm

from lib.metrics import RMSE, auc_mu
from lib.read_data import read_data, test_column, test_date_column
from lib.read_data_2015 import read_data_2015
from lib.combinations import (
    gen_combinations,
    gen_flattened_combinations,
    train_generated_cols,
    gen_product_generators_2,
    print_cols,
    compare_cols,
    train_generated_cols_sum,
)

from lib.literature_genes import get_literature_genes_cols
from lib.train import train_data_series
from lib.filter_data import filter_data
from lib.shap_utils import (
    explain_model,
    shap_force_plot,
    shap_force_plots,
    shap_dependence_plot,
)
from lib.data_series import (
    DataSerie,
    prepare_data_series,
    prepare_data_serie,
    class_prepare_data_series,
)
from lib.plot_train_results import (
    plot_results_sorted_by_target_and_prediction,
    plot_results_sorted_by_amh_and_target,
    plot_amh_histogram,
    plot_results_with_segments_and_groups,
)
from lib.shap_utils import (
    explain_model,
    shap_dependence_plot,
    shap_get_feature_importance_df,
    get_most_important_cols,
)
from lib.boruta_selector import (
    boruta_select_cols,
    boruta_select_longlist,
    boruta_select_shortlist,
)
from lib.consts import get_consts, get_color
from lib.base_experiments import perform_base_experiments
from lib.column_sets import prepare_column_sets
from lib.gene_column_sets import prepare_gene_column_sets
from lib.simulate import simulate_columns
from lib.plot_targets import plot_targets
from lib.plot_column_relation_with_targets import plot_column_relation_with_targets
from lib.show_process import show_patient, show_process
from lib.plot_targets import plot_scatter_and_trend
from lib.roc import show_roc_curve
from lib.variant_analysis import variant_analysis
from lib.gene_profiles import add_features_csv, add_features_lda, add_features_knn
from lib.kernels import (
    TherapyPredictor,
    TherapyPredictorFactory,
    get_therapy_predictor_class_names,
    prepare_therapy_models,
)
from lib.protocol_sets import prepare_protocol_sets
from lib.split_utils import split_train_test
from lib.data_filter import DataFilter
from lib.therapy_analysis import (
    plot_all_models,
    plot_protocols,
    calculate_rmse_for_protocols,
    plot_kde_density_for_protocols,
    VisitExplainer,
    VisitExplainers,
)
from lib.genetic_data import (
    get_group_proportions_df,
    get_corelations_df,
    output_df_with_description,
    get_group_characteristics_df,
)
from lib.rarest_columns import rarest_variants_combinations, get_differences_df
from lib.statistical_tests_genes import get_statistical_significance
from lib.variant_analysis import two_proprotions_test
from lib.correspondant_analysis import (
    row_coordinates,
    column_coordinates,
    plot_CA,
    prepare_CA_data,
    summarize_group,
)
from lib.som import plot_som
from lib.utils import dataframe_to_latex
from lib.utils import dataframe_to_latex, translate_label
from lib.counterfactual_plot import (
    Target,
    Segmentation,
    Grouping,
    GroupingByRoundedValue,
    counterfactual_plot,
    LabelsTranslator,
    prepare_segmentation,
    prepare_grouping,
)
from lib.vcf import (
    read_vcf_files,
    LocusInfo,
    SnipInfo,
    Genome,
    GeneticData,
    prepare_genome_df,
    prepare_hmap_files,
    prepare_phase_files,
    read_mappings,
    read_allele_file,
    prepare_genetic_data,
    read_vcf_files_list,
    read_file_list,
)
from lib.read_haplotypes import (
    read_phase_inp_files,
    read_fastphase_files,
    read_phase_files,
    read_haploview_column,
    read_haploview_data,
    read_haploview_file,
    read_hmap_file,
)
from lib.p_value import (
    calculate_bootstrap_p_value,
    calculate_bootstrap_p_value_for_bool_columns,
)
from sklearn.model_selection import KFold
