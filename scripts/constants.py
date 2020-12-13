"""
This script contains the constant options used through the rest of the scripts.
"""
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np
import math

# path to the data folder
data_path = '../data/'

# CSV file containing the SKEMPI v2 dataset
skempi_csv = data_path + 'skempi_v2_cleaned.csv'

# wild-type complexes data
wt_pdb_path = data_path + 'pdbs_wt/'
wt_features_path = data_path + 'features_wt/'

# mutant complexes data
mut_pdb_path = data_path + 'pdbs_mut/'
mut_features_path = data_path + 'features_mut/'

# gas constant. units: kcal mol^-1 K^-1
R = (8.314/4184)


def _test_metrics_pearsonr(y_real, y_pred):
    R = pearsonr(y_real, y_pred)[0]
    return 0 if math.isnan(R) else R


# all these metrics are of the form fun(y_real, y_pred) and return a float32
test_metrics = {
    "pearsonr": _test_metrics_pearsonr,
    "MSE": mean_squared_error,
    "RMSE":
    lambda y_real, y_pred: mean_squared_error(y_real, y_pred, squared=False)
}

# features for MLP and XGBoost output file
mlp_features = data_path + 'extracted_features.csv'

# MLP features columns (in order)
mlp_features_columns = [
    "mut", "d_mat_wt_mean", "d_mat_wt_std",
    "u_lj_wt_mean",  "u_lj_wt_std", "u_el_wt_mean",  "u_el_wt_std",
    "d_mat_mut_mean", "d_mat_mut_std", "u_lj_mut_mean",  "u_lj_mut_std",
    "u_el_mut_mean",  "u_el_mut_std", "temp", "DDG"
]
