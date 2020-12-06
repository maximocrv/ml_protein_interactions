"""
This script contains the constant options used through the rest of the scripts.
"""
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# path to the data folder
data_path = '../data/'

# CSV file containing the SKEMPI v2 dataset
skempi_csv = data_path + 'skempi_v2_cleaned.csv'

# wild-type complexes data
wt_pdb_path = data_path + 'pdbs_wt/'
wt_features_path = data_path + 'features_wt/'

# mutant complexes data
mut_pdb_path = data_path + 'pdbs_wt/'
mut_features_path = data_path + 'features_mut/'

# gas constant. units: kcal mol^-1 K^-1
R = (8.314/4184)

# all these metrics are of the form fun(y_real, y_pred) and return a float32
test_metrics = {
    "pearsonr": pearsonr,
    "MSE": mean_squared_error,
    "RMSE":
    lambda y_real, y_pred: mean_squared_error(y_real, y_pred, squared=False)
}

# features for MLP output file
mlp_features = data_path + 'mlp_features.csv'
