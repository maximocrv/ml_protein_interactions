"""
This script tunes XGBoost regression model using Bayesian Optimization.
"""

import os
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from munch import munchify
from scipy.stats import pearsonr
from bayes_opt import BayesianOptimization
from sklearn.model_selection import  train_test_split

from constants import mlp_features
from utilities import load_data, open_log, clip_features_inplace, transform_data, generate_scatter_plot


def train_model_bayes_opt(train_dmatrix: xgb.DMatrix, model_settings: dict) -> Tuple[xgb.XGBClassifier, dict]:
    
    def model_function(max_depth: float, gamma: float, num_boost_round: float,
                       eta: float, subsample: float, max_delta_step: float,
                       alpha: float, reg_lambda: float) -> float:
        params = {
            'booster': 'gbtree',
            'max_depth': int(max_depth),
            'gamma': gamma,
            'eta': eta,
            'subsample': subsample,
            'max_delta_step': max_delta_step,
            'alpha': alpha,
            'reg_lambda': reg_lambda,
            'seed': model_settings.seed,
            'eval_metric': model_settings.eval_metric,
            'objective': model_settings.objective
        }

        cv_result = xgb.cv(params, train_dmatrix, num_boost_round=int(num_boost_round), nfold=model_settings.n_fold,
                           early_stopping_rounds=model_settings.early_stopping_rounds)

        # We return the negative value of confidence interval for RMSE as the Bayesian Optimization library attempts to
        # maximize the objective
        mean_rmse = cv_result[f'test-{model_settings.eval_metric}-mean'].iloc[-1]
        sd_rmse = cv_result[f'test-{model_settings.eval_metric}-std'].iloc[-1]
        return -1 * (mean_rmse + 2 * sd_rmse)

    model_settings = munchify(model_settings)

    xgb_bayesian_optimisation = BayesianOptimization(model_function, {
        'max_depth': model_settings.max_depth_range,
        'gamma': model_settings.gamma_range,
        'num_boost_round': model_settings.num_boost_rounds_range,
        'eta': model_settings.learning_rate_range,
        'subsample': model_settings.subsample_range,
        'max_delta_step': model_settings.max_delta_step_range,
        'alpha': model_settings.alpha_range,
        'reg_lambda': model_settings.lambda_range,
        }, random_state=model_settings.seed)

    xgb_bayesian_optimisation.maximize(n_iter=model_settings.n_bayesian_optimization_iterations,
                                       init_points=model_settings.n_init_points_bayesian_optimization, acq='ei')

    params = munchify(xgb_bayesian_optimisation.max['params'])

    params['max_depth'] = int(params.max_depth)
    params['objective'] = model_settings.objective
    params['eval_metric'] = model_settings.eval_metric
    params.update({'seed': model_settings.seed})

    cv_result = xgb.cv(params, train_dmatrix, num_boost_round=int(params.num_boost_round),
                       nfold=model_settings.n_fold,
                       early_stopping_rounds=model_settings.early_stopping_rounds)

    xgboost_model = xgb.train(params, train_dmatrix, num_boost_round=len(cv_result))

    return xgboost_model, params, cv_result


if __name__ == "__main__":
    # check if we are in a conda virtual env
    try:
        os.environ["CONDA_DEFAULT_ENV"]
    except KeyError:
        print("\tPlease init the conda environment!\n")
        exit(1)
    X, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    x_train, x_test = transform_data(x_train, x_test, degree=1, log=True, cross=True)
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

    print('x_train: ', x_train.shape, 'x_test: ', x_test.shape, '\n')

    d_train_mat = xgb.DMatrix(x_train, y_train)
    d_test_mat = xgb.DMatrix(x_test, y_test)

    # parameter ranges for bayesian optimization
    bayes_dictionary = {
        'n_fold': 10,
        'early_stopping_rounds': 20,
        'max_depth_range': (3, 10),
        'gamma_range': (0, 15),
        'num_boost_rounds_range': (100, 150),
        'learning_rate_range': (0.01, 0.3),
        'subsample_range': (0.8, 1),
        'max_delta_step_range': (0, 30),
        'lambda_range': (0, 5),
        'alpha_range': (0, 5),
        'seed': 1,
        # objective dictates learning task (and is used to obtain evaluation metric) 
        # refer to learning task parameters under https://xgboost.readthedocs.io/en/latest/parameter.html
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        # the following two params are parameters for bayesian optimization, not for actual xgboost
        'n_bayesian_optimization_iterations': 400,
        'n_init_points_bayesian_optimization': 40
    }

    model, paras, cv_result = train_model_bayes_opt(d_train_mat, bayes_dictionary)

    print('model: ', model, '\n', 'paras: ', paras, '\n ', 'cv_result: ', cv_result)

    d_test_mat = xgb.DMatrix(x_test, y_test)
    model.eval(d_test_mat)
