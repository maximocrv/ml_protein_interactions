"""Tunes binary classification XGBoost model using Bayesian Optimization."""
import json
import random
from itertools import product
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from bayes_opt import BayesianOptimization
from munch import munchify
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


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
        # note that BO maximizes the output of the function (so if loss is the output, return the negative loss)
        return 1 * cv_result[f'test-{model_settings.eval_metric}-mean'].iloc[-1]

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

    params.update({'best_ntree_limit': len(cv_result)})

    xgboost_model = xgb.train(params, train_dmatrix, num_boost_round=params.best_ntree_limit)

    return xgboost_model, params, cv_result


if __name__ == "__main__":
    X, y = make_moons(10000, noise=0.3, random_state=0) 
    X = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    d_train_mat = xgb.DMatrix(x_train, y_train)

    # parameter ranges for bayesian optimization
    bayes_dictionary = {
        'n_fold': 5,
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
        'objective': 'binary:hinge',
        'eval_metric': 'map',
        # the following two params are parameters for bayesian optimization, not for actual xgboost
        'n_bayesian_optimization_iterations': 15,
        'n_init_points_bayesian_optimization': 6
    }
    
    params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'binary:hinge',
    "eval_metric": "map"
    } 

    d_test_mat = xgb.DMatrix(x_test, y_test)
    # model = xgb.train(params, d_train_mat, evals=[(d_test_mat, "test")])

    # cv_result = xgb.cv(params, d_train_mat, nfold=5)
    # print(cv_result)

    model, paras, cv_result = train_model_bayes_opt(d_train_mat, bayes_dictionary)

    print('model: ', model, '\n', 'paras: ', paras, '\n ', 'cv_result: ', cv_result)
    #https://ayguno.github.io/curious/portfolio/bayesian_optimization.html
    #https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
    #https://github.com/fmfn/BayesianOptimization
