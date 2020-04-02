import warnings
warnings.simplefilter("ignore")
import pandas as pd
import os
import sys
import logging
from code_preprocessing.selection_strategy import *
from code_preprocessing.logging_utils import setup_default_logging
from code_preprocessing.preprocess import *
from code_preprocessing.cv_model import *
from code_preprocessing.datasets import read_data
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
setup_default_logging()
logger = logging.getLogger("test_selection")

def preprocess(train_pd):
    null_imputer = NullImputer()
    remove_imputer = RemovelColsImputer()
    imputer = Pipeline([('null_imputer', null_imputer), ("remove_imputer", remove_imputer)])
    return imputer.fit(train_pd)

def find_list_category_columns(X):
    uniq_count = X.nunique()
    category_crit = int(np.quantile(uniq_count, 0.75))
    is_category = (X.dtypes == "object") | (uniq_count <= category_crit)
    return is_category

def find_col_levels(X):
    col_levels = {}
    is_category = find_list_category_columns(X)
    for col in is_category.index:
        if is_category[col]:
            col_levels[col] = list(X[col].value_counts().index)
        else:
            col_levels[col] = []
    return col_levels

def select_features_step(train_pd, model_, strategy):
    select_tool = SelectionFeatures(model_)
    return select_tool.select_features(strategy)

def cv_process(results_fit, model_):
    selected_cols = []
    for result_fit in results_fit:
        cols, result_ = result_fit.cols, result_fit.result_
        cv_model = SMWrapper(model_, cols)
        cv = ShuffleSplit(n_splits=10, random_state=0)
        score = make_scorer(mean_squared_log_error, greater_is_better=False)
        cv_score = -np.mean(cross_val_score(cv_model, model_.data, model_.data[model_.target_col], cv=cv, scoring=score))
        logger.info(f"With {cols} - number_features = {len(cols)} - cv_score: {cv_score}")
        selected_cols.append((cols, result_, cv_score))

    return min(selected_cols, key=lambda v : v[2])

def get_model(model_class, train_pd):
    predictor_cols = list(train_pd.columns[1: -1])
    target_col = train_pd.columns[-1]
    col_levels = find_col_levels(train_pd[predictor_cols])
    return model_class(train_pd, predictor_cols, target_col, col_levels)

def process(model_class):
    train_pd = read_data('train.csv')
    imputer = preprocess(train_pd)
    train_pd = imputer.transform(train_pd)
    model_ = get_model(model_class, train_pd)
    results_fit = select_features_step(train_pd, model_, "mix")
    selection = cv_process(results_fit, model_)
    final_cols, final_model = selection[:2]

    logger.info(f"Final results: With cols = {selection[0]} - number_features = {len(selection[0])} - cv_score: {np.sqrt(selection[2])}")

    test_pd = read_data("test.csv")
    test_pd = imputer.transform(test_pd)

    test_result = final_model.predict(test_pd[final_cols]).rename(train_pd.columns[-1])
    test_result =  pd.concat([test_pd['Id'], test_result], axis = 1)
    test_result.to_csv(os.path.dirname(os.path.dirname(__file__)) + f"/outputs/{model_.name}_output_forward.csv", index=False)

process(ModelOLSStats)