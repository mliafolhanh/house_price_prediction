
import pandas as pd
import os
import sys
import logging
from selection_strategy import *
from logging_utils import setup_default_logging
from preprocess import *
from cv_model import *
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import pickle
setup_default_logging()
logger = logging.getLogger("test_selection")
def read_data(path_file):
    train_pd = pd.read_csv(path_file)
    train_pd = train_pd.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThirdSsnPorch"})
    return train_pd
def preprocess(train_pd):
    imputer = DataFrameImputer()
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

def select_features_step(train_pd):
    predictor_cols = list(train_pd.columns[1: -1])
    target_col = train_pd.columns[-1]
    col_levels = find_col_levels(train_pd[predictor_cols])
    select_tool = SelectionFeatures(ModelOLSStats)
    return select_tool.select_features_mix(train_pd, predictor_cols, target_col, col_levels)

def cv_process(train_pd, list_cols_combine):
    predictor_cols = list(train_pd.columns[1: -1])
    target_col = train_pd.columns[-1]
    selected_cols = []
    col_levels = find_col_levels(train_pd[predictor_cols])
    print(col_levels["RoofMatl"])
    for cols, model in list_cols_combine:
        cv_model = SMWrapper(ModelOLSStats, predictor_cols, target_col, cols, col_levels)
        cv = ShuffleSplit(n_splits=10, random_state=0)
        cv_score = -np.mean(cross_val_score(cv_model, train_pd, train_pd[target_col], cv=cv, scoring="neg_mean_squared_error"))
        logger.info(f"With {cols} - number_features = {len(cols)} - cv_score: {cv_score}")
        selected_cols.append((cols, model, cv_score))

    return min(selected_cols, key=lambda v : v[2])
    

train_pd = read_data("data/train.csv")
imputer = preprocess(train_pd)
train_pd = imputer.transform(train_pd)
list_cols_combine = select_features_step(train_pd)
selection = cv_process(train_pd, list_cols_combine)
final_cols, final_model = selection[:2]

logger.info(f"Final results: With cols = {selection[0]} - number_features = {len(selection[0])} - cv_score: {np.sqrt(selection[2])}")

test_pd = read_data("data/test.csv")
test_pd = imputer.transform(test_pd)

test_result = final_model.predict(test_pd[final_cols]).rename(train_pd.columns[-1])
test_result =  pd.concat([test_pd['Id'], test_result], axis = 1)
test_result.to_csv("outputs/ols_output_forward.csv", index=False)


