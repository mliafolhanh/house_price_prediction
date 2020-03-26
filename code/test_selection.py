
import pandas as pd
import os
import sys
import logging
from selection_strategy import *
from logging_utils import setup_default_logging
from preprocess import *
from cv_model import *
from sklearn.model_selection import cross_val_score
import numpy as np
setup_default_logging()
logger = logging.getLogger("test_selection")
def read_data():
    train_file =  "house_price_prediction/data/train.csv"
    train_pd = pd.read_csv(train_file)
    train_pd = train_pd.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThirdSsnPorch"})
    return train_pd
def preprocess(train_pd):
    imputer = DataFrameImputer()
    return imputer.fit_transform(train_pd)

def select_features_step(train_pd):
    predictor_cols = list(train_pd.columns[1: -1])
    target_col = train_pd.columns[-1]
    select_tool = SelectionFeatures(ModelOLSStats)
    return select_tool.select_features_forward(train_pd, predictor_cols, target_col)

def cv_process(train_pd, list_cols_combine):
    predictor_cols = list(train_pd.columns[1: -1])
    target_col = train_pd.columns[-1]
    selected_cols = []
    for cols, model in list_cols_combine:
        cv_model = SMWrapper(ModelOLSStats, predictor_cols, target_col, cols)
        cv_score = np.mean(cross_val_score(cv_model, train_pd, train_pd[target_col], scoring="mean_squared_error"))
        logger.info(f"With {cols} - number_features = {len(cols)} - cv_score: {cv_score}")
        selected_cols.append((cols, model, cv_score))

    return min(selected_cols, lambda v : v[2])
    

train_pd = read_data()
train_pd = preprocess(train_pd)
list_cols_combine = select_features_step(train_pd)
selection = cv_process(train_pd, list_cols_combine)
final_model = selection[1]
logger.info(f"Final results: With cols = {selection[0]} - number_features = {len(selection[0])} - cv_score: {selection[2]}")
