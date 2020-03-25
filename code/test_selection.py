
import pandas as pd
import os
import sys
from selection_strategy import *
from logging_utils import setup_default_logging
from preprocess import *
from cv_model import *
from sklearn.model_selection import cross_val_score
setup_default_logging()
def read_data():
    train_file =  "house_price_prediction/train.csv"
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
    return select_tool.select_features_mix(train_pd, predictor_cols, target_col)

def cv_process(train_pd, list_cols_combine):
    predictor_cols = list(train_pd.columns[1: -1])
    target_col = train_pd.columns[-1]
    for cols in list_cols_combine:
        cv_model = SMWrapper(ModelOLSStats, predictor_cols, target_col, cols)
        cross_val_score(SMWrapper(sm.OLS), train_pd, None, scoring="neg_mean_squared_error"

train_pd = read_data()
train_pd = preprocess(train_pd)

print(best_result_fit.summary())
