
import pandas as pd
import os
import sys
from selection_strategy import *
from logging_utils import setup_default_logging
from preprocess import *
setup_default_logging()
def read_data():
    train_file =  "house_price_prediction/train.csv"
    train_pd = pd.read_csv(train_file)
    train_pd = train_pd.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThirdSsnPorch"})
    return train_pd
def preprocess(train_pd):
    imputer = DataFrameImputer()
    return imputer.fit_transform(train_pd)

train_pd = read_data()
train_pd = preprocess(train_pd)
predictor_cols = list(train_pd.columns[1: -1])
target_col = train_pd.columns[-1]
select_tool = SelectionFeatures(ModelOLSStats)
best_features, best_result_fit = select_tool.select_features_mix(train_pd, predictor_cols, target_col)
print(best_result_fit.summary())
