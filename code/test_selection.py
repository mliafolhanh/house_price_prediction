
import pandas as pd
import os
import sys
from selection_strategy import *
from logging_utils import setup_default_logging
setup_default_logging()
train_file =  "house_price_prediction/train.csv"
train_pd = pd.read_csv(train_file)
train_pd = train_pd.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThirdSsnPorch"})
predictor_cols = train_pd.columns[1: -1]
target_col = train_pd.columns[-1]
select_tool = SelectionFeatures(ModelOLSStats)
best_features, best_result_fit = select_tool.select_feature_mix(train_pd, predictor_cols, target_col)
print(best_result_fit.summary())
