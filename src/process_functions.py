import warnings
warnings.simplefilter("ignore")
import pandas as pd
import os
import sys
import logging
from src.feature_selection import *
from src.logging_utils import setup_default_logging
from src.preprocess import *
from src.model_selection import *
from src.datasets import read_data
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
setup_default_logging()
logger = logging.getLogger("test_selection")

def get_imputer(train_pd):
    null_imputer = NullImputer()
    remove_imputer = RemovelColsImputer()
    imputer = Pipeline([('null_imputer', null_imputer), ("remove_imputer", remove_imputer)])
    return imputer.fit(train_pd)

def select_features_step(train_pd, model_, strategy):
    select_tool = SelectionFeatures(model_)
    return select_tool.select_features(strategy)

def get_model(model_class, train_pd, col_levels, predictor_cols, target_col):
    return model_class(train_pd, predictor_cols, target_col, col_levels)

def process(model_class):
    train_pd = read_data('train.csv')
    imputer = get_imputer(train_pd)
    train_pd = imputer.transform(train_pd)
    predictor_cols = train_pd.columns[:-1]
    target_col = train_pd.columns[-1]
    model_ = get_model(model_class, train_pd, imputer.steps[0][1].col_levels, predictor_cols, target_col)
    results_fit = select_features_step(train_pd, model_, "mix")
    results_cv = get_results_cv(results_fit, model_)
    selection = plain_selection(results_cv)
    final_cols, final_model = selection["cols"], selection["results"]

    output_info = f"Final results: With cols = {final_cols} - number_features = {len(final_cols)} - cv_score: {np.sqrt(selection['cv_score_mean'])}"
    logger.info(output_info)

    test_pd = read_data("test.csv")
    test_pd = imputer.transform(test_pd)
    test_result = final_model.predict(test_pd[final_cols])
    test_result = pd.concat([pd.Series(test_pd.index, name="Id"), test_result], axis=1)
    output_path = os.path.dirname(os.path.dirname(__file__)) + "/outputs"
    test_result.to_csv(os.path.join(output_path, f"{model_.name}_output_mix.csv"), index=False)
    with open(os.path.join(output_path, f"{model_.name}_output_info.txt"), "w") as f:
        f.write(output_info)