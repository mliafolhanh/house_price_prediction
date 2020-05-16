from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error
from src.cv_model import SMWrapper
import numpy as np
import logging
k_fold = 10
logger_model_selection = logging.getLogger("model_selection")

def get_cv_score(model_, cols):
    cv_model = SMWrapper(model_, cols)
    cv = ShuffleSplit(n_splits=k_fold, random_state=0)
    score = make_scorer(mean_squared_log_error, greater_is_better=False)
    cv_scores= cross_val_score(cv_model, model_.data, model_.data[model_.target_col], cv=cv, scoring=score)
    cv_score_mean = -np.mean(cv_scores)
    cv_score_std = np.std(cv_scores) / np.sqrt(k_fold)
    return cv_score_mean, cv_score_std

def get_results_cv(results_fit, model_):
    results_cv = []
    attr = ["cols", "results", "cv_score_mean", "cv_score_std"]
    for result_fit in results_fit:
        cols, result_ = result_fit.cols, result_fit.result_
        cv_score_mean, cv_score_std= get_cv_score(model_, cols)
        logger_model_selection.info(f"With {cols} - number_features = {len(cols)} - cv_score: {cv_score_mean} - cv_std: {cv_score_std}")
        results_cv.append(dict(zip(attr, [cols, result_, cv_score_mean, cv_score_std])))
    return results_cv

def plain_selection(results_cv):
    return min(results_cv, key=lambda v: v["cv_score_mean"])

def one_standard_error_rule_selection(results_cv):
    plain_solution = plain_selection(results_cv)
    logger_model_selection.info(f"Select_cols = {plain_solution['cols']}")
    final_result = plain_solution
    upper_cv = plain_solution["cv_score_mean"] + plain_solution["cv_score_std"]
    logger_model_selection.info(f"mean_cv = {np.sqrt(plain_solution['cv_score_mean'])} --- std_cv = {plain_solution['cv_score_std']}")
    for result in results_cv:
        if (result["cv_score_mean"] < upper_cv):
            final_result = result
            break
    return final_result