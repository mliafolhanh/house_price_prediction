import scipy.stats as stats
import numpy as np
import logging
import pandas as pd
import re
from pandas import DataFrame
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.linear_model import LassoCV


class ResultFit:
    def __init__(self, result_, cols):
        self.result_ = result_
        self.cols = cols

    def predict(self, X):
        predict_value = self.result_.predict(X)
        return pd.Series([u if u > 0 else 1 for u in predict_value], name="SalePrice")

class ModelAbstract:
    def __init__(self, data, predictor_cols, target_col, col_levels):
        self.data = data
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.col_levels = col_levels
        self.tss = np.var(data[target_col]) * len(data)
        self.nobs = len(data)


    def fit(self, predictors_col, target_col):
        pass

    def score(self, result_fit):
        pass

    def rss(self, result_fit):
        pass

    def df_residual(self, result_fit):
        pass

    def get_p_fvalue(self, result_fit1, result_fit2):
        anova_table = anova_lm(result_fit1.result_, result_fit2.result_)
        return anova_table["Pr(>F)"][1]

class ModelOLSStats(ModelAbstract):
    def __init__(self, data , predictor_cols, target_col, col_levels):
        super().__init__(data , predictor_cols, target_col, col_levels)
        self.name = 'stats_ols'

    def fit(self, cols):
        formula = f'{self.target_col} ~ '
        predictors_pattern = [col if len(self.col_levels[col]) == 0 else f'C({col}, levels={self.col_levels[col]})' for col in cols]
        formula = f'{self.target_col} ~ ' + '+'.join(predictors_pattern)
        return ResultFit(ols(formula, self.data).fit(), cols)

    def score(self, result_fit):
        if result_fit is None:
            return float('inf')
        actual_value = self.data[self.target_col]
        predict_value = result_fit.predict(self.data)
        return mean_squared_log_error(actual_value, predict_value)

    def rss(self, result_fit):
        if result_fit is None:
            return self.tss
        else:
            return result_fit.result_.ssr

    def df_residual(self, result_fit):
        if result_fit is None:
            return self.nobs - 1
        else:
            return result_fit.result_.df_resid 

class ModelOLSLasso(ModelAbstract):
    def __init__(self, data , predictor_cols, target_col, col_levels):
        super().__init__(data , predictor_cols, target_col, col_levels)
        self.name = 'ols_lasso'

    def fit(self, cols):
        reg = LassoCV(cv=10, random_state=0).fit()
        return ResultFit(ols(formula, self.data).fit(), cols)

    def score(self, result_fit):
        if result_fit is None:
            return float('inf')
        actual_value = self.data[self.target_col]
        predict_value = result_fit.predict(self.data)
        return mean_squared_error(actual_value, predict_value)

    def rss(self, result_fit):
        if result_fit is None:
            return self.tss
        else:
            return result_fit.result_.ssr


        