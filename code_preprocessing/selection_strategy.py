import scipy.stats as stats
import numpy as np
import logging
import re
from pandas import DataFrame
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm
from sklearn.metrics import mean_squared_log_error, mean_squared_error, make_scorer
from code_preprocessing.cv_model import SMWrapper
from sklearn.model_selection import cross_val_score, ShuffleSplit

class MyAnova:
    def __init__(self):
        pass

    @staticmethod
    def anova_lm(model_, result_fit1, result_fit2, scale = None):
        test = 'F'
        pr_test = 'p_fvalue'
        names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
        table = DataFrame(np.zeros((2, 6)), columns=names)
        table["ssr"] = [model_.rss(mdl) for mdl in [result_fit1, result_fit2]]
        table["df_resid"] = [model_.df_residual(mdl) for mdl in [result_fit1, result_fit2]]

        if not scale: # assume biggest model is last
            scale = table['ssr'].iloc[-1] / table['df_resid'].iloc[-1]

        table.loc[table.index[1:], "df_diff"] = -np.diff(table["df_resid"].values)
        table["ss_diff"] = -table["ssr"].diff()
        if test == "F":
            table["F"] = table["ss_diff"] / table["df_diff"] / scale
            table[pr_test] = stats.f.sf(table["F"], table["df_diff"],
                                        table["df_resid"])
            table[pr_test][table['F'].isnull()] = np.nan
        return table

class ResultFit:
    def __init__(self, result_, cols):
        self.result_ = result_
        self.cols = cols

    def predict(self, X):
        return self.result_.predict(X)

class ModelAbstract:
    def __init__(self, data, predictor_cols, target_col, col_levels):
        self.data = data
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.col_levels = col_levels
        self.tss = np.var(data[target_col]) * len(data)
        self.nobs = len(data)
        self.history_cvscore  = {}


    def fit(self, predictors_col, target_col):
        pass

    def score(self, result_fit):
        pass

    def rss(self, result_fit):
        pass

    def df_residual(self, result_fit):
        pass

    def cv_score(self, result_fit):
        if result_fit is None:
            return self.tss
        if tuple(result_fit.cols) in self.history_cvscore:
            return self.history_cvscore[tuple(result_fit.cols)]
        cv_model = SMWrapper(self, result_fit.cols)
        cv = ShuffleSplit(n_splits=5, random_state=0)
        cv_score = -np.mean(cross_val_score(cv_model, self.data, self.data[self.target_col], cv=cv, scoring="neg_mean_squared_error"))
        self.history_cvscore[tuple(result_fit.cols)] = cv_score * self.nobs
        return cv_score * self.nobs

    def get_p_fvalue(self, result_fit1, result_fit2):
        return MyAnova.anova_lm(self, result_fit1, result_fit2)['p_fvalue'].iloc[1]

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
        


class SelectionFeatures:
    min_pvalue = 0.05
    def __init__(self, model_):
        self.model_ = model_;
        self.logger = logging.getLogger(f"select_features_with{model_.name}")

    def select_features(self, strategy):
        if strategy == "mix":
            return self.select_features_mix()
        elif strategy == "forward":
            return self.select_features_forward()
        elif strategy == "backward":
            return self.select_features_forward()


    def select_features_mix(self):
        current_list = []
        current_result_fit = self.fit_model(self.model_, current_list)
        candidate_cols = list(self.model_.predictor_cols)
        return self.feature_selection_step(self.mix_step, self.model_, current_list, current_result_fit, candidate_cols)

    def select_features_forward(self):
        current_list = []
        current_result_fit = self.fit_model(self.model_, current_list)
        candidate_cols = list(self.model_.predictor_cols)
        return self.feature_selection_step(self.forward_step, self.model_, current_list, current_result_fit, candidate_cols)

    def select_features_backward(self):
        current_list = []
        current_result_fit = self.fit_model(self.model_, current_list)
        candidate_cols = list(self.model_.predictor_cols)
        return self.feature_selection_step(self.backward_step, self.model_, current_list, current_result_fit, candidate_cols)


    def feature_selection_step(self, selection_func, stats_model, current_list, current_result_fit, candidate_cols):
        results = []
        cnt = 0
        while True:
            cnt += 1
            self.logger.info(f"Current itteration: {cnt}")
            select_cols, select_result_fit = selection_func(stats_model, current_list, current_result_fit, candidate_cols) 
            print("select_cols = ", select_cols)
            if select_cols is None:
                break
            results.append(ResultFit(select_result_fit, select_cols,))
            current_list, current_result_fit = select_cols, select_result_fit
        return results


    def mix_step(self, stats_model, current_list, current_result_fit, candidate_cols):
        if len(candidate_cols) > 0:
            forward_cols, forward_result_fit = self.forward_step(stats_model, current_list, current_result_fit, candidate_cols)
            if forward_cols is None:
                return None, None
            while True:
                backward_cols, backward_result_fit = self.backward_step(stats_model, forward_cols, forward_result_fit, candidate_cols)
                if backward_cols is None:
                    break
                else:
                    forward_cols, forward_result_fit = backward_cols, backward_result_fit
            return forward_cols, forward_result_fit
        else:
            return None, None

    def forward_step(self, stats_model, current_list, current_result_fit, candidate_cols):
        candidates = []
        for col in candidate_cols:
            test_list = current_list + [col]
            result_fit = self.fit_model(stats_model, test_list)
            if result_fit is not None:
                crit = self.get_criterion_value(stats_model, current_result_fit, result_fit)
                if self.satisfy_lowerber(crit):
                    candidates.append((col, result_fit))
        best_col, best_result_fit = self.select_best_from(stats_model, candidates)
        if self.first_result_better(stats_model, best_result_fit, current_result_fit):
            self.logger.info(f"Select feature {best_col} with imporve rss = {stats_model.score(best_result_fit)}")
            candidate_cols.remove(best_col)
            return current_list + [best_col], best_result_fit
        else:
            return None, None

    def backward_step(self, stats_model, current_list, current_result_fit, candidate_cols):
        result_list = list(current_list)
        is_remove = False
        candidates = []
        if len(current_list) == 1:
            return None, None
        for col in current_list:
            test_list = list(current_list)
            test_list.remove(col)
            result_fit_without_col = self.fit_model(stats_model, test_list)
            crit = self.get_criterion_value(stats_model, result_fit_without_col, current_result_fit)
            if not self.satisfy_lowerber(crit):
                self.logger.info(f"Remove feature {col} with imporve min_pvalue = {crit}")
                result_list.remove(col)
                is_remove = True
        if not is_remove:
            return None, None
        result_fit = self.fit_model(stats_model, result_list)
        self.logger.info(f"After remove, rss = {stats_model.score(result_fit)}")
        return result_list, result_fit

    def fit_model(self, stats_model, selected_cols):
        if len(selected_cols) == 0:
            return None
        model = stats_model.fit(selected_cols)
        return model

    def select_best_from(self, stats_model, candidates):
        select_col, select_result_fit = candidates[0]
        for col, result_fit in candidates[1:]:
            if self.first_result_better(stats_model, result_fit, select_result_fit):
                select_col, select_result_fit = col, result_fit
        return select_col, select_result_fit

    def first_result_better(self, stats_model, result1, result2):
        return stats_model.score(result1) < stats_model.score(result2)

    def get_criterion_value(self, stats_model, result_fit_without_col, result_fit):
        return stats_model.get_p_fvalue(result_fit_without_col, result_fit)

    def satisfy_lowerber(self, criterion_value):
        return criterion_value < SelectionFeatures.min_pvalue