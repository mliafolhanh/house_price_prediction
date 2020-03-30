import scipy.stats as stats
import numpy as np
import logging
import re
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm
from sklearn.metrics import mean_squared_log_error

class ModelAbstract:
    def __init__(self, type_model):
        pass

    def fit(self, predictors_col, target_col):
        pass

    def score(self, result_fit):
        pass

    def getPValue(self, result_fit, col):
        pass

class ModelOLSStats(ModelAbstract):
    def __init__(self, data, predictor_cols, target_col, col_levels):
        self.data = data
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.col_levels = col_levels

    def fit(self, predictors_col):
        formula = f'{self.target_col} ~ '
        predictors_pattern = [col if len(self.col_levels[col]) == 0 else f'C({col}, levels={self.col_levels[col]})' for col in predictors_col]
        formula = f'{self.target_col} ~ ' + '+'.join(predictors_pattern)
        return ols(formula, self.data).fit()

    def score(self, result_fit):
        if result_fit is None:
            return float('inf')
        return result_fit.ssr

    def score(self, result_fit):
        if result_fit is None:
            return float('inf')
        actual_value = self.data[self.target_col]
        predict_value = result_fit.predict(self.data)
        return mean_squared_log_error(actual_value, predict_value)

    def getPValue(self, result_fit1, result_fit2):
        anova_table = anova_lm(result_fit1, result_fit2)
        return anova_table["Pr(>F)"][1]


class SelectionFeatures:
    min_pvalue = 0.01
    def __init__(self, type_model):
        self.type_model = type_model
        self.logger = logging.getLogger(f"select_features_with{self.type_model}")

    def select_features_mix(self, data, predictor_cols, target_col, col_levels):
        stats_model = self.type_model(data, predictor_cols, target_col, col_levels)
        current_list = []
        current_result_fit = self.fit_model(stats_model, current_list)
        candidate_cols = list(predictor_cols)
        return self.feature_selection_step(self.mix_step, stats_model, current_list, current_result_fit, candidate_cols)

    def select_features_forward(self, data, predictor_cols, target_col, col_levels):
        stats_model = self.type_model(data, predictor_cols, target_col, col_levels)
        current_list = []
        current_result_fit = self.fit_model(stats_model, current_list)
        candidate_cols = list(predictor_cols)
        return self.feature_selection_step(self.forward_step, stats_model, current_list, current_result_fit, candidate_cols)

    def select_features_backward(self, data, predictor_cols, target_col, col_levels):
        stats_model = self.type_model(data, predictor_cols, target_col, col_levels)
        current_list = predictor_cols
        current_result_fit = self.fit_model(stats_model, current_list)
        print(current_list)
        candidate_cols = list(predictor_cols)
        return self.feature_selection_step(self.backward_step, stats_model, current_list, current_result_fit, candidate_cols)


    def feature_selection_step(self, selection_func, stats_model, current_list, current_result_fit, candidate_cols):
        results = []
        cnt = 0
        while True:
            cnt += 1
            select_cols, select_result_fit = selection_func(stats_model, current_list, current_result_fit, candidate_cols) 
            print("select_cols = ", select_cols)
            if select_cols is None:
                break
            results.append((select_cols, select_result_fit))
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
                candidates.append((col, result_fit))
        best_col, best_result_fit = self.select_best_from(stats_model, candidates)
        if self.first_result_better(stats_model, best_result_fit, current_result_fit):
            self.logger.info(f"Select feature {best_col} with imporve rss = {stats_model.score(best_result_fit)}")
            candidate_cols.remove(best_col)
            return current_list + [best_col], best_result_fit
        else:
            return None, None

    # def backward_step(self, stats_model, current_list, current_result_fit, candidate_cols):
    #     result_list = list(current_list)
    #     is_remove = False
    #     candidates = []
    #     for col in current_list:
    #         test_list = list(current_list)
    #         test_list.remove(col)
    #         result_fit = self.fit_model(stats_model, test_list)
    #         if result_fit is not None:
    #             candidates.append((col, result_fit))
    #     if len(candidates) == 0:
    #         return None, None
    #     best_col, best_result_fit = self.select_best_from(stats_model, candidates)
    #     print("best_col = " , best_col)
    #     if self.first_result_better(stats_model, best_result_fit, current_result_fit):
    #         self.logger.info(f"Remove feature {best_col} with imporve rss = {stats_model.score(best_result_fit)}")
    #         current_list.remove(best_col)
    #         return current_list, best_result_fit
    #     else:
    #         return None, None

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
        # except:
        #     model = None
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
        return stats_model.getPValue(result_fit_without_col, result_fit)

    def satisfy_lowerber(self, criterion_value):
        return criterion_value < SelectionFeatures.min_pvalue