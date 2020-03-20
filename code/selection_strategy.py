import scipy.stats as stats
import numpy as np
import logging
from statsmodels.formula.api import ols
import re
class ModelAbstract:
    def __init__(self, type_model):
        pass

    def fit(self, predictors_col, target_col):
        pass

    def getRSS(self, result_fit):
        pass

    def getPValue(self, result_fit, col):
        pass

class ModelOLSStats(ModelAbstract):
    def __init__(self, data, predictor_cols, target_col):
        self.data = data
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.is_category = self.find_list_category_columns()

    def find_list_category_columns(self):
        uniq_count = self.data.nunique()
        category_crit = int(np.quantile(uniq_count, 0.75))
        is_category = uniq_count <= category_crit
        return is_category

    def fit(self, predictors_col):
        formula = f'{self.target_col} ~ '
        predictors_pattern = [col if self.is_category[col] else f'C({col})' for col in predictors_col]
        formula = f'{self.target_col} ~ ' + '+'.join(predictors_col)
        return ols(formula, self.data).fit()

    def getRSS(self, result_fit):
        if result_fit is None:
            return float('inf')
        return result_fit.mse_resid

    def getPValue(self, result_fit, col):
        pattern = f'{col}'
        pvalues = result_fit.pvalues
        sumpvalue = 0.0
        npvalue = 0.0
        for col in pvalues.index:
            if col.find(pattern) >= 0:
                sumpvalue += pvalues[col]
                npvalue += 1.0
        return sumpvalue / npvalue

class SelectionFeatures:
    min_pvalue = 0.05
    def __init__(self, type_model):
        self.type_model = type_model
        self.logger = logging.getLogger(f"select_features_with{self.type_model}")

    def select_feature_mix(self, data, predictor_cols, target_col):
        stats_model = self.type_model(data, predictor_cols, target_col)
        current_list = []
        current_result_fit = self.fit_model(stats_model, current_list)
        candidate_cols = list(predictor_cols)
        return self.feature_selection_step(self.mix_step, stats_model, current_list, current_result_fit, candidate_cols)

    def select_feature_forward(self, data, predictor_cols, target_col):
        stats_model = self.type_model(data, predictor_cols, target_col)
        current_list = []
        current_result_fit = self.fit_model(stats_model, current_list)
        candidate_cols = list(predictor_cols)
        return self.feature_selection_step(self.forward_step, stats_model, current_list, current_result_fit, candidate_cols)

    def select_features_backward(self, data, predictor_cols, target_col):
        stats_model = self.type_model(data, predictor_cols, target_col)
        current_list = predictor_cols
        current_result_fit = self.fit_model(stats_model, current_list)
        candidate_cols = list(predictor_cols)
        return self.feature_selection_step(self.backward_step, stats_model, current_list, current_result_fit, candidate_cols)


    def feature_selection_step(self, selection_func, stats_model, current_list, current_result_fit, candidate_cols):
        while True:
            select_cols, select_result_fit = selection_func(stats_model, current_list, current_result_fit, candidate_cols) 
            if select_cols is None:
                break
            current_list, current_result_fit = select_cols, select_result_fit
        return current_list, current_result_fit


    def mix_step(self, stats_model, current_list, current_result_fit, candidate_cols):
        if len(candidate_cols) > 0:
            forward_cols, forward_result_fit = self.forward_step(stats_model, current_list, current_result_fit, candidate_cols)
            if forward_cols is None:
                return None, None
            backward_cols, backward_result_fit = self.backward_step(stats_model, forward_cols, forward_result_fit, candidate_cols)
            if backward_cols is None:
                return forward_cols, forward_result_fit
            else:
                return backward_cols, backward_result_fit

    def forward_step(self, stats_model, current_list, current_result_fit, candidate_cols):
        candidates = []
        for col in candidate_cols:
            test_list = current_list + [col]
            result_fit = self.fit_model(stats_model, test_list)
            if result_fit is not None:
                candidates.append((col, result_fit))
        best_col, best_result_fit = self.select_best_from(stats_model, candidates)
        if self.first_result_better(stats_model, best_result_fit, current_result_fit):
            self.logger.info(f"Select feature {best_col} with imporve rss = {stats_model.getRSS(best_result_fit)}")
            candidate_cols.remove(best_col)
            return current_list + [best_col], best_result_fit
        else:
            return None, None

    def backward_step(self, stats_model, current_list, current_result_fit, candidate_cols):
        result_list = list(current_list)
        is_remove = False
        for col in current_list:
            criterion_value = self.get_criterion_value(stats_model, current_result_fit, col)
            if not self.satisfy_lowerber(criterion_value):
                result_list.remove(col)
                self.logger.info(f"Remove feature {col} with avg p_valuue = {criterion_value}")
                is_remove = True
        if is_remove:
            return result_list, self.fit_model(stats_model, result_list)
        else:
            return None, None

    def fit_model(self, stats_model, selected_cols):
        if len(selected_cols) == 0:
            return None
        try:
            model = stats_model.fit(selected_cols)
        except:
            model = None
        return model

    def select_best_from(self, stats_model, candidates):
        select_col, select_result_fit = candidates[0]
        for col, result_fit in candidates[1:]:
            if self.first_result_better(stats_model, result_fit, select_result_fit):
                select_col, select_result_fit = col, result_fit
        return select_col, select_result_fit

    def first_result_better(self, stats_model, result1, result2):
        return stats_model.getRSS(result1) < stats_model.getRSS(result2)

    def get_criterion_value(self, stats_model, result_fit, col):
        return stats_model.getPValue(result_fit, col)

    def satisfy_lowerber(self, criterion_value):
        return criterion_value < SelectionFeatures.min_pvalue


            

        

