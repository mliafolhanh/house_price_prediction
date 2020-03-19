import scipy.stats as stats
import numpy as np
from statsmodels.formula.api import ols
class ModelAbstract:
    def __init__(self, type_model):
        pass

    def fit(self, data, predictors_col, target_cols):
        pass

class ModelOLSStats(ModelAbstract):
    def __init__(self, data):
        self.data = data
        self.is_category = self.find_list_category_columns()

    def find_list_category_columns(self, data):
        all_cols = data.columns
        uniq_count = data.nunique()
        category_crit = int(np.quantile(uniq_count, 0.75))
        is_category = uniq_count <= category_crit

    def fit(self, data, predictors_col, target_col):
        formula = f'{target_col} ~ '
        predictors_pattern = [col if self.is_category[col] else f'C({col})' for col in predictors_col]
        formula = f'{target_col} ~ ' + '+'.join(predictors_col)
        return ols(formula, data).fit()
        

class SelectionCategoryFeatures:
    pvalue = 0.05
    def __init__(self, type_model):
        self.type_model = type_model

    def forward_step(self, data, current_list, cols):
        candidates = []
        for col in cols:
            test_list = current_list + [col]
            model = self.create_model(data, test_list)
            candidates.append((col, model))
        best_col, best_model = self.select_best_from(candidates)
        return best_col, best_model

    def backward_step(self, current_list, model):
        remove_candidates = list(current_list)
        for col in remove_candidates:
            criterion_value = self.get_criterion_value(model, col)
            if not self.satisfy_lowerber(criterion_value):
                current_list.remove(col)

    def create_model(self, data, selected_cols):
        model = self.type_model(data)
        return self.fit()

            

        

