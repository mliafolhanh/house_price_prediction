from src.models import *

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
                #crit = self.get_criterion_value(stats_model, current_result_fit, result_fit)
                #if self.satisfy_lowerber(crit):
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