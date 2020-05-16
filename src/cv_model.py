from sklearn.base import BaseEstimator, RegressorMixin
import copy

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_, select_cols):
        self.model_ = model_
        self.select_cols = select_cols

    def fit(self, X, y = None):
        self.new_model_ = copy.deepcopy(self.model_)
        self.new_model_.data = X #change the data
        self.result_fit = self.new_model_.fit(self.select_cols)

    def predict(self, X):
        return self.result_fit.predict(X[self.select_cols])