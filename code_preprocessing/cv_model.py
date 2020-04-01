from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_, select_cols):
        self.model_ = model_
        self.select_cols = select_cols

    def fit(self, X, y = None):
        self.results_ = self.model_.fit(self.select_cols)

    def predict(self, X):
        return self.results_.predict(X[self.select_cols])