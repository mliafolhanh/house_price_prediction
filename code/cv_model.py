from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, predictor_cols, target_col, select_cols):
        self.model_class = model_class
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.select_cols = select_cols

    def fit(self, X, y = None):
        self.model_ = self.model_class(X, self.predictor_cols, self.target_col)
        self.results_ = self.model_.fit(select_cols)

    def predict(self, X):
        return self.results_.predict(X[self.select_cols])