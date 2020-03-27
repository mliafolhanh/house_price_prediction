from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, predictor_cols, target_col, select_cols, col_levels):
        self.model_class = model_class
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.select_cols = select_cols
        self.col_levels = col_levels

    def fit(self, X, y = None):
        self.model_ = self.model_class(X, self.predictor_cols, self.target_col, self.col_levels)
        self.results_ = self.model_.fit(self.select_cols)
        self.mean = X[self.target_col].mean()

    def predict(self, X):
        return self.results_.predict(X[self.select_cols])