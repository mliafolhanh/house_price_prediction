from code_preprocessing.preprocess import DataFrameImputer
from statsmodels.api.formula import ols
class RankColumns:
    max_ratio_cat= 0.95
    def __init__(self):
        pass

    def rank_columns(self, X, y):
        imputer = DataFrameImputer().fit(X)
        category_cols = imputer.is_category[imputer.is_category == True].index
        quanlitative_cols = imputer.is_category[imputer.is_category == False].index
        rank_category = self.rank_category_columns(X, y, category_cols)
        rank_quanlitative = self.rank_quanlitative_columns(X, y, quanlitative_cols)

    def rank_category_columns(self, X, y, category_cols):
        
        for col in category_cols:
            if (X[col].value_counts[0] / X.shape[0]) > max_ratio_cat:

                




