from sklearn.base import TransformerMixin
from code_preprocessing.rank_columns import RankColumns
import pandas as pd
import numpy as np
max_ratio_null = 0.75
max_ratio_cat = 0.95
def find_list_category_columns(X):
    uniq_count = X.nunique()
    category_crit = int(np.quantile(uniq_count, 0.75))
    is_category = (X.dtypes == "object") | (uniq_count <= category_crit)
    return is_category

def find_col_levels(X):
    col_levels = {}
    is_category = find_list_category_columns(X)
    for col in is_category.index:
        if is_category[col]:
            col_levels[col] = list(X[col].value_counts().index)
        else:
            col_levels[col] = []
    return col_levels

class NullImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y = None):
        self.is_category = find_list_category_columns(X)
        self.col_levels = find_col_levels(X)
        self.fill = pd.Series([X[c].value_counts().index[0] if self.is_category[c] else X[c].mean() 
                             for c in X.columns], index=X.columns)
        self.col_exceptions = ['Alley', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                               'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        #special case
        self.fill.loc[self.col_exceptions] = 'None'
        return self
        
    def transform(self, X, y = None):
        X_new = X
        for col in self.is_category.index:
            if self.is_category[col]:
                value_counts = X[col].value_counts()
                for value in value_counts.index:
                    if value not in self.col_levels[col]:
                        X_new.replace({col: value}, value=np.NaN, inplace=True)    
        return X_new.fillna(self.fill)

class RemovelColsImputer(TransformerMixin):

    def fit(self, X, y = None):
        min_fpvalue = 0.08
        self.is_category = find_list_category_columns(X)
        self.remove_categorys = []
        for col in X.columns:
            if self.is_category[col]:
                if (X[col].value_counts().iloc[0] / X.shape[0]) > max_ratio_cat:
                    self.remove_categorys.append(col)
        # column_ranker = RankColumns()
        # tmp_remove_categorys = list(self.remove_categorys)
        # for col in tmp_remove_categorys:
        #     if column_ranker.cal_fpvalue_special_col(X, col) < min_fpvalue:
        #         self.remove_categorys.remove(col)
        return self

    def transform(self, X, y = None):
        X_new = X.drop(self.remove_categorys, axis=1)
        return X_new




