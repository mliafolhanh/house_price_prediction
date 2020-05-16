from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
max_ratio_null = 0.75
max_ratio_cat = 0.95
def find_list_category_columns(X):
    uniq_count = X.nunique()
    category_crit = 0
    is_category = (X.dtypes == "object") | (uniq_count <= category_crit)
    return is_category

def find_col_levels(X, is_category):
    col_levels = {}
    for col in X.columns:
        if is_category[col]:
            col_levels[col] = sorted(list(X[col].value_counts().index))
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
        self.col_levels = find_col_levels(X, self.is_category)
        self.fill = pd.Series([X[c].value_counts().index[0] if self.is_category[c] else X[c].mean() 
                             for c in X.columns], index=X.columns)
        self.col_exceptions = ['Alley', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                               'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        #special case
        self.col_exceptions = list(set(self.col_exceptions).intersection(set(X.columns)))
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
        X_new = X_new.fillna(self.fill)
        self.col_levels = find_col_levels(X_new, self.is_category)
        return X_new

class RemovelColsImputer(TransformerMixin):

    def __init__(self, is_category = None):
        self.is_category = is_category

    def fit(self, X, y = None):
        min_fpvalue = 0.08
        if self.is_category is None:
            self.is_category = find_list_category_columns(X)
        self.remove_categorys = []
        for col in X.columns:
            if self.is_category[col]:
                if (X[col].value_counts().iloc[0] / X.shape[0]) > max_ratio_cat:
                    self.remove_categorys.append(col)
        return self

    def transform(self, X, y = None):
        X_new = X.drop(self.remove_categorys, axis=1)
        return X_new

class MyOneHotEncode(TransformerMixin):

    def fit(self, X, y = None):
        self.is_category = find_list_category_columns(X)
        self.col_levels = find_col_levels(X, self.is_category)
        self.one_hot = {}
        for col in self.is_category.index:
            if self.is_category[col]:
                enc = OneHotEncoder([self.col_levels[col]], drop="first", sparse=False).fit(X[[col]])
                self.one_hot[col] = enc
        return self

    def transform(self, X, y = None):
        X_result = pd.DataFrame(index=X.index)
        for col in self.is_category.index:
            if self.is_category[col]:
                result_transform = self.one_hot[col].transform(X[[col]])
                pd_transform = pd.DataFrame(result_transform, columns = [f"{col}_{value}" for value in self.col_levels[col][1:]], index = X.index)
                X_result = pd.concat([X_result, pd_transform], axis = 1)
            else:
                X_result = pd.concat([X_result, X[[col]]], axis = 1)
        return X_result

class MyStandardScaler(TransformerMixin):

    def fit(self, X, y = None):
        self.is_category = find_list_category_columns(X)
        #self.quant_cols = [col for col in self.is_category.index if not self.is_category[col]]
        self.scaler = StandardScaler().fit(X)
        return self

    def transform(self, X, y = None):
        X_result = X
        X_result[X.columns] = self.scaler.transform(X)
        return X_result



        




