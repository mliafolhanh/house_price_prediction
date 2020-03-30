from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
max_ratio_null = 0.75
max_ratio_cat = 0.95
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def find_list_category_columns(self, X):
        uniq_count = X.nunique()
        category_crit = int(np.quantile(uniq_count, 0.75))
        is_category = (X.dtypes == "object") | (uniq_count <= category_crit)
        return is_category

    def find_col_levels(self, X):
        col_levels = {}
        is_category = self.find_list_category_columns(X)
        for col in is_category.index:
            if is_category[col]:
                col_levels[col] = list(X[col].value_counts().index)
            else:
                col_levels[col] = []
        return col_levels

    def fitNull(self, X, y = None):
        self.is_category = self.find_list_category_columns(X)
        self.col_levels = self.find_col_levels(X)
        self.fill = pd.Series([X[c].value_counts().index[0] if self.is_category[c] else X[c].mean() 
                             for c in X.columns], index=X.columns)
        #special case
        self.fill.loc['Alley'] = 'None'

    def fitRemove(self, X, y = None):
        self.is_category = self.find_list_category_columns(X)
        percent_null = X.isnull().sum() / X.shape[0]
        self.remove_cols = list(percent_null[percent_null > max_ratio_null].index)
        for col in X.columns:
            if self.is_category[col] and col not in self.remove_cols:
                if (X[col].value_counts().iloc[0] / X.shape[0]) > max_ratio_cat:
                    self.remove_cols.append(col)
        #special case
        self.remove_cols.remove('Alley')

    def fit(self, X, y = None):
        self.fitRemove(X, y)
        self.fitNull(X, y)
        return self
        
    def transform(self, X, y = None):
        X_new = X.drop(self.remove_cols, axis=1)
        for col in self.is_category.index:
            if self.is_category[col]:
                value_counts = X[col].value_counts()
                for value in value_counts.index:
                    if value not in self.col_levels[col]:
                        X_new.replace({col: value}, value=np.NaN, inplace=True)    
        return X_new.fillna(self.fill)



