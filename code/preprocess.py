from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
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
        is_category = uniq_count <= category_crit
        return is_category

    def fitNull(self, X, y = None):
        is_category = self.find_list_category_columns(X)
        self.fill = pd.Series([X[c].value_counts().index[0] if is_category[c] else X[c].mean() 
                             for c in X.columns], index=X.columns)

    def fitRemove(self, X, y = None):
        percent_null = X.isnull().sum() / X.shape[0]
        self.remove_cols = percent_null[percent_null > 0.75].index

    def fit(self, X, y = None):
        self.fitRemove(X, y)
        self.fitNull(X, y)
        return self
        
    def transform(self, X, y = None):
        X = X.drop(self.remove_cols, axis=1)
        return X.fillna(self.fill)



