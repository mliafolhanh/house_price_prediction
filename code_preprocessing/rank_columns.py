from statsmodels.formula.api import ols
class RankColumns:
    max_ratio_cat= 0.95
    def __init__(self):
        pass
    def getOLS(self, data, predictor):
        result = ols(f"SalePrice ~ C({predictor})", data=data).fit()
        return result

    def cal_fpvalue_special_col(self, train_pd, col_name):
        if train_pd[col_name].nunique() == 2:
            return float('inf')
        mode = train_pd[col_name].value_counts().index[0]
        new_train_pd = train_pd[train_pd[col_name] != mode]
        result_ols = self.getOLS(new_train_pd, col_name)
        return result_ols.f_pvalue

    def rank_columns(self, X, y):
        imputer = DataFrameImputer().fit(X)
        category_cols = imputer.is_category[imputer.is_category == True].index
        quanlitative_cols = imputer.is_category[imputer.is_category == False].index
        rank_category = self.rank_category_columns(X, y, category_cols)
        rank_quanlitative = self.rank_quanlitative_columns(X, y, quanlitative_cols)

    # def rank_category_columns(self, X, y, category_cols):

    #     for col in category_cols:
    #         if (X[col].value_counts[0] / X.shape[0]) > max_ratio_cat:

                




