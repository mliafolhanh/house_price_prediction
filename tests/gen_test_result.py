import pandas as pd
import pickle
from code_preprocessing.preprocess import *
from code_preprocessing.selection_strategy import *
from code_preprocessing.logging_utils import setup_default_logging
from code_preprocessing.cv_model import *
from code_preprocessing.datasets import read_data

# def read_data(path_file):
#     """This function will be handled in the dataset module."""
#     train_pd = pd.read_csv(path_file)
#     train_pd = train_pd.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThirdSsnPorch"})
#     return train_pd

def preprocess(train_pd):
    imputer = DataFrameImputer()
    return imputer.fit_transform(train_pd)

test_pd = read_data("test.csv")
test_pd = preprocess(test_pd)
selection = pickle.load(open("models/ols_model.pkl", "rb"))

 #|-1.674
# final_model.predict(test_pd[final_cols])