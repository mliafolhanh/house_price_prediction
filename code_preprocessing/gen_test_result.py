import pandas as pd
import pickle
from preprocess import *
from selection_strategy import *
from logging_utils import setup_default_logging
from cv_model import *


def read_data(path_file):
    train_pd = pd.read_csv(path_file)
    train_pd = train_pd.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThirdSsnPorch"})
    return train_pd
def preprocess(train_pd):
    imputer = DataFrameImputer()
    return imputer.fit_transform(train_pd)

test_pd = read_data("data/test.csv")
test_pd = preprocess(test_pd)
selection = pickle.load(open("models/ols_model.pkl", "rb"))

 #|-1.674
final_model.predict(test_pd[final_cols])