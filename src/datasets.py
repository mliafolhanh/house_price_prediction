# This module is to handle sample datasets provided by the package (similar to sklearn.datasets)

import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(__file__), '..') + '/data'

def read_data(file_name):
    train_pd = pd.read_csv(os.path.join(data_dir, file_name))
    train_pd = train_pd.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThirdSsnPorch"})
    train_pd = train_pd.set_index("Id")
    return train_pd