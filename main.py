# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from LabData.DataLoaders.Loader import Loader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader


def load_filter_data():
    df = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first').df # Get only first appointment of 10K
    df = df.select_dtype(include = np.number) # Take only numerical values for now



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print('Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_relevant_patient(df, y):
    # Get DF and a feature, returns the DF with only relevant patient with non-NAN entries in the feature column
    relevant_patients = df[y].notna()#consider only non-NAN patients
    number_of_patients = relevant_patients.sum()
    x = df.iloc[relevant_patients,:]
    return x

def remove_nan_columns(x : pd.DataFrame):
#     Get the features matrix X and removes all columns with large amount of Nans
#     percent_of_nan = 0.66
#     columns_to_remove = x.columns
#     x = x.drop(())


def remove_nan_rows(x,y):
#     remove rows with Nan values in some columns that have not been filtered out by now

def remove_outliers(x,y):
#     remove all rows with features valued outside x /sigma




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    X = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first')
    x_raw = X.df
    # x_noHT = x_raw.drop('on_hormone_therapy', axis=1)
    x_numerics_only = x_raw.select_dtypes(include = np.number)
    x_numerics_only = x_numerics_only.dropna(axis = 1, how = 'all')
    # x_numerics_only_filtted = x_numerics_only[(np.abs(stats.zscore(x_numerics_only, nan_policy = 'omit')) < 3).all(axis = 1)]
    # for i in x_numerics_only.columns:
    #     print(i)
    #     print(np.abs(stats.zscore(x_numerics_only[i], axis = 0, nan_policy = 'omit')) < 10)
    # # x_ol = x_raw[(np.abs(stats.zscore(x_raw)) < 3).all(axis=1)]
    # print("here")
    print("MB Change!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
