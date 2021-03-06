# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn import preprocessing

# from LabData.DataLoaders.Loader import Loader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataMergers.DataMerger import DataMerger
from LabData.DataPredictors.DataPredictors import DataPredictor
from LabData.DataPredictors.PredictorParams import PredictorParams
from LabUtils.addloglevels import sethandlers
from LabUtils.Utils import mkdirifnotexists
from LabData.DataAnalyses.TenK_Trajectories.archive.defs import config, DATA_SETS, MAIN_PREDS_DIR, SCREEN_BASELINE

#
# def load_filter_data():
#     df = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first').df # Get only first appointment of 10K
#     df = df.select_dtype(include = np.number) # Take only numerical values for now
#


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print('Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




def alter_categories(df: pd.DataFrame):
    res = df[[c for c
        in list(df)
        if (len(df[c].unique()) - 1 > 10 or len(df[c].unique()) - 1 == 2)]]
    res = pd.get_dummies(res, dummy_na=False, drop_first=True)

    # nuniques = df.nunique(axis = 0, dropna = True)
    # for feature in nuniques.items():
    #     num_uniques = feature[1]
    #     if num_uniques < 2:
    #         df.drop(feature[0])
    #         continue
    #     elif num_uniques == 2:
    #         df[feature[0]] = pd.get_dummies(df[feature[0]], dummy_na=False, drop_first=True)
    #     elif num_uniques < 10:
    #         df.drop(df[feature[0]], axis = 1)
    #         continue
    return res




def get_relevant_patients_per_outcome(df : pd.DataFrame , y):
    # Get DF and a feature, returns the DF with only relevant patient with non-NAN entries in the feature column
    relevant_patients = df[y].notna()#consider only non-NAN patients
    number_of_patients = relevant_patients.sum() #if this is zero, we should skip this feature as y
    x = df[relevant_patients]
    return x, number_of_patients


def remove_nan_columns(df : pd.DataFrame):
#     Get the features matrix X and removes all columns with large amount of Nans
    percent_of_non_nan = 0.2
    x = df.dropna(axis = 1, thresh = np.ceil(percent_of_non_nan * len(df)))
    return x


def remove_nan_rows(df : pd.DataFrame ):
#     remove rows with Nan values in some columns that have not been filtered out by now
#     patients_to_remove = x.notna().sum
    x = df.dropna(axis = 'index', how = 'any')
    return x


def remove_outliers(df : pd.DataFrame):
#     remove all rows with features valued outside x /sigma
    return df[(np.abs(stats.zscore(df, axis = 1, nan_policy = 'omit')) < 5).all(axis = 1)]


def impute_data(X_train, X_test):
    imputer = KNNImputer(n_neighbors = 5)
    imputer.fit_transform(X_train)
    imputer.transform(X_test)


def normalize (X):
    scaler = preprocessing.StandardScaler()
    scaler.fit_transform(X)
    # scaler.transform(X_test)

def XY_gen_f(x_df, y_col):
    X , Y = DataMerger(x_df, y_col).get_xy(y_col = y_col, inexact_index='Date')
    return f"BML {y_col}", X, Y, None, None


# Press the green button in the gutter to run the script.
def Xy_gen_fs():
    # Load only 10K cohort with only first appointment patients
    df = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present = 500).df

    df = alter_categories(df)
    for y in df.columns:
        df_filtered, num_of_patients = get_relevant_patients_per_outcome(df, y)
        df_filtered = remove_nan_columns(df_filtered)
        df_filtered = remove_outliers(df_filtered)
        # it shouldn't matter if it's done before train-test split as it randomly assigns them to each group

        X = df_filtered.loc[:, df_filtered.columns != y]
        Y = df_filtered.loc[:, y]

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        # Before imputing the data, it might be a nice to see if they are missing at random or not
        # impute_data(X_train, X_test)


        normalize(X)

        yield lambda: XY_gen_f(X, y)


def main():
    # sethandlers(file_dir=config.log_dir)
    sethandlers()
    y_col = 'waist'
    # analysis_dir = '/net/mraid08/export/genie/LabData/Tom'
    analysis_dir = '/net/mraid08/export/genie/LabData/Analyses/galavner'
    res = DataPredictor(PredictorParams('--predictor_type XGBRegressor --num_cv_folds 3')).predict_multi(Xy_gen_fs,work_dir=analysis_dir)
    print(res)

if __name__ == '__main__':
    main()




    #
    # print_hi('PyCharm')
    # X = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first')
    # x_raw = X.df
    # # x_noHT = x_raw.drop('on_hormone_therapy', axis=1)
    # x_numerics_only = x_raw.select_dtypes(include = np.number)
    # x_numerics_only = x_numerics_only.dropna(axis = 1, how = 'all')
    # # x_numerics_only_filtted = x_numerics_only[(np.abs(stats.zscore(x_numerics_only, nan_policy = 'omit')) < 3).all(axis = 1)]
    # # for i in x_numerics_only.columns:
    # #     print(i)
    # #     print(np.abs(stats.zscore(x_numerics_only[i], axis = 0, nan_policy = 'omit')) < 10)
    # # # x_ol = x_raw[(np.abs(stats.zscore(x_raw)) < 3).all(axis=1)]
    # print("here")
    # print("MB Change!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
