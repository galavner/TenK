# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import time
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn import preprocessing

# from LabData.DataLoaders.Loader import Loader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataMergers.DataMerger import DataMerger
from LabData.DataPredictors.DataPredictors import DataPredictor
from LabData.DataPredictors.PredictorParams import PredictorParams
from LabUtils.addloglevels import sethandlers
from LabUtils.Utils import mkdirifnotexists
from LabQueue.qp import qp, fakeqp
from LabData.DataAnalyses.TenK_Trajectories.archive.defs import config, DATA_SETS, MAIN_PREDS_DIR, SCREEN_BASELINE


import GaussianMapping
import LGBM
#

os.chdir('/net/mraid08/export/genie/LabData/Analyses/galavner/Predictions/')
sethandlers()


def get_y_loaders():
    body = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                         norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5, 'remove_sigmas': 8}).df
    blood = BloodTestsLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                        norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5, 'remove_sigmas': 8}).df

    body = body.droplevel(level=1)
    blood = blood.droplevel(level=1)

    df = pd.merge(blood, body, right_index=True, left_index=True)


def alter_categories(df: pd.DataFrame):
    res = df[[c for c
        in list(df)
        if (len(df[c].unique()) - 1 > 10 or len(df[c].unique()) - 1 == 2)]]
    # res = pd.get_dummies(res, dummy_na=False, drop_first=True)
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


# def remove_outliers(df : pd.DataFrame):
# #     remove all rows with features valued outside x /sigma
#     return df[(np.abs(stats.zscore(df, axis = 1, nan_policy = 'omit')) < 5).all(axis = 1)]


# def impute_data(X_train, X_test):
#     train_columns = X_train.columns
#     test_columns = X_test.columns
#     imputer = KNNImputer(n_neighbors = 5)
#     train_data = imputer.fit_transform(X_train)
#     test_data = imputer.transform(X_test)
#     train_imputed = pd.DataFrame(train_data, columns=train_columns)
#     test_imputed = pd.DataFrame(test_data, test_columns)
#     return train_imputed, test_imputed



def normalize(X: pd.DataFrame):
    scaler = preprocessing.StandardScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return X_normalized
    # scaler.transform(X_test)

def XY_gen_f(x_df, y_col):
    X, Y = DataMerger(x_df, y_col).get_xy(y_col=y_col, inexact_index='Date')
    return f"BML {y_col}", X, Y, None, None


def Xy_gen_fs():

    with qp(jobname='Gal_LGBM', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=200,
                       _mem_def=2) as q:
        q.startpermanentrun()

        sm = SerumMetabolomicsLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                                norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5, 'remove_sigmas': 8})
        sm_df = sm.df
        # ReIndex s.t. the index is now RegistrationCode, and not SerumName, and only the first value
        sm_df = sm_df.set_index(sm.df_metadata.RegistrationCode.drop_duplicates())

        blood = BloodTestsLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                            norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5, 'remove_sigmas': 8})
        blood_df = blood.df
        x = sm_df
        y_df = blood_df

        x = alter_categories(x)
        y_df = alter_categories(y_df)

        y_df.index = y_df.index.get_level_values('RegistrationCode')
        # for y_df in df.columns:

        tkttores = {}
        for i, y_col in enumerate(['bt__cpk']):

            y = y_df[y_col].dropna()
            x = x.reindex(y.index).dropna(how='all')
            y = y.reindex(x.index)
            # df_filtered, num_of_patients = get_relevant_patients_per_outcome(df, y_col)
            # df_filtered = remove_nan_columns(df_filtered)



            # X = df_filtered.loc[:, df_filtered.columns != y_col]
            # Y = df_filtered.loc[:, y_col]

            if x.shape[0] < 500:
                continue



            # X = normalize(X)

            # LGBM.LGBMPredict(x, y)


            tkttores[(i, y_col)] = q.method(LGBM.LGBMPredict, (x, y))

            # time.sleep(120)




        # X.to_csv(os.path.join(db_path, f'features {y_df}.csv'))
        # Y.to_csv(os.path.join(db_path, f'results {y_df}.csv'))

        # yield lambda: XY_gen_f(X, y_df)


def main():
    Xy_gen_fs()
    print('Done')
    # # sethandlers(file_dir=config.log_dir)
    # y_col = 'waist'
    # # analysis_dir = '/net/mraid08/export/genie/LabData/Tom'
    # analysis_dir = '/net/mraid08/export/genie/LabData/Analyses/galavner'
    # res = DataPredictor(PredictorParams('--predictor_type XGBRegressor --num_cv_folds 3')).predict_multi(Xy_gen_fs,work_dir=analysis_dir)
    # print(res)

if __name__ == '__main__':
    main()


