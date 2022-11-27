import os
from os.path import join

import pickle
import glob
import pandas as pd

BASE_PATH = os.path.join('/net', 'mraid08', 'export', 'genie', 'LabData', 'Analyses', 'galavner')

PREDICTIONS_PATH = os.path.join(BASE_PATH, 'Predictions')
DB_PATH = os.path.join(BASE_PATH, 'DB')
PLOTS_PATH = os.path.join(BASE_PATH, 'Plots')

names_dict = {
    'sm': 'SerumMetabolomics',
    'ag': 'Age_Gender',
    'agb': 'Age_Gender_BMI',
    'smag': 'SerumMetabolomics_Age_Gender',
    'smagb': 'SerumMetabolomics_Age_Gender_BMI',
    'blood': 'BloodTest',
    'body': 'BodyMeasures',
    'gmb': 'GutMicroBiome',
    'rs': 'RiskScores'
}


PREDICTION_RISK_SCORE_PATH = join(PREDICTIONS_PATH, names_dict['rs'])
PLOT_RISK_SCORE_PATH = join(PLOTS_PATH, names_dict['rs'])
DB_RISK_SCORE_PATH = join(DB_PATH, names_dict['rs'])


# LightGBM hyper-parameter search
hyper_params_dict_lg = \
    {
        'boosting_type': ['gbdt'],
        'objective': ['regression'], #can be 'regression', 'binary', 'multiclass'....
        'metric': ['l2'],
        'learning_rate':  [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'num_leaves':  range(10, 35),
        'max_depth': [-1, 3, 5, 10],
        'num_iterations':  range(500, 2000, 100),
        'min_data_in_leaf':  range(10, 100, 5),
        'feature_fraction': [0.1, 0.2, 0.3, 0.5],  # [1] when using dummy variables
        'bagging_fraction': [0.6, 0.7, 0.8],
        'bagging_freq': [1],
        'lambda_l1': [0, 0.005, 0.01],
        'early_stopping_round': [None],
        'verbose': [-1],
        'num_threads': [2],
        'silent': [True],
    }


# ElasticNet l1 ratio penalty hyper-parameter, when equals to 1, it is just a regular lasso, when equals to 0, it is ridge
l1_ratio = [.1, .5, .7, .9, .95, .99, 1]


risk_factors = ['visceral_fat_volume', 'fat_mass_index', 'appendicular_lean_mass_index', 'bt__hba1c',
                'bt__triglycerides', 'bt__total_cholesterol', 'sitting_blood_pressure_systolic',
                'intima_media_th_mm_2_intima_media_thickness', 'q_box_mean_kpa_mean_elasticity',
                'q_box_mean_pa_s_vi_plus_mean', 'bt__albumin', 'bt__creatinine', 'AHI', 'TotalSleepTime', 'r_abi',
                'qrs_ms', 'qtc_ms', 'pcp_hf', 'fib_4', 'ascvd_percent_risk', 'smoke_tobacco_now']


def load_pickle(path: os.path):
    with open(path, 'rb') as f:
        file_to_load = pickle.load(f)
    return file_to_load


def mkdir_if_not_exist(path: os.path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def save_pickle(file_to_save, path: os.path, file_name_to_save: str):
    mkdir_if_not_exist(path=path)
    full_path = join(path, file_name_to_save)
    with open(full_path, 'wb') as f:
        pickle.dump(file_to_save, f)
    return


def postfix_path(path: os.path, to_add: str = 'smag'):
    if to_add not in names_dict.keys():
        return join(path, to_add)
    new_path = join(path, names_dict[to_add])
    return new_path


def load_from_subfolders(base_path: os.path, path_in_folder: os.path, file_name: str, list_of_folder: list = risk_factors):
    df = pd.DataFrame()
    for folder in list_of_folder:
        tmp_path = join(base_path, folder)
        for file in glob.glob(join(tmp_path, path_in_folder, file_name)):
            df = pd.concat([df, pd.DataFrame(load_pickle(file), index=[folder])])

    return df

