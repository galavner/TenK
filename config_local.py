import os

import pickle

BASE_PATH = os.path.join('/net', 'mraid08', 'export', 'genie', 'LabData', 'Analyses', 'galavner')

PREDICTIONS_PATH = os.path.join(BASE_PATH, 'Predictions')
DB_PATH = os.path.join(BASE_PATH, 'DB')
PLOTS_PATH = os.path.join(BASE_PATH, 'Plots')

loaders_name_dict = {
    'sm': 'SerumMetabolomics',
    'ag': 'age_gender',
    'smag': 'SerumMetabolomics_age_gender',
    'blood': 'BLoodTest',
    'body': 'BodyMeasures',
    'gmb': 'GutMicroBiome'
}


def load_pickle(path: os.path):
    with open(path, 'rb') as f:
        file_to_load = pickle.load(f)
    return file_to_load


def save_pickle(file_to_save, path: os.path, file_name_to_save: str):
    with open(os.path.join(path, file_name_to_save), 'wb') as f:
        pickle.dump(file_to_save, f)
    return
