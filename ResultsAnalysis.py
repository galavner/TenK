import ast
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ResultAnalysis:
    def __init__(self, path: os.path):
        self.path = path
        self.pickle_file = load_pickle(path)
        self.metrics = None
        self.pearson_r
        self.shap_vals = None
        self.shap_freq = None

    def set_metrics(self, columns_to_flatten: str = None):
        metrics_tmp = self.pickle_file.results_df
        metrics_tmp = self._flatten_dicts_to_columns(self, column_to_flatten=columns_to_flatten)
        self.metrics = metrics_tmp


    def get_metrics(self):
        return self.metrics

    def get_pearson_r(self):
        if self.metrics is None:
            self.set_metrics(self)
        self.pearson_r = self.metrics.pearson_r
        return self.pearson_r


    def _flatten_dicts_to_columns(df: pd.DataFrame or pd.Series, **column_to_flatten: str = None):
        # The json_normalize does not keep the indices
        df_temp = df
        if column_to_flatten is not None:
            df = pd.json_normalize(data=df[column_to_flatten].to_list())
        else:
            df = pd.json_normalize(data=df.tolist())
        df.index = df_temp.index
        return df


def load_csv(path: os.path):
    df = pd.read_csv(path)
    if df is None:
        return None
    # pd reads csv in a funny way s.t. the indices are now the first column
    df.index = df[df.columns[0]]
    df = df.drop(columns=df.columns[0])
    return df


def load_pickle(path: os.path):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    return df


def cast_to_dicts(df: pd.DataFrame, column_to_cast: str):
    for i in range(len(df.index)):
        df[column_to_cast][i] = ast.literal_eval(df[column_to_cast].tolist()[i])
    return df




def convert_to_np(sr: pd.Series):
    nparray = sr.values
    return nparray




def main():
    base_path = "/net/mraid08/export/genie/LabData/Analyses/galavner/Predictions/"
    file_name = 'Results_SM_pickle'

    path_to_file = os.path.join(base_path, file_name)

    df = load_pickle(path_to_file)
    metrics = df['results_df']
    metrics = flatten_dicts_to_columns(metrics)
    pearson_r = metrics['pearson_r']



