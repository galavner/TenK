import ast
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def flatten_dicts_to_columns(df: pd.DataFrame or pd.Series, column_to_flatten: str = None):
    # The json_normalize does not keep the indices
    df_temp = df
    if column_to_flatten is not None:
        df = pd.json_normalize(data=df[column_to_flatten].to_list())
    else:
        df = pd.json_normalize(data=df.tolist())
    df.index = df_temp.index
    return df


def convert_to_np(sr: pd.Series):
    nparray = sr.values
    return nparray


def scatter_plot(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of keyword arguments to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.scatter(data1, data2, **param_dict)
    return out


def add_text(ax, loc_x, loc_y, text: str):
    out = ax.text(loc_x, loc_y, text)
    return out



def plot_scatter(pred: pd.Series, test: pd.Series):
    fig, ax = plt.subplots()
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(test.name)
    x = convert_to_np(pred)
    y = convert_to_np(test)
    ax.plot(x, y)



def main():
    base_path = "/net/mraid08/export/genie/LabData/Analyses/galavner/Predictions/"
    file_name = 'Results_SM_pickle'

    path_to_file = os.path.join(base_path, file_name)

    df = load_pickle(path_to_file)
    metrics = df['results_df']
    metrics = flatten_dicts_to_columns(metrics)
    pearson_r = metrics['pearson_r']



