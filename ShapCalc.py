import os

import pickle
import numpy as np
import pandas as pd
import shap

import ResultsAnalysis


def load_pickle(path: os.path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    assert isinstance(data, object)
    return data


def shapExplainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer


def shapValues(explainer, x):
    shap_values = explainer.shap_values(x)
    return shap_values


def shapDF(shap_values, x):
    df = pd.DataFrame(shap_values)
    df.columns = x.columns.to_list()
    df.index = x.index.to_list()
    return df


def column_abs_mean(df: pd.DataFrame):
    mean_df = df.abs().mean(axis='index')
    return mean_df



def get_n_important_features(mean_shap_calc: pd.DataFrame, n: int = 5):
    importance = pd.DataFrame()
    shap_t = mean_shap_calc.T
    for col in shap_t.columns:
        ser = pd.Series(shap_t.loc[shap_t.nlargest(n, col).index, col])
        importance = pd.concat([importance, ser], axis='columns')
    return importance.T



def get_most_important_frequency(most_important: pd.DataFrame, freq_threshold: int = 0):
    freq = most_important.count()
    freq = freq[freq > freq_threshold]
    return freq
    # freq = {}
    # most_important_values = np.array(most_important.values())
    # values, count = np.unique(most_important_values, return_counts=True)
    # for i, feature in enumerate(values):
    #     if count[i] > freq_threshold:
    #         freq[feature] = count[i]
    # return freq


def calc_abs_mean_shap(path: os.path, n: int = 5):
    data = load_pickle(path)
    indices = []
    abs_mean_shap_values = pd.DataFrame()
    for i in range(len(data)):
        model = data.model[i]
        x = data.x_test[i]
        y = data.y_test[i]
        explainer = shapExplainer(model)
        shap_values = shapValues(explainer, x)
        shap_df = shapDF(shap_values, x)
        shap_df = column_abs_mean(shap_df)
        # shap_df.rename(y.columns[0])
        indices.append(y.columns[0])
        shap_df = pd.DataFrame(shap_df).T

        abs_mean_shap_values = abs_mean_shap_values.append(shap_df)
    abs_mean_shap_values.index = indices
    return abs_mean_shap_values


