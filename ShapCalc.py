import os

import pickle
import numpy as np
import pandas as pd
import shap


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


def calc_abs_mean_shap(path: os.path):
    data = load_pickle(path)
    mean_shap_values = pd.DataFrame()
    for i in range(len(data)):
        model = data.model[i]
        x = data.x_test[i]
        y = data.y_test[i]
        explainer = shapExplainer(model)
        shap_values = shapValues(explainer, x)
        shap_df = shapDF(shap_values, x)
        shap_df = column_abs_mean(shap_df)
        shap_df.name = y[i].columns[0]
        shap_df = pd.DataFrame(shap_df).T

        mean_shap_values = mean_shap_values.append(shap_df)

    return mean_shap_values


