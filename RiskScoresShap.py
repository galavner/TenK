import os
import sys

from contextlib import contextmanager

import pickle
import numpy as np
import pandas as pd
import shap

import ResultsAnalysis
import config_local as cl

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def load_pickle(path: os.path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    assert isinstance(data, object)
    return data


def shapExplainer(model, model_type, x):
    with suppress_stdout_stderr():
        if model_type == 'tree':
            explainer = shap.explainers.Tree(model, x)
            shap_values = explainer.shap_values(x, check_additivity=False)
        else:
            explainer = shap.LinearExplainer(model, x)
            shap_values = explainer.shap_values(x)

    return explainer, shap_values


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

    importance = pd.Series(mean_shap_calc.nlargest(n))
    return importance
    # for col in shap_t.columns:
    #     ser = pd.Series(shap_t.loc[shap_t.nlargest(n, col).index, col])
    #     importance = pd.concat([importance, ser], axis='columns')
    # return importance.T


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


def ShapClac(
        x: pd.DataFrame(), model, risk_score: str, iteration: int,  model_type: str = 'tree',
        output_path: os.path = None, n_imp_features: int = 5):

    explainer, shap_values = shapExplainer(model, model_type, x)

    shap_df = pd.DataFrame(data=shap_values, columns=x.columns, index=x.index)
    abs_mean_shap = pd.Series(shap_df.abs().mean(axis='index'), name=risk_score).rename_axis(risk_score)
    n_important = pd.Series(get_n_important_features(abs_mean_shap, n=n_imp_features), name=risk_score).rename_axis(risk_score)

    cl.save_pickle(explainer, output_path, 'explainer.pickle')
    cl.save_pickle(shap_values, output_path, 'shap_values.pickle')
    cl.save_pickle(shap_df, output_path, 'shap_df.pickle')
    cl.save_pickle(abs_mean_shap, output_path, 'abs_mean_shap.pickle')
    cl.save_pickle(n_important, output_path, f'{n_imp_features}_important_features.pickle')

    print(f'Finished {risk_score} iteration {iteration} model {model_type}')

    return


