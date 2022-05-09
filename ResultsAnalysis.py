import ast
import os
import sys

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


def cast_to_dicts(df: pd.DataFrame, column_to_cast: str):
    for i in range(len(df.index)):
        df[column_to_cast][i] = ast.literal_eval(df[column_to_cast].tolist()[i])
    return df


def flatten_dicts_to_columns(df: pd.DataFrame, column_to_flatten: str):
    # The json_normalize does not keep the indices
    df_temp = df
    df = pd.json_normalize(data=df[column_to_flatten].to_list())
    df.index = df_temp.index
    return df


