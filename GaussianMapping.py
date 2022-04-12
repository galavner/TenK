import numpy as np
import pandas as pd

from scipy.stats import shapiro
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, FunctionTransformer

def power_transfom(feature_train: pd.Series, feature_test: pd.Series):
    pt_yj = PowerTransformer('yeo-johnson')
    pt_bc = PowerTransformer('box-cox')

    sw_yj = sw_bc = 0
    train_bc = test_bc = None
    # bc Does not allow for negative input

    if not ((feature_train.values < 0).any() and (feature_test.values < 0).any()):
        pt_bc.fit(feature_train)
        train_bc = pt_bc.transform(feature_train)
        test_bc = pt_bc.transform(feature_test)

        sw_bc = shapiro(train_bc)

    pt_yj.fit(feature_train)
    train_yj = pt_yj.transform(feature_train)
    test_yj = pt_yj.transform(feature_test)

    sw_yj = shapiro(train_yj)

    return sw_yj, train_yj, test_yj, sw_bc, train_bc, test_bc


def log_transform(feature_train: pd.Series, feature_test: pd.Series):

    lt = FunctionTransformer(np.log1p, validate=True)

    train_log = lt.transform(feature_train)
    test_log = lt.transform(feature_test)

    sw_log = shapiro(train_log)

    return sw_log, train_log, test_log



def best_transformer(feature_train: pd.Series, feature_test: pd.Series, alpha):
    sw_yj, train_yj, test_yj, sw_bc, train_bc, test_bc = power_transfom(feature_train, feature_test)
    sw_log, train_log, test_log = log_transform(feature_train, feature_test)

    sw = (sw_yj, sw_bc, sw_log)
    train = (train_yj, train_bc, train_log)
    test = (test_yj, test_bc, test_log)

    for transfomer_name in range(len(sw)):
        if sw[transfomer_name] == max(sw) and sw[transfomer_name] > alpha:
            return train[transfomer_name], test[transfomer_name]
    return feature_train, feature_test




def make_gaussian(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, alpha: float64):
    x_train_transformed = x_train
    x_test_transformed = x_test
    y_train_transformed = y_train
    y_test_transformed = y_test


    for feature in x_train.columns:
        x_train_transformed[feature], x_test_transformed[feature] = best_transformer(x_train[feature], x_test[feature], alpha)

    y_train_transformed, y_test_transformed = best_transformer(y_train, y_test, alpha)

    return x_train_transformed, x_test_transformed, y_train_transformed, y_test_transformed


















