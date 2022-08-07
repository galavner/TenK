import os
import sys

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

from pandas import DataFrame, concat, Series
from sklearn import metrics
from sklearn.linear_model import RidgeCV, LarsCV, LassoCV, ElasticNetCV, LassoLarsCV, RANSACRegressor, HuberRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score, precision_recall_curve
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer

import config_local as cl
from config_local import save_pickle, mkdir_if_not_exist


def RiskScoresPredictor(
        x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
        folder_dir: os.path, model_type='lightgbm'):

    # y is a 1-dim DataFrame for convenience, so it has only one column
    risk_name = y_train.columns.to_list()[0]

    risk_dir = os.path.join(folder_dir, risk_name)
    tree_dir = os.path.join(risk_dir, 'tree')
    lasso_dir = os.path.join(risk_dir, 'lasso')

    mkdir_if_not_exist(risk_dir)
    mkdir_if_not_exist(tree_dir)
    mkdir_if_not_exist(lasso_dir)

    # Normalizing the datasets the same way for both model so their results are more comparable
    x_train, x_test, x_norm_params = Standardization(x_train, x_test)
    y_train, y_test, y_norm_params = Standardization(y_train, y_test)

    save_pickle(x_train, risk_dir, 'x_train_normalizes.pickle')
    save_pickle(x_test, risk_dir, 'x_test_normalizes.pickle')
    save_pickle(x_norm_params, risk_dir, 'x_normalization_parameters.pickle')
    save_pickle(y_train, risk_dir, 'y_train_normalizes.pickle')
    save_pickle(y_test, risk_dir, 'y_test_normalizes.pickle')
    save_pickle(y_norm_params, risk_dir, 'y_normalization_parameters.pickle')

    # Classification or regression
    is_classification = IsClassification(y_train[risk_name])

    y_pred_tree, model_tree, metrics_tree = predictLightGBM(
        x_train, x_test, y_train, y_test, is_classification=is_classification)

    y_pred_lasso, model_lasso, metrics_lasso = predictLassoCV(
        x_train, x_test, y_train, y_test, is_classification=is_classification)

    save_pickle(y_pred_tree, tree_dir, 'y_pred.pickle')
    save_pickle(model_tree, tree_dir, 'model.pickle')
    save_pickle(metrics_tree, tree_dir, 'metrics.pickle')
    DataFrame(metrics_tree, index=[risk_name]).to_csv(os.path.join(tree_dir, 'metrics.csv'))

    save_pickle(y_pred_lasso, lasso_dir, 'y_pred.pickle')
    save_pickle(model_lasso, lasso_dir, 'model.pickle')
    save_pickle(metrics_lasso, lasso_dir, 'metrics.pickle')
    DataFrame(metrics_lasso, index=[risk_name]).to_csv(os.path.join(lasso_dir, 'metrics.csv'))

    results_dict = {
        'x_train': x_train,
        'x_test': x_test,
        'x_norm_params': x_norm_params,
        'y_train': y_train,
        'y_test': y_test,
        'y_norm_params': y_norm_params,
        'y_pred_tree': y_pred_tree,
        'model_tree': model_tree,
        'metrics_tree': metrics_tree,
        'y_pred_lasso': y_pred_lasso,
        'model_lasso': model_lasso,
        'metrics_lasso': metrics_lasso
    }

    # Convert the dict to DF with index label being the current risk score
    results_df = pd.json_normalize(results_dict, max_level=0)
    results_df.index = [risk_name]

    cl.save_pickle(results_df, risk_dir, 'results.pickle')
    results_df.to_csv(os.path.join(risk_dir, 'results.csv'))

    return results_df



def IsClassification(y):
    if (y.unique().shape[0] == 2) and ((type(y.unique().max()) == str) |
                                       (y.sort_values().unique() == np.array([0., 1.])).all()):
        return True
    return False


def predictLightGBM(x_train: DataFrame, x_test: DataFrame, y_train: Series, y_test: Series,
                    is_classification: bool = False):
    tree_model = lgb.LGBMClassifier() if is_classification else lgb.LGBMRegressor()
    rscv = RandomizedSearchCV(tree_model, cl.hyper_params_dict_lg)
    rscv.fit(x_train, y_train)
    predictor = rscv.best_estimator_
    y_pred = get_prediction(x_test, predictor)
    tree_metrics = evaluate_performance(y_pred, y_test, is_classification)

    return y_pred, predictor, tree_metrics


def predictLassoCV(x_train: DataFrame, x_test: DataFrame, y_train: Series, y_test: Series,
                   is_classification: bool = False):
    if x_train.isna().sum or x_test.isna().sum:
        x_train, x_test = _imputation(x_train=x_train, x_test=x_test)
    lasso_model = LassoCV().fit(x_train, y_train)
    y_pred = get_prediction(x_test, lasso_model)
    lasso_metrics = evaluate_performance(y_pred, y_test, is_classification)

    return y_pred, lasso_model, lasso_metrics


def evaluate_performance(y_pred, y_test, classification_problem: bool = False):
    results_dict = {}
    if classification_problem:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        results_dict = {
            'prevalence': float(y_test.sum()) / y_test.shape[0],
            'AUC': metrics.auc(fpr, tpr),
            'Precision_Recall': metrics.auc(recall, precision)
        }
    else:
        results_dict = {
            'Coefficient_of_determination': r2_score(y_true=y_test, y_pred=y_pred),
            'explained_variance_score': explained_variance_score(y_true=y_test, y_pred=y_pred),
            'pearson_r': pearsonr(y_pred, y_test)[0],
            'pearson_p_value': pearsonr(y_pred, y_test)[1],
            'spearman_r': spearmanr(y_pred, y_test)[0],
            'spearman_p_value': spearmanr(y_pred, y_test)[1]
        }
    return results_dict


def get_prediction(x: DataFrame, model):
    return model.predict(x).ravel()


def Standardization(x_train, x_test):
    scaler = StandardScaler()
    x_train_norm = DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_norm = DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    norm_params = scaler.get_params()
    return x_train_norm, x_test_norm, norm_params


def _imputation(x_train: DataFrame, x_test: DataFrame):
    train_columns = x_train.columns
    train_index = x_train.index
    test_columns = x_test.columns
    test_index = x_test.index

    imputer = KNNImputer(n_neighbors=5)
    train_data = imputer.fit_transform(x_train)
    test_data = imputer.transform(x_test)

    train_imputed = pd.DataFrame(train_data, columns=train_columns, index=train_index)
    test_imputed = pd.DataFrame(test_data, columns=test_columns, index=test_index)

    return train_imputed, test_imputed
