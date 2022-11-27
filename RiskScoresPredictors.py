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
        folder_dir: os.path, iter: int = 0, model_type='lightgbm'):

    # y is a 1-dim DataFrame for convenience, so it has only one column
    risk_name = y_train.columns.to_list()[0]

    risk_dir = os.path.join(folder_dir, risk_name, str(iter))
    tree_dir = os.path.join(risk_dir, 'tree')
    lasso_dir = os.path.join(risk_dir, 'lasso')
    elastic_net_dir = os.path.join(risk_dir, 'elastic_net')

    mkdir_if_not_exist(risk_dir)
    mkdir_if_not_exist(tree_dir)
    mkdir_if_not_exist(lasso_dir)
    mkdir_if_not_exist(elastic_net_dir)

    # Classification or regression
    is_classification = IsClassification(y_train[risk_name])

    # Normalizing the datasets the same way for both model so their results are more comparable
    x_train, x_test, x_norm_scaler = Standardization(x_train, x_test)
    y_norm_scaler = None
    if not is_classification:
        y_train, y_test, y_norm_scaler = Standardization(y_train, y_test)

    save_pickle(x_train, risk_dir, 'x_train_normalizes.pickle')
    save_pickle(x_test, risk_dir, 'x_test_normalizes.pickle')
    save_pickle(x_norm_scaler, risk_dir, 'x_scaler.pickle')
    save_pickle(y_train, risk_dir, 'y_train_normalizes.pickle')
    save_pickle(y_test, risk_dir, 'y_test_normalizes.pickle')
    save_pickle(y_norm_scaler, risk_dir, 'y_scaler.pickle')

    y_pred_tree, model_tree, metrics_tree = predictLightGBM(
        x_train, x_test, y_train, y_test, is_classification=is_classification)

    y_pred_lasso, model_lasso, metrics_lasso = predictLassoCV(
        x_train, x_test, y_train, y_test, is_classification=is_classification)

    y_pred_elastic_net, model_elastic_net, metrics_elastic_net = predictElasticNetCV(
        x_train, x_test, y_train, y_test, is_classification=is_classification
    )

    save_pickle(y_pred_tree, tree_dir, 'y_pred.pickle')
    save_pickle(model_tree, tree_dir, 'model.pickle')
    save_pickle(metrics_tree, tree_dir, 'metrics.pickle')
    DataFrame(metrics_tree, index=[risk_name]).to_csv(os.path.join(tree_dir, 'metrics.csv'))

    save_pickle(y_pred_lasso, lasso_dir, 'y_pred.pickle')
    save_pickle(model_lasso, lasso_dir, 'model.pickle')
    save_pickle(metrics_lasso, lasso_dir, 'metrics.pickle')
    DataFrame(metrics_lasso, index=[risk_name]).to_csv(os.path.join(lasso_dir, 'metrics.csv'))

    save_pickle(y_pred_elastic_net, elastic_net_dir, 'y_pred.pickle')
    save_pickle(model_elastic_net, elastic_net_dir, 'model.pickle')
    save_pickle(metrics_elastic_net, elastic_net_dir, 'metrics.pickle')
    DataFrame(metrics_elastic_net, index=[risk_name]).to_csv(os.path.join(elastic_net_dir, 'metrics.csv'))

    results_dict = {
        'x_train': x_train,
        'x_test': x_test,
        'x_norm_scaler': x_norm_scaler,
        'y_train': y_train,
        'y_test': y_test,
        'y_norm_scaler': y_norm_scaler,
        'y_pred_tree': y_pred_tree,
        'model_tree': model_tree,
        'metrics_tree': metrics_tree,
        'y_pred_lasso': y_pred_lasso,
        'model_lasso': model_lasso,
        'metrics_lasso': metrics_lasso,
        'y_pred_elastic_net': y_pred_elastic_net,
        'model_elastic_net': model_elastic_net,
        'metrics_elastic_net': metrics_elastic_net
    }

    # Convert the dict to DF with index label being the current risk score
    results_df = pd.json_normalize(results_dict, max_level=0)
    results_df.index = [risk_name]

    cl.save_pickle(results_df, risk_dir, 'results.pickle')
    results_df.to_csv(os.path.join(risk_dir, 'results.csv'))

    return results_df



def IsClassification(y):
    if (y.unique().shape[0] == 2) and ((type(y.unique().max()) == str) |
                                       (y.sort_values().unique() == np.array([0., 1.])) |
                                      (y.sort_values().unique() == np.array([0, 1]))).all():
        return True
    return False


def predictLightGBM(x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                    is_classification: bool = False):
    hyper_params_dict = cl.hyper_params_dict_lg

    if x_train.shape[1] < 10:
        hyper_params_dict['feature_fraction'] = [1]

    tree_model = lgb.LGBMClassifier() if is_classification else lgb.LGBMRegressor()
    rscv = RandomizedSearchCV(tree_model, hyper_params_dict)
    rscv.fit(x_train, y_train)
    predictor = rscv.best_estimator_
    y_pred = DataFrame(get_prediction(x_test, predictor), index=x_test.index, columns=y_test.columns)
    tree_metrics = evaluate_performance(y_pred, y_test, is_classification)

    return y_pred, predictor, tree_metrics


def predictLassoCV(x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                   is_classification: bool = False):
    if x_train.isna().sum().sum() or x_test.isna().sum().sum():
        x_train, x_test = _imputation(x_train=x_train, x_test=x_test)
    lasso_model = LassoCV().fit(x_train, y_train)
    y_pred = DataFrame(get_prediction(x_test, lasso_model), index=x_test.index, columns=y_test.columns)
    lasso_metrics = evaluate_performance(y_pred, y_test, is_classification)

    return y_pred, lasso_model, lasso_metrics


def predictElasticNetCV(x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                   is_classification: bool = False):
    if x_train.isna().sum().sum() or x_test.isna().sum().sum():
        x_train, x_test = _imputation(x_train=x_train, x_test=x_test)
    l1_ratio = cl.l1_ratio
    elastic_net_model = ElasticNetCV(l1_ratio=l1_ratio).fit(x_train, y_train)
    y_pred = DataFrame(get_prediction(x_test, elastic_net_model), index=x_test.index, columns=y_test.columns)
    elastic_net_metrics = evaluate_performance(y_pred, y_test, is_classification)

    return y_pred, elastic_net_model, elastic_net_metrics


def evaluate_performance(y_pred, y_test, classification_problem: bool = False):
    results_dict = {}
    y_name = y_test.columns[0]
    if classification_problem:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        results_dict = {
            'prevalence': float(y_test[y_name].sum()) / y_test[y_name].shape[0],
            'AUC': metrics.auc(fpr, tpr),
            'Precision_Recall': metrics.auc(recall, precision)
        }
    else:
        results_dict = {
            'Coefficient_of_determination': r2_score(y_true=y_test, y_pred=y_pred),
            'explained_variance_score': explained_variance_score(y_true=y_test, y_pred=y_pred),
            'pearson_r': pearsonr(y_pred[y_name], y_test[y_name])[0],
            'pearson_p_value': pearsonr(y_pred[y_name], y_test[y_name])[1],
            'spearman_r': spearmanr(y_pred[y_name], y_test[y_name])[0],
            'spearman_p_value': spearmanr(y_pred[y_name], y_test[y_name])[1]
        }
    return results_dict


def get_prediction(x: DataFrame, model):
    return model.predict(x).ravel()


def Standardization(x_train, x_test):
    scaler = StandardScaler()
    if 'gender' in x_train.columns:
        x_train_tmp = x_train.drop(['gender'], axis=1)
        x_test_tmp = x_test.drop(['gender'], axis=1)
        x_train_norm = DataFrame(scaler.fit_transform(x_train_tmp), columns=x_train_tmp.columns, index=x_train_tmp.index)
        x_test_norm = DataFrame(scaler.transform(x_test_tmp), columns=x_test_tmp.columns, index=x_test_tmp.index)
        x_train_norm = pd.concat([x_train_norm, x_train['gender']], axis=1)
        x_test_norm = pd.concat([x_test_norm, x_test['gender']], axis=1)
    else:
        x_train_norm = DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
        x_test_norm = DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    norm_params = scaler.get_params()
    return x_train_norm, x_test_norm, scaler


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
