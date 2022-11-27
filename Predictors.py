###################################################################################################
# File: RSCV.py
# Version: 0.0
# Date: 04.06.2019
# Noam Bar, noam.bar@weizmann.ac.il
#
#
# Python version: 3.7
###################################################################################################

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
# from Analyses.P3_AnalysisHelperFunctions import make_dir_if_not_exists, compute_abs_SHAP, compute_signed_abs_SHAP, \
#     log_run_details
import lightgbm as lgb
import shap
import os
import sys

import pickle
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, precision_recall_curve, explained_variance_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso
import argparse
import numpy as np
import pandas as pd

# from addloglevels import sethandlers
from LabUtils.addloglevels import sethandlers
from LabUtils.Utils import mkdirifnotexists
# from queue.qp import qp, fakeqp
from LabQueue.qp import qp, fakeqp
from datetime import datetime
# import Utils # TODO: change all pickle writings to csv

# LightGBM params
learning_rate = [0.1, 0.05, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.002, 0.001, 0.0005, 0.0001]
num_leaves = range(2, 35)
max_depth = [-1, 2, 3, 4, 5, 10, 20, 40, 50]
min_data_in_leaf = range(1, 45, 2)
feature_fraction = [i / 10. for i in range(2, 11)]  # [1] when using dummy variables
metric = ['l2']
early_stopping_rounds = [None]
num_threads = [1]
verbose = [-1]
silent = [True]
n_estimators = range(100, 2500, 50)
bagging_fraction = [i / 10. for i in range(2, 11)]
bagging_freq = [0, 1, 2]
lambda_l1 = [0, 0.001, 0.005, 0.01, 0.1]

# Lasso params
alpha = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]


lightgbm_rscv_space = {'learning_rate': learning_rate, 'max_depth': max_depth,
                       'feature_fraction': feature_fraction, 'num_leaves': num_leaves,
                       'min_data_in_leaf': min_data_in_leaf, 'metric': metric,
                       'early_stopping_rounds': early_stopping_rounds, 'n_estimators': n_estimators,
                       'bagging_fraction': bagging_fraction, 'bagging_freq': bagging_freq,
                       'num_threads': num_threads, 'verbose': verbose, 'silent': silent, 'lambda_l1': lambda_l1}

lasso_rscv_space = {'alpha': alpha}

rscv_space_dic = {'lightgbm': lightgbm_rscv_space, 'Lasso': lasso_rscv_space}

predictors_type = {'trees': ['lightgbm'], 'linear': ['Lasso']}

def compute_abs_SHAP(dic):
    abs_shap = pd.DataFrame(index=dic.keys(), columns=dic[dic.keys()[0]].columns)
    for k in dic:
        abs_shap.loc[k, :] = dic[k].apply(np.abs).apply(np.mean).values.ravel()
    return abs_shap


def compute_signed_abs_SHAP(dic, X):
    signed_shap = pd.DataFrame(index=dic.keys(), columns=dic[dic.keys()[0]].columns)
    for k in dic.keys():
        temp_X = X.loc[dic[k].index].copy()
        temp_dic = dic[k].copy()
        #         for c in signed_shap.columns:
        #             signed_shap.loc[k, c] = np.sign(spearmanr(temp_X[c], temp_dic[c], nan_policy='omit')[0])
        signed_shap.loc[k, :] = temp_X.apply(lambda x: np.sign(spearmanr(x, temp_dic[x.name], nan_policy='omit')[0]))
    return signed_shap

# def log_run_details(command_args):
#     """
#
#     :param command_args:
#     :return:
#     """
#     with open(command_args.output_dir + '/log_run_' + str(datetime.now()).split('.')[0].replace(' ', '_') + '.txt',
#               'w') as handle:
#         handle.write('### Arguments ###\n')
#         handle.write(str(sys.argv[0]) + '\n')
#         for arg in vars(command_args):
#             handle.write(str(arg) + '\t' + str(getattr(command_args, arg)) + '\n')
#         handle.write('\n### Code ###\n')
#         with open(sys.argv[0], 'r') as f:
#             for l in f.readlines():
#                 handle.write(l)

def randomized_search_cv_and_shap(params: dict, Y, idx):
    # print('randomized_search_cv_and_shap', str(idx))
    # if command_args.path_to_X.endswith('.csv'):
    #     X = pd.read_csv(command_args.path_to_X, index_col=0)
    # else:
    #     X = pd.read_pickle(command_args.path_to_X)
    X = params['X']
    # if command_args.log_transform_x:
    #     X = X.apply(np.log10)
    results_df = pd.DataFrame(index=Y.columns)
    predictions_df = pd.DataFrame(index=Y.index, columns=Y.columns)
    shap_values_dic = {}

    # added on 5.3.2019 - if data is small, making this parameter smaller
    if X.shape[0] < 200 or Y.shape[0] < 200:
        min_data_in_leaf = range(1, 25, 2)
        lightgbm_rscv_space['min_data_in_leaf'] = min_data_in_leaf

    for y_name in Y.columns:

        y = Y[y_name].dropna().astype(float).copy()
        X_temp = X.reindex(y.index).dropna(how='all').copy()
        y = y.reindex(X_temp.index)

        if y.shape[0] < 200:
            min_data_in_leaf = range(1, 25, 2)
            lightgbm_rscv_space['min_data_in_leaf'] = min_data_in_leaf

        print (y_name)
        print (y.sort_values().unique())
        print(y.sort_values().unique() == np.array([0., 1.]))
        if (y.unique().shape[0] == 2) and ((type(y.unique().max()) == str) |
                                           (y.sort_values().unique() == np.array([0., 1.])).all()):
            classification_problem = True
        else:
            classification_problem = False

        groups = np.array(range(X_temp.shape[0]))
        group_kfold = GroupKFold(n_splits=params['k_folds'])
        shap_values = pd.DataFrame(np.nan, index=X_temp.index, columns=X_temp.columns)
        final_pred = pd.DataFrame(index=X_temp.index, columns=[y_name])

        try:
            for train_index, test_index in group_kfold.split(X_temp, y, groups):
                X_train, X_test = X_temp.iloc[train_index, :], X_temp.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # if classification_problem:
                #     gbm = lgb.LGBMClassifier()
                # else:
                #     gbm = lgb.LGBMRegressor()
                model = _choose_model(params['model'], classification_problem)

                rscv = RandomizedSearchCV(model, rscv_space_dic[params['model']], n_iter=20)
                rscv.fit(X_train, y_train)
                # use best predictor according to random hyper-parameter search
                if classification_problem:
                    y_pred = rscv.best_estimator_.predict_proba(X_test)
                    final_pred.loc[X_test.index, :] = np.expand_dims(y_pred[:, 1], 1)
                else:
                    y_pred = rscv.best_estimator_.predict(X_test)
                    final_pred.loc[X_test.index, :] = np.expand_dims(y_pred, 1)
                # run SHAP
                if params['model'] in predictors_type['trees']:
                    explainer = shap.TreeExplainer(rscv.best_estimator_)
                    try:
                        # changed on 3.10.2018, last column is the bias column
                        shap_values.loc[X_test.index, :] = explainer.shap_values(X_test)
                    except Exception as e:
                        shap_values.loc[X_test.index, :] = explainer.shap_values(X_test)[:, :-1]
            shap_values_dic[y_name] = shap_values
            results_df = _evaluate_performance(y_name, final_pred.values.ravel(), y, results_df, classification_problem)
            predictions_df.loc[final_pred.index, y_name] = final_pred.values.ravel()
        except Exception as e:
            print(e)
            print('RandomizedSearchCV failed with metabolite %s' % y_name, params['job_name'])
            print('y shape', y.shape)
            continue
    _save_temporary_files(params, idx, shap_values_dic, results_df, predictions_df)
    return

# def _cross_validation(X, y, y_name, do_bootstrap, command_args, model, classification_problem,
#                       random_state=0):
#     """
#
#     :param X:
#     :param y:
#     :param y_name:
#     :param do_bootstrap:
#     :param command_args:
#     :param classification_problem:
#     :param predictor_params:
#     :param random_state:
#     :return:
#     """
#     groups = np.array(range(X.shape[0]))
#     group_kfold = GroupKFold(n_splits=command_args.k_folds)
#     final_pred = pd.DataFrame(index=X.index, columns=[y_name])
#     for train_index, test_index in group_kfold.split(X, y, groups):
#         X_train, X_test = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()
#         y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
#         if do_bootstrap:
#             boot = resample(X_train.index, replace=True, n_samples=X_train.shape[0])  # , random_state=random_state
#             X_train, y_train = X_train.loc[boot], y_train.loc[boot]
#
#         model.fit(X_train, y_train)
#         if classification_problem:
#             y_pred = model.predict_proba(X_test)
#             final_pred.loc[X_test.index, :] = np.expand_dims(y_pred[:, 1], 1)
#         else:
#             y_pred = model.predict(X_test)
#             final_pred.loc[X_test.index, :] = np.expand_dims(y_pred, 1)
#         final_pred.loc[X_test.index, :] = np.expand_dims(y_pred, 1)
#     return final_pred


def _choose_model(model, classification_problem):
    if model == 'lightgbm':
        if classification_problem:
            return lgb.LGBMClassifier()
        else:
            return lgb.LGBMRegressor()
    # In a linear regression model we don't allow any missing values,
    # so I need to add code that checks no missing values are present
    elif model == 'Lasso':
        return Lasso()
    else:
        return None


def _save_temporary_files(params, idx, shap_values_dic, results_df, predictions_df):
    with open(params['output_dir'] + '/temp_resdf_' + str(idx) + '.pkl', 'wb') as fout:
        pickle.dump(results_df, fout)
    if params['model'] in predictors_type['trees']:
        with open(params['output_dir'] + '/temp_shap_' + str(idx) + '.pkl', 'wb') as fout:
            pickle.dump(shap_values_dic, fout)
    with open(params['output_dir'] + '/temp_pred_' + str(idx) + '.pkl', 'wb') as fout:
        pickle.dump(predictions_df, fout)
    return


def _evaluate_performance(y_name, y_pred, y_test, results_df, classification_problem):
    results_df.loc[y_name, 'Size'] = y_pred.shape[0]
    if classification_problem:
        # Prevalence
        results_df.loc[y_name, 'prevalence'] = float(y_test.sum()) / y_test.shape[0]
        # AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        results_df.loc[y_name, 'AUC'] = metrics.auc(fpr, tpr)
        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        results_df.loc[y_name, 'Precision_Recall'] = metrics.auc(recall, precision)
    else:
        results_df.loc[y_name, 'Coefficient_of_determination'] = r2_score(y_true=y_test, y_pred=y_pred)
        results_df.loc[y_name, 'explained_variance_score'] = explained_variance_score(y_true=y_test, y_pred=y_pred)
        results_df.loc[y_name, 'pearson_r'], results_df.loc[y_name, 'pearson_p'] = pearsonr(y_pred, y_test)
        results_df.loc[y_name, 'spearman_r'], results_df.loc[y_name, 'spearman_p'] = spearmanr(y_pred, y_test)
    return results_df


def concat_outputs(params):
    print('concat_outputs')
    all_temp_files = os.listdir(params['output_dir'])
    resdf_files = [params['output_dir'] + f for f in all_temp_files if f.startswith('temp_resdf_')]
    shap_files = [params['output_dir'] + f for f in all_temp_files if f.startswith('temp_shap_')]
    pred_files = [params['output_dir'] + f for f in all_temp_files if f.startswith('temp_pred_')]
    _concat_files(resdf_files, params['output_dir'] + '/results.pkl', how='dataframe')
    if params['model'] in predictors_type['trees']:
        _concat_files(shap_files, params['output_dir'] + '/shap_values.pkl', how='dic')
    _concat_files(pred_files, params['output_dir'] + '/predictions_df.pkl', how='dataframe', axis=1)
    return


def _concat_files(files, final_path, how='dataframe', axis=0):
    if how == 'dataframe':
        final_file = pd.DataFrame()
        for f in files:
            final_file = pd.concat((final_file, pd.read_pickle(f)), axis=axis)
            os.remove(f)
        with open(final_path, 'wb') as fout:
            pickle.dump(final_file, fout)
        final_file.to_csv(((final_path.split('.pkl')[0]).split('.dat')[0]) + '.csv')
    elif how == 'dic':
        final_file = {}
        for f in files:
            final_file.update(pd.read_pickle(f))
            os.remove(f)
        with open(final_path, 'wb') as fout:
            pickle.dump(final_file, fout)
    return
    # TODO: add also csv option...


def _compute_abs_and_sign_SHAP(shap_dir, x):
    print('loading shap_values.pkl ...')
    shap_values_dic = pd.read_pickle(shap_dir + 'shap_values.pkl')
    if not os.path.exists(shap_dir + 'abs_shap.dat'):
        abs_shap = compute_abs_SHAP(shap_values_dic).astype(float)
        # Utils.Write(shap_dir + 'abs_shap.dat', abs_shap)
        abs_shap.to_csv(shap_dir + 'abs_shap.csv')
    else:
        abs_shap = pd.read_pickle(shap_dir + 'abs_shap.dat')

    if not os.path.exists(shap_dir + 'signed_shap.dat'):
        # if features_path.endswith('.csv'):
        #     X = pd.read_csv(features_path, index_col=0)
        # else:
        #     X = pd.read_pickle(features_path)
        X = x

        signed_shap = compute_signed_abs_SHAP({ind: shap_values_dic[ind] for ind in abs_shap.index}, X).fillna(0)
        # Utils.Write(shap_dir + 'signed_shap.dat', signed_shap)
        signed_shap.to_csv(shap_dir + 'signed_shap.csv')
    else:
        signed_shap = pd.read_pickle(shap_dir + 'signed_shap.dat')

    abs_signed_shap = (abs_shap.copy() * signed_shap.copy()).fillna(0)
    # Utils.Write(shap_dir + 'abs_signed_shap.dat', abs_signed_shap)
    abs_signed_shap.to_csv(shap_dir + 'abs_signed_shap.csv')

    return

def upload_these_jobs(q, params: dict):
    print('upload_these_jobs')
    waiton = []
    Y = params['Y']
    # if command_args.path_to_Y.endswith('.csv'):
    #     Y = pd.read_csv(command_args.path_to_Y, index_col=0)
    # else:
    #     Y = pd.read_pickle(command_args.path_to_Y)

    for idx in range(0, Y.shape[1], 10):
        waiton.append(q.method(randomized_search_cv_and_shap, (params,
                                                               Y.iloc[:, idx:idx + 10],
                                                               idx)))
    print('Will run a total of ' + str(len(waiton)) + ' jobs')
    res = q.waitforresults(waiton)
    # merge the temp results files
    concat_outputs(params)
    # compute absolute SHAP values and signed absolute SHAP values
    if params['model'] in predictors_type['trees']:
        _compute_abs_and_sign_SHAP(params['output_dir'], params['X'])
    return res


def predict(**kwargs):
    from LabData import config_global as config
    sethandlers(file_dir=config.log_dir)
    print('Predicting')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('output_dir', help='Path to output directory', type=str, default=None)
    # parser.add_argument('-model', help='Which prediction model to use', type=str, default='lightgbm')
    # parser.add_argument('-n_cols_per_job', help='Number of columns per job', type=int, default=10)
    # parser.add_argument('-n_random', help='Number of random samples', type=int, default=20)
    # parser.add_argument('-path_to_X', '--path_to_X', help='Path to features data - X', type=str,
    #                     default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Metabolon/SHAP/dataframes/mar17_features+diet+MPA_species.dat')
    # parser.add_argument('-path_to_Y', help='Path to labels - Y', type=str,
    #                     default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Metabolon/SHAP/dataframes/mar17_metabolomics_unnormed.dat')
    # parser.add_argument('-k_folds', help='Number of folds', type=int, default=10)
    # parser.add_argument('-only_concat', help='Whether to only concatenate the output files', type=bool, default=False)
    # parser.add_argument('-only_compute_abs_SHAP', help='Whether to only compute absolute SHAP values', type=bool,
    #                     default=False)
    # parser.add_argument('-mem_def', help='Amount of memory per job', type=int, default=2)
    # parser.add_argument('-job_name', help='Job preffix for q', type=str, default='SHAP-RSCV')
    # parser.add_argument('-log_transform_x', help='Whether to log transform the X', type=bool, default=False)
    # command_args = parser.parse_args()

    # parser = argparse.ArgumentParser()

    job_name = 'SHAP-RSCV'
    mem_def = 2

    model = kwargs['model']
    X = kwargs['X']
    Y = kwargs['Y']
    k_folds = kwargs['k_folds']
    output_dir = "/net/mraid08/export/genie/LabData/Analyses/galavner/Predictions/"

    params = {
        'job_name': job_name,
        'mem_def': mem_def,
        'model': model,
        'X': X,
        'Y': Y,
        'k_folds': k_folds,
        'output_dir': output_dir    }

       # if command_args.only_concat:
        #     concat_outputs(command_args)
        #     return
        #
        # if command_args.only_compute_abs_SHAP:
        #     _compute_abs_and_sign_SHAP(command_args.output_dir, command_args.path_to_X);
        #     return

        # mkdirifnotexists(command_args.output_dir)

        # log_run_details(command_args)

        # qp = fakeqp
    os.chdir("/net/mraid08/export/genie/LabData/Analyses/galavner/Predictions/")
    with qp(jobname=job_name, q=['himem7.q'], _mem_def=str(mem_def) + 'G',
            _trds_def=2, max_u=650) as q:
        q.startpermanentrun()
        upload_these_jobs(q, params)

    return