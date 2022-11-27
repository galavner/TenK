import os
import time

import pickle
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score, explained_variance_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import pearsonr, spearmanr
import shap
import lightgbm as lgb
import statistics
import scipy.stats

import config_local as cl
from LabData import config_global as config

# LightGBM
lg_boosting_type = 'gbdt'
lg_objective = 'regression' # TODO change to classifier accordingly?
lg_metric = 'l2'
lg_learning_rate = 0.03
lg_num_leaves = 31
lg_max_depth = 4
lg_num_iterations = 2000
lg_min_data_in_leaf = 20
lg_feature_fraction = 0.1
lg_bagging_fraction = 0.7
lg_bagging_freq = 1
lg_lambda_l1 = 0
lg_early_stopping_round = None
lg_verbose = -1
lg_num_threads = 2
lg_silent = True

# LightGBM hyper parameter search
hyper_params_dict_lg = \
    {
        'boosting_type': ['gbdt'],
        'objective': ['regression'], #can be 'regression', 'binary', 'multiclass'....
        'metric': ['l2'],
        'learning_rate': [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
        'num_leaves': range(2, 35),
        'max_depth': [-1, 2, 3, 4, 5, 10],
        'num_iterations': range(100, 2500, 50),
        'min_data_in_leaf': range(5, 100, 5),
        'feature_fraction': [i / 10. for i in range(2, 11)],  # [1] when using dummy variables
        'bagging_fraction': [i / 10. for i in range(4, 11)],
        'bagging_freq': [1],
        'lambda_l1': [0, 0.001, 0.005, 0.01],
        'early_stopping_round': [None],
        'verbose': [-1],
        'num_threads': [2],
        'silent': [True],
    }

n_permutations = 100


def is_classification(y):
    if (y.unique().shape[0] == 2) and ((type(y.unique().max()) == str) |
                                       (y.sort_values().unique() == np.array([0., 1.])).all()):
        return True
    return False


def fit_hyper_params_search(x, y, classification_problem, is_unbalance: bool = False):
    if classification_problem:
        lgb_model = lgb.LGBMClassifier()
        hyper_params_dict_lg['objective'] = ['binary']
        if is_unbalance:
            hyper_params_dict_lg['is_unbalance'] = [True]
    else:
        lgb_model = lgb.LGBMRegressor()

    rscv = RandomizedSearchCV(lgb_model, hyper_params_dict_lg)
    rscv.fit(x, y)
    predictor = rscv.best_estimator_
    model = predictor
    return model


def calc_shap(model, x):
    explainer = shap.TreeExplainer(model)
    x_shap = x
    shap_values = explainer.shap_values(x_shap)
    # shap_df = DataFrame(columns=x.columns, index=x_shap.index, data=shap_values[0])
    return explainer


def get_prediction(x: DataFrame, model: lgb.LGBMRegressor):
    return model.predict(x).ravel()


def evaluate_performance(y_pred, y_test, classification_problem):
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


def save_files(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
               y_pred: pd.DataFrame, results_df: pd.DataFrame, model,
               perm_results: pd.DataFrame, is_signal: bool, out_dir, y_col):
    x_train.to_csv(os.path.join(out_dir, 'x_train.csv'))
    x_test.to_csv(os.path.join(out_dir, 'x_test.csv'))
    y_train.to_csv(os.path.join(out_dir, 'y_train.csv'))
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'))
    y_pred.to_csv(os.path.join(out_dir, 'y_pred.csv'))
    results_df.to_csv(os.path.join(out_dir, 'results.csv'))
    model.booster_.save_model('LGBM_model.txt')
    perm_results.to_csv(os.path.join(out_dir, f'Permutations_is_signal_{is_signal}'))

    # combined_dict = {
    #     'x_train': x_train,
    #     'x_test': x_test,
    #     'y_train': y_train,
    #     'y_test': y_test,
    #     'y_pred': y_pred,
    #     'model': model,
    #     'results_df': results_df,
    #     'perm_results': perm_results
    # }
    # combined_df = DataFrame(combined_dict, index=y_col)

    cl.save_pickle(x_train, out_dir, 'x_train.pickle')
    cl.save_pickle(x_test, out_dir, 'x_test.pickle')
    cl.save_pickle(y_train, out_dir, 'y_train.pickle')
    cl.save_pickle(y_test, out_dir, 'y_test.pickle')
    cl.save_pickle(y_pred, out_dir, 'y_pred.pickle')
    cl.save_pickle(results_df, out_dir, 'results.pickle')
    cl.save_pickle(perm_results, out_dir, f'Permutations_is_signal_{is_signal}.pickle')
    cl.save_pickle(model, out_dir, 'Model.pickle')

    # cl.save_pickle(combined_df, out_dir, 'combined_df.pickle')

    # model.booster_.save_model('LGBM_model.txt')


def get_sample_pvalue_out_of_data(data: list, sample: float):
    mu = statistics.mean(data)
    sigma = statistics.stdev(data)
    z = (sample - mu) / sigma
    p = scipy.stats.norm.sf(z)
    return mu, sigma, p


def draw_permutation_distribution(data: list, real_point: float, basepath: os.path):
    mu, sigma, p = get_sample_pvalue_out_of_data(data, real_point)
    sns.histplot(data)
    plt.annotate(f'pvalue={p}', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.axline((real_point, 0), (real_point, 1), linewidth=1, color='r', label='Real Pearson_r')
    plt.savefig(os.path.join(basepath, 'permutation_histogram.png'), bbox_inches='tight')
    return


def permutations_test(x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                      classification_problem: bool, real_results_df, basepath: os.path,
                      do_permutations: bool = True, is_unbalance: bool = False):

    if not do_permutations:
        return None, None

    better_perm_score_count = 0
    is_signal = True
    perm_results = [] #{}
    if not classification_problem:
        metric = 'pearson_r'
    else:
        metric = 'AUC'

    if real_results_df[metric][0] < 0:
        is_signal = False
        return is_signal, perm_results
    for perm_iter in range(n_permutations):
        x_perm = x_train.copy()
        x_perm.index = np.random.permutation(x_perm.index)
        x_perm = x_perm.sample(frac=1)
        results_dict = train_model(x_perm, x_test, y_train, y_test, classification_problem=classification_problem,
                                   is_permuted=True, is_unbalance=is_unbalance)
        perm_results.append(results_dict[metric])
        # for k, v in results_dict.items():
        #     if perm_iter == 0:
        #         perm_results[k] = [v]
        #     else:
        #         perm_results[k].append(v)
        if real_results_df[metric][0] < results_dict[metric]:
            better_perm_score_count = better_perm_score_count + 1
        # if perm_iter > 0
        #     mu, sigma, p = get_sample_pvalue_out_of_data(perm_results, real_results_df[metric][0])
        #     sns.histplot(perm_results)
        #     plt.annotate(f'pvalue={p}', xy=(0.05, 0.95), xycoords='axes fraction')
        #     plt.show()
        if perm_iter == 4 or perm_iter == 9:
            if better_perm_score_count >= 3:
                is_signal = False
                # draw_permutation_distribution(perm_results, real_results_df[metric][0], basepath=basepath)
                return is_signal, perm_results
        if perm_iter == 99:
            if better_perm_score_count >= 5:
                is_signal = False
                # draw_permutation_distribution(perm_results, real_results_df[metric][0], basepath=basepath)
                return is_signal, perm_results
        if perm_iter >= 99:
            if better_perm_score_count/perm_iter > 0.06:
                is_signal = False
                return is_signal, perm_results
    # draw_permutation_distribution(perm_results, real_results_df[metric][0], basepath=basepath)
    return is_signal, perm_results


def train_model(x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                classification_problem: bool, is_permuted: bool = False, is_unbalance: bool=False):
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    model = fit_hyper_params_search(x_train, y_train, classification_problem, is_unbalance=is_unbalance)
    y_pred = get_prediction(x_test, model)
    results_dict = evaluate_performance(y_pred, y_test, classification_problem)

    if is_permuted:
        return results_dict

    return y_pred, model, results_dict


def LGBMPredict(x: DataFrame, y: DataFrame, base_path: os.path, do_permutations: bool = False, is_unbalance: bool = False):
    # base_path = "/net/mraid08/export/genie/LabData/Analyses/galavner/Predictions/AG_to_ABI"
    out_dir = os.path.join(base_path, y.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(out_dir)
    # with qp(jobname='Gal_LGBM', q=['himem7.q'], _mem_def=str(2) + 'G',
    #         _trds_def=2, max_u=650) as q:
    #     q.startpermanentrun()
    classification_problem = is_classification(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # model = fit_hyper_params_search(x_train, y_train, classification_problem)
    # model.booster_.save_model('LGBM_model.txt')

    # y_pred = get_prediction(x_test, model)
    # results_dict = evaluate_performance(y_pred, y_test, classification_problem)
    y_pred, model, results_dict =\
        train_model(x_train, x_test, y_train, y_test, classification_problem=classification_problem, is_unbalance=is_unbalance)
    y_train = pd.DataFrame(y_train, columns=[y.name], index=x_train.index)
    y_test = pd.DataFrame(y_test, columns=[y.name], index=x_test.index)
    y_pred = pd.DataFrame(y_pred, columns=[y.name], index=x_test.index)
    result_df = pd.DataFrame(results_dict, index=[y.name])
    is_signal, permutations_results = permutations_test(x_train, x_test, y_train, y_test,
                                                        classification_problem=classification_problem,
                                                        real_results_df=result_df, basepath=out_dir,
                                                        do_permutations=do_permutations, is_unbalance=is_unbalance)
    perm_df = pd.DataFrame(permutations_results, columns=[y.name])
    explainer = calc_shap(model, x_test)


    # result_df.to_csv(os.path.join(out_dir, 'results.csv'))
    save_files(x_train, x_test, y_train, y_test, y_pred, result_df, model, perm_df, is_signal, out_dir, y.name)
    plt.clf()
    shap.summary_plot(explainer.shap_values(x_test), x_test, show=False)
    plt.savefig('summary_plot.png', bbox_inches='tight')



    print(f'finished {y.name} at {time.strftime("%H:%M:%S", time.localtime())}')
    return x_train, x_test, y_train, y_test, y_pred, model, results_dict, permutations_results, is_signal

