import os
import time

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score, explained_variance_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import pearsonr, spearmanr
import shap
import lightgbm as lgb

from LabData import config_global as config

# LightGBM
lg_boosting_type = 'gbdt'
lg_objective = 'regression'
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
        'objective': ['regression'],
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


#
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=20,
#                 valid_sets=lgb_eval,
#                 callbacks=[lgb.early_stopping(stopping_rounds=5)])
#
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
# print(f'The RMSE of prediction is: {rmse_test}')


def is_classification(y):
    if (y.unique().shape[0] == 2) and ((type(y.unique().max()) == str) |
                                       (y.sort_values().unique() == np.array([0., 1.])).all()):
        return True
    return False


def fit_hyper_params_search(x, y, classification_problem):
    lgb_model = lgb.LGBMClassifier() if classification_problem else lgb.LGBMRegressor()
    rscv = RandomizedSearchCV(lgb_model, hyper_params_dict_lg)
    rscv.fit(x, y)
    predictor = rscv.best_estimator_
    model = predictor
    return model


def calc_shap(model, x):
    explainer = shap.TreeExplainer(model)
    x_shap = x
    shap_values = explainer.shap_values(x_shap)
    shap_df = DataFrame(columns=x.columns, index=x_shap.index, data=shap_values)
    return shap_df, explainer


def get_prediction(x: DataFrame, model: lgb.LGBMRegressor):
    return model.predict(x).ravel()



def evaluate_performance(y_pred, y_test, classification_problem):
    results_dict = {}
    if classification_problem:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        result_dict = {
            'prevalence': float(y_test.sum()) / y_test.shape[0],
            'AUC': metrics.auc(fpr, tpr),
            'Precision_Recall': metrics.auc(recall, precision)
        }
    else:
        results_dict = {
            'Coefficient_of_determination': r2_score(y_true=y_test, y_pred=y_pred),
            'explained_variance_score': explained_variance_score(y_true=y_test, y_pred=y_pred),
            'pearson_p': pearsonr(y_pred, y_test),
            'spearman_p': spearmanr(y_pred, y_test)
        }
    return pd.Series(results_dict)



def LGBMPredict(x: DataFrame, y: DataFrame):
    base_path = "/net/mraid08/export/genie/LabData/Analyses/galavner/Predictions/"
    out_dir = os.path.join(base_path, y.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(out_dir)
    # with qp(jobname='Gal_LGBM', q=['himem7.q'], _mem_def=str(2) + 'G',
    #         _trds_def=2, max_u=650) as q:
    #     q.startpermanentrun()
    classification_problem = is_classification(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    model = fit_hyper_params_search(x_train, y_train)
    shap_df, explainer = calc_shap(model, x_test)
    yhat = get_prediction(x_test, model)
    result_df = evaluate_performance(yhat, y_test, classification_problem)
    result_df.to_csv(os.path.join(out_dir, 'results.csv'))
    plt.clf()
    shap.summary_plot(explainer.shap_values(x_test), x_test, show=False)
    plt.savefig('summary_plot.png')
    print('finish')

