import os
import sys
import time
import glob
from os.path import join

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from LabUtils.addloglevels import sethandlers
from LabUtils.Utils import mkdirifnotexists
from LabQueue.qp import qp, fakeqp

import config_local as cl
import RiskScoresPredictors
import RiskScoresPredictorsFullTrain


os.chdir(cl.PREDICTIONS_PATH)
sethandlers()

X_NAME = 'smagb'

num_of_iterations = 100


def main():
    x_loader = cl.names_dict[X_NAME]
    # run_prediction(x_loader=x_loader)
    full_training_set(x_loader=x_loader)



def run_prediction(x_loader: str):
    
    with qp(jobname='Gal_RF', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=500,
                       _mem_def=6) as q:
        q.startpermanentrun()

        prediction_dir = join(cl.PREDICTIONS_PATH, 'RiskScores', x_loader)
        db_dir = join(cl.DB_PATH, 'RiskScores')
        x_path = join(db_dir, x_loader)
        y_path = join(db_dir, 'y')

        cl.mkdir_if_not_exist(prediction_dir)
        # The DB folder should be there, otherwise there is no data to actually load

        # commented out as the train_test split is done a-priori to all x's
        # Load all different DataFrames risk scores in the y_path directory to a single y_df variable
        x_df = pd.DataFrame()
        for file in glob.glob(join(x_path, 'x.pickle')):
            x_df = pd.concat([x_df, cl.load_pickle(file)], axis=1)
        # x_train_df = cl.load_pickle(join(x_path, 'x_train.pickle'))
        # x_test_df = cl.load_pickle(join(x_path, 'x_test.pickle'))



        # Load all different DataFrames risk scores in the y_path directory to a single y_df variable
        y_df = pd.DataFrame()
        for file in glob.glob(join(y_path, '*.pickle')):
            y_df = pd.concat([y_df, cl.load_pickle(file)], axis=1)

        y_df = y_df[cl.risk_factors]

        # commented out as the train_test split is done a-priori to all x's

        # In this case, x_df.shape < y_df.shape, maybe it should be the other way around if it is reversed, or just taking the max
        # y_df = y_df.reindex(x_df.index)

        # commented out as the train_test split is done a-priori to all x's
        # x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, test_size=0.2, random_state=None)

        # probably unnecessary
        # y_train_df = y_df.reindex(x_train_df.index)
        # y_test_df = y_df.reindex(x_test_df.index)

        x_df = x_df.reindex(y_df.dropna(how='all').index).dropna(how='all')
        y_df = y_df.reindex(x_df.index).dropna(how='all')
        x_df = x_df.reindex(y_df.index)

        tkttores = {}
        col_names = []


        # y_df = y_df[['gender']]

        for iter in range(num_of_iterations):

            x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, test_size=0.2, random_state=None)

            for i, y_col in enumerate(y_df.columns):

                # Do not overwrite existing models, pass over them
                if os.path.exists(join(prediction_dir, y_col, str(iter), 'lasso', 'model.pickle')):
                    continue


                y_train = pd.DataFrame(y_train_df[y_col].dropna())
                y_test = pd.DataFrame(y_test_df[y_col].dropna())

                x_train = x_train_df.reindex(y_train.index).dropna(how='all')
                x_test = x_test_df.reindex(y_test.index).dropna(how='all')

                y_train = y_train.reindex(x_train.index)
                y_test = y_test.reindex(x_test.index)

                if x_train.shape[0] < 500:
                    continue

                tkttores[(iter, y_col)] = q.method(RiskScoresPredictors.RiskScoresPredictor, (
                    x_train, x_test, y_train, y_test, prediction_dir, iter))
                col_names.append(y_col)

        tkttores = {k: q.waitforresult(v) for k, v in tkttores.items()}

        # Combining all dataframes that are stored in the tkttores dict's values
        results = pd.concat(list(tkttores.values()))


        # Old method from SerumMetabolomics.py, doesn't work for some reason now

        # results = pd.DataFrame(tkttores).T
        # results.columns = ['x_train', 'x_test', 'y_train', 'y_test', 'y_pred', 'model',
        #                    'results_dict', 'permutation_results', 'is_signal']
        # results.index = col_names

        results.to_csv(join(prediction_dir, 'combined_results.csv'))

        cl.save_pickle(results, prediction_dir, 'combined_results.pickle')



def full_training_set(x_loader: str):
    with qp(jobname='Gal_RF', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=500,
                       _mem_def=6) as q:
        q.startpermanentrun()

        prediction_dir = join(cl.PREDICTIONS_PATH, 'RiskScores', x_loader)
        db_dir = join(cl.DB_PATH, 'RiskScores')
        x_path = join(db_dir, x_loader)
        y_path = join(db_dir, 'y')

        cl.mkdir_if_not_exist(prediction_dir)
        # The DB folder should be there, otherwise there is no data to actually load

        # commented out as the train_test split is done a-priori to all x's
        # Load all different DataFrames risk scores in the y_path directory to a single y_df variable
        x_df = pd.DataFrame()
        for file in glob.glob(join(x_path, 'x.pickle')):
            x_df = pd.concat([x_df, cl.load_pickle(file)], axis=1)
        # x_train_df = cl.load_pickle(join(x_path, 'x_train.pickle'))
        # x_test_df = cl.load_pickle(join(x_path, 'x_test.pickle'))



        # Load all different DataFrames risk scores in the y_path directory to a single y_df variable
        y_df = pd.DataFrame()
        for file in glob.glob(join(y_path, '*.pickle')):
            y_df = pd.concat([y_df, cl.load_pickle(file)], axis=1)

        y_df = y_df[cl.risk_factors]

        # commented out as the train_test split is done a-priori to all x's

        # In this case, x_df.shape < y_df.shape, maybe it should be the other way around if it is reversed, or just taking the max
        # y_df = y_df.reindex(x_df.index)

        # commented out as the train_test split is done a-priori to all x's
        # x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, test_size=0.2, random_state=None)

        # probably unnecessary
        # y_train_df = y_df.reindex(x_train_df.index)
        # y_test_df = y_df.reindex(x_test_df.index)

        x_df = x_df.reindex(y_df.dropna(how='all').index).dropna(how='all')
        y_df = y_df.reindex(x_df.index).dropna(how='all')
        x_df = x_df.reindex(y_df.index)

        tkttores = {}
        col_names = []


        # y_df = y_df[['gender']]


        for i, y_col in enumerate(y_df.columns):


            y = pd.DataFrame(y_df[y_col].dropna())

            x = x_df.reindex(y.index).dropna(how='all')

            y = y.reindex(x.index)

            if x.shape[0] < 500:
                continue

            tkttores[(iter, y_col)] = q.method(RiskScoresPredictorsFullTrain.RiskScoresPredictorFullTrain, (
                x, y, prediction_dir))
            col_names.append(y_col)

        tkttores = {k: q.waitforresult(v) for k, v in tkttores.items()}

        # Combining all dataframes that are stored in the tkttores dict's values
        results = pd.concat(list(tkttores.values()))


        # Old method from SerumMetabolomics.py, doesn't work for some reason now

        # results = pd.DataFrame(tkttores).T
        # results.columns = ['x_train', 'x_test', 'y_train', 'y_test', 'y_pred', 'model',
        #                    'results_dict', 'permutation_results', 'is_signal']
        # results.index = col_names

        results.to_csv(join(prediction_dir, 'combined_results.csv'))

        cl.save_pickle(results, prediction_dir, 'combined_results.pickle')



if __name__ == '__main__':
    main()
