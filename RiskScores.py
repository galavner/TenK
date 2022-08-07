import os
import sys
import time
import glob

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from LabUtils.addloglevels import sethandlers
from LabUtils.Utils import mkdirifnotexists
from LabQueue.qp import qp, fakeqp

import config_local as cl
import RiskScoresPredictors


os.chdir(cl.PREDICTIONS_PATH)
sethandlers()


def main():


    with qp(jobname='Gal_LGBM', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=500,
                       _mem_def=6) as q:
        q.startpermanentrun()

        folder_dir = os.path.join(cl.PREDICTIONS_PATH, 'RiskScores')
        db_dir = os.path.join(cl.DB_PATH, 'RiskScores')
        x_path = os.path.join(db_dir, 'x')
        y_path = os.path.join(db_dir, 'y')

        cl.mkdir_if_not_exist(folder_dir)

        # Load all different DataFrames risk scores in the y_path directory to a single y_df variable
        x_df = pd.DataFrame()
        for file in glob.glob(os.path.join(x_path, '*.pickle')):
            x_df = pd.concat([x_df, cl.load_pickle(file)], axis=1)

        # Load all different DataFrames risk scores in the y_path directory to a single y_df variable
        y_df = pd.DataFrame()
        for file in glob.glob(os.path.join(y_path, '*.pickle')):
            y_df = pd.concat([y_df, cl.load_pickle(file)], axis=1)

        # In this case, x_df.shape < y_df.shape, maybe it should be the other way around if it is reversed, or just taking the max
        y_df = y_df.reindex(x_df.index)

        x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

        tkttores = {}
        col_names = []

        for i, y_col in enumerate(y_df.columns):

            y_train = pd.DataFrame(y_train_df[y_col].dropna())
            y_test = pd.DataFrame(y_test_df[y_col].dropna())

            x_train = x_train_df.reindex(y_train.index).dropna(how='all')
            x_test = x_test_df.reindex(y_test.index).dropna(how='all')

            y_train = y_train.reindex(x_train.index)
            y_test = y_test.reindex(x_test.index)

            if x_train.shape[0] < 500:
                continue

            tkttores[(i, y_col)] = q.method(RiskScoresPredictors.RiskScoresPredictor, (x_train, x_test,
                                                                                       y_train, y_test, folder_dir))
            col_names.append(y_col)

        tkttores = {k: q.waitforresult(v) for k, v in tkttores.items()}

        # Combining all dataframes that are stored in the tkttores dict's values
        results = pd.concat(list(tkttores.values()))


        # Old method from SerumMetabolomics.py, doesn't work for some reason now

        # results = pd.DataFrame(tkttores).T
        # results.columns = ['x_train', 'x_test', 'y_train', 'y_test', 'y_pred', 'model',
        #                    'results_dict', 'permutation_results', 'is_signal']
        # results.index = col_names

        results.to_csv(os.path.join(folder_dir, 'combined_results.csv'))

        cl.save_pickle(results, folder_dir, 'combined_results.pickle')





if __name__ == '__main__':
    main()
