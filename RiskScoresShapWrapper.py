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
import RiskScoresShap


os.chdir(cl.PREDICTIONS_PATH)
sethandlers()

X_NAME = 'smagb'

n_iter = 1
full_train = True


def main():
    x_loader = cl.names_dict[X_NAME]
    run_prediction(x_loader=x_loader)


def run_prediction(x_loader: str):
    with qp(jobname='Gal_Shap', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=500,
            _mem_def=6) as q:
        q.startpermanentrun()

        prediction_dir = join(cl.PREDICTIONS_PATH, 'RiskScores', x_loader)
        db_dir = join(cl.DB_PATH, 'RiskScores')
        x_path = join(db_dir, x_loader)
        y_path = join(db_dir, 'y')

        tkttores = {}


        for rs in cl.risk_factors:
            base_path = join(prediction_dir, rs)

            for iter in range(n_iter):
                iter_path = join(base_path, str(iter))

                if full_train:
                    iter_path = join(base_path, 'full_train')

                x_n_train = cl.load_pickle(join(iter_path, 'x_train_normalizes.pickle'))
                x_n_test = cl.load_pickle(join(iter_path, 'x_test_normalizes.pickle')) if not full_train else None
                x = pd.concat([x_n_train, x_n_test], axis=0)
                model_names = ['elastic_net'] #?['tree', 'lasso', 'elastic_net']

                for model_name in model_names:
                    model_path = join(iter_path, model_name)
                    model = cl.load_pickle(join(model_path, 'model.pickle'))
                    shap_path = join(model_path, 'shap')
                    cl.mkdir_if_not_exist(shap_path)

                    tkttores[(model_name, iter, rs)] = q.method(RiskScoresShap.ShapClac, (
                        x, model, rs, iter, model_name, shap_path, 5))


        tkttores = {k: q.waitforresult(v) for k, v in tkttores.items()}



if __name__ == '__main__':
    main()
