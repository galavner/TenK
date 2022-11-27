import os

import pickle
import numpy as np
import pandas as pd

from LabUtils.addloglevels import sethandlers
from LabQueue.qp import qp, fakeqp

import config_local as cl
import LGBM

sethandlers()

out_dir = os.path.join(cl.PREDICTIONS_PATH, 'SM_to_FRS')
x_path = os.path.join(out_dir, 'x_df.pickle')
y_path = os.path.join(out_dir, 'y_df.pickle')

# def SendToLGBM(x_path: os.path = x_path, y_path: os.path = y_path, out_dir: os.path =out_dir, do_permutations: bool = True):
os.chdir(out_dir)


with qp(jobname='Gal_LGBM', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=500,
        _mem_def=6) as q:
    q.startpermanentrun()
    x_df = cl.load_pickle(x_path)
    y_df = cl.load_pickle(y_path)
    tkttores = {}
    indx = []

    for i, y_col in enumerate(y_df.columns):
        y = y_df[y_col].dropna()
        x = x_df.reindex(y.index).dropna(how='all')
        y = y.reindex(x.index)

        if x.shape[0] < 500:
            continue

        tkttores[(i, y_col)] = q.method(LGBM.LGBMPredict, (x, y, out_dir, True))
        indx.append(y_col)

    tkttores = {k: q.waitforresult(v) for k, v in tkttores.items()}

    results = pd.DataFrame(tkttores).T
    results.columns = ['x_train', 'x_test', 'y_train', 'y_test', 'y_pred', 'model',
                       'results_dict', 'permutation_results', 'is_signal']
    results.index = indx

    cl.save_pickle(results, out_dir, 'all_combined.pickle')

# return results
