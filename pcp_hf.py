import os
from os.path import join

import numpy as np
import pandas as pd

import config_local as cl


def PCPHF(age, gender,
          systolic, glucose, cholesterol, hdl, bmi, qrs,
          race='white', is_systolic_treated=False, smoker=0, is_glucose_treated=0):

    # Load the coefficients table
    coeff = cl.load_pickle(join(cl.DB_PATH, 'RiskScores', 'pcp_hf', 'coeff.pickle'))

    # The score of the participant
    score = 0

    # Taking the log of the different features, as instructed by the paper
    ln_age = np.log(age)
    ln_systolic = np.log(systolic)
    ln_glucose = np.log(glucose)
    ln_cholesterol = np.log(cholesterol)
    ln_hdl = np.log(hdl)
    ln_bmi = np.log(bmi)
    ln_qrs = np.log(qrs)

    # Taking the correct coefficients corresponding the sex and race
    if gender==0 and race=='white':
        coeff = coeff['white_male']
    if gender==1 and race=='white':
        coeff = coeff['white_female']
    if gender==0 and race=='black':
        coeff = coeff['black_male']
    if gender==1 and race=='black':
        coeff = coeff['black_female']

    # Adding the different features' contributions to the score
    score += ln_age * coeff['ln_age']
    score += ln_age**2 * coeff['ln_age_squared']
    score += ln_systolic * coeff['ln_sys_treated'] if is_systolic_treated \
        else ln_systolic * coeff['ln_sys_untreated']
    score += ln_age * ln_systolic * coeff['ln_age_sys_treated'] if is_systolic_treated\
        else ln_age * ln_systolic * coeff['ln_age_sys_untreated']
    score += smoker * coeff['smoker']
    score += ln_age * smoker * coeff['ln_age_smoker']
    score += ln_glucose * coeff['ln_glucose_treated'] if is_glucose_treated\
        else ln_glucose * coeff['ln_glucose_untreated']
    score += ln_cholesterol * coeff['ln_cholesterol']
    score += ln_hdl * coeff['ln_hdl']
    score += ln_bmi * coeff['ln_bmi']
    score += ln_age * ln_bmi * coeff['ln_age_bmi']
    score += ln_qrs * coeff['ln_qrs']

    # Taking the formula parameters
    mean_cv = coeff['mean_cv']
    s0 = coeff['s0']

    # The actual risk score
    risk = 1 - s0 ** (np.exp(score - mean_cv))

    return risk

