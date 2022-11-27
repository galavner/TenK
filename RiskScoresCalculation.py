import os

import pandas as pd

from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader

import pcp_hf
import HardASCVDRiskScore
import FraminghamRiskScore


def main():

    life_style_ld = LifeStyleLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500)
    body = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                     norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5,
                                                        'remove_sigmas': 8})
    body_df = body.df.droplevel(['Date'])

    blood = BloodTestsLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                    norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5,
                                                       'remove_sigmas': 8})
    blood_df = blood.df


    ecg = ECGTextLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                             norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5,
                                                'remove_sigmas': 8})

    smoke = life_style_ld.df.smoke_tobacco_now.droplevel(['Date']).dropna()
    systolic = body_df.sitting_blood_pressure_systolic.dropna().astype('float')
    hdl = blood_df.bt__hdl_cholesterol.droplevel(['Date']).dropna()
    cholesterol = blood_df.bt__total_cholesterol.droplevel(['Date']).dropna()
    age = body.df_metadata.age.droplevel(['Date']).dropna().astype('int')

    # only people above 40 and under 80
    age = age[age > 39]
    age = age[age < 81]
    gender = body.df_metadata.gender.droplevel(['Date']).dropna().astype('int')

    pcp_body = body.df.droplevel(['Date'])[['standing_one_min_blood_pressure_systolic', 'bmi']].astype('float')
    pcp_ag = pd.concat([age, gender], axis=1)
    pcp_smoke = life_style_ld.df['smoke_tobacco_now'].droplevel(['Date'])
    pcp_smoke = pcp_smoke.apply(lambda x: 0 if x == 0 else 1)
    pcp_blood = blood.df.droplevel(['Date'])[['bt__glucose', 'bt__total_cholesterol', 'bt__hdl_cholesterol']]
    pcp_ecg = ecg.df.droplevel(['Date'])['qrs_ms']
    pcp_features = pd.concat([pcp_body, pcp_ag, pcp_blood, pcp_ecg, pcp_smoke], axis=1)
    pcp_features = pcp_features.dropna()

    ascvd = pd.concat([gender, age, cholesterol, hdl, systolic, smoke], axis=1)
    ascvd = ascvd.dropna(how='any')
    ascvd[['ascvd_percent_risk']] = ascvd.apply(HardASCVDRiskScore.HardASCVDRiskScore, axis=1, args=[1, 0, 5, 4, 2, 3],
                                                result_type="expand")

    pcp_risk = pd.Series(pcp_features.apply(lambda row: pcp_hf.PCPHF(
        age=row.age, gender=row.gender, systolic=row.standing_one_min_blood_pressure_systolic, glucose=row.bt__glucose,
        cholesterol=row.bt__total_cholesterol, hdl=row.bt__hdl_cholesterol, qrs=row.qrs_ms, bmi=row.bmi,
        smoker=row.smoke_tobacco_now)
                                            , axis=1), name='pcp_hf')

    frs = pd.concat([gender, age, cholesterol, hdl, systolic, smoke], axis=1)
    frs = frs.dropna()
    frs[['frs_score', 'frs_percent_risk']] = frs.apply(FraminghamRiskScore._calculate_framingham_risk_score, axis=1,
                                                       result_type="expand")

    print(frs, pcp_risk, ascvd)


if __name__ == '__main__':
    main()