import pandas as pd


def fix_indices(loader, index_as_registration_code: bool = False):
    if index_as_registration_code:
        loader_df = set_index_to_registration_code(loader)
    else:
        loader_df = loader.df
    return loader_df


def set_index_to_registration_code(loader):
    loader_df = loader.df
    loader_metadata = loader.df_metadata
    loader_df = loader_df.set_index(loader_metadata.RegistrationCode.drop_duplicates())
    return loader_df



def get_BodyMeasuresLoaderDF():
    from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
    body = BodyMeasuresLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                     norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5,
                                                        'remove_sigmas': 8})
    return body


def get_BloodTestsLoaderDF():
    from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
    blood = BloodTestsLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                    norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5,
                                                       'remove_sigmas': 8})
    return blood


def get_SerumMetabolomicsLoader(index_as_registration_code: bool = True):
    from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
    sm = SerumMetabolomicsLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                                norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5, 'remove_sigmas': 8}
                                            , precomputed_loader_fname='metab_10k_data_RT_clustering')
    sm_df = fix_indices(sm, index_as_registration_code)
    return sm_df




def get_GutMBLoader(index_as_registration_code: bool = True):
    from LabData.DataLoaders.GutMBLoader import GutMBLoader
    mb = GutMBLoader().get_data(df = 'segal_species', study_ids=[10, 1001], take_log=True, min_col_present_frac=0.2,
                            research_stafe='baseline', groupby_reg='first', min_col_present=500,
                            norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5, 'remove_sigmas': 8})
    mb_df = fix_indices(mb, index_as_registration_code)
    return mb_df


def get_physical_activity():
    phys_act = pd.read_csv('/net/mraid08/export/genie/LabData/Analyses/galavner/DB/physical_activity.csv')
    phys_act.index = phys_act.participant_id
    phys_act = phys_act.drop(columns=['participant_id', phys_act.columns[0]])
    return phys_act


def get_ABILoader():
    from LabData.DataLoaders.ABILoader import ABILoader
    abi = ABILoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                                                norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5,
                                                                   'remove_sigmas': 8})
    return abi


def get_ECGTextLoader():
    from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
    ecg = ECGTextLoader().get_data(study_ids='10K', groupby_reg='first', min_col_present=500,
                             norm_dist_capping={'sample_size_frac': 0.95, 'clip_sigmas': 5,
                                                'remove_sigmas': 8})
    return ecg

