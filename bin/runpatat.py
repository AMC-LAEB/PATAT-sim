#!/usr/bin/env python3

'''
Wrapper script to run PATAT-sim and analyse results
'''

import re
import ast
import numpy as np
import json
import gzip
import os
import itertools
import sys
from scipy import sparse
import subprocess
import argparse
import pandas as pd
import datetime as dt
import numpy as np

import multiprocessing as mp
mp.set_start_method("fork", force=True)

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import PATATv1 as p1
import PATATv2 as p2

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def simulate_read_excel(input_file):
    social_entity_list = ['overseas', 'household', 'school', 'school_class', 'workplace_formal', 'workplace_informal', 'community', 'church', 'bars']
    isoquar_list = ['isolation', 'quarantine', 'hospitalised', 'self_isolation']

    all_input_dicts = {}
    for par_type in input_file['par_type'].unique():

        pt_input_file = input_file[input_file['par_type']==par_type]
        input_pars_dict = {}

        for parameter in pt_input_file['parameter'].unique():
            sub_par_df = pt_input_file[pt_input_file['parameter']==parameter]

            if len(sub_par_df) == 1:
                value = sub_par_df['value'].iloc[0]
                if isinstance(value, str):
                    if re.search('none', value, re.I):
                        value = None
                    else:
                        if re.search('^\[', value):
                            value = np.array(ast.literal_eval(value))
                input_pars_dict[parameter] = value
            else:
                if parameter == 'testing_strategies':
                    for sub_par in sub_par_df['sub_par'].unique():
                        subsub_par_df = sub_par_df[sub_par_df['sub_par']==sub_par]
                        # get index sub pars
                        if sub_par in social_entity_list:
                            sub_par = social_entity_list.index(sub_par)
                        if sub_par in isoquar_list:
                            sub_par = isoquar_list.index(sub_par)

                        for r, row in subsub_par_df.iterrows():
                            subsub_par = row.subsub_par
                            value = row.value
                            if isinstance(value, str):
                                if re.search('^\[', value):
                                    value = ast.literal_eval(value)
                            try:
                                input_pars_dict[parameter][sub_par][subsub_par] = value
                            except:
                                try:
                                    input_pars_dict[parameter][sub_par] = {subsub_par:value}
                                except:
                                    input_pars_dict[parameter] = {sub_par:{subsub_par:value}}
                else:
                    for r, row in sub_par_df.iterrows():
                        sub_par = row.sub_par
                        value = row.value
                        if isinstance(value, str):
                            if re.search('^\[', value):
                                value = np.array(ast.literal_eval(value))
                        if parameter == 'testing_sensitivity_input':
                            sub_par = ast.literal_eval(sub_par)
                        # get index sub pars
                        if sub_par in social_entity_list:
                            sub_par = social_entity_list.index(sub_par)
                        if sub_par in isoquar_list:
                            sub_par = isoquar_list.index(sub_par)
                        try:
                            input_pars_dict[parameter][sub_par] = value
                        except:
                            input_pars_dict[parameter] = {sub_par:value}
        all_input_dicts[par_type] = input_pars_dict
    return all_input_dicts

def run_sim(outfolder, input_pars_dict, total_days, start_date, version, slim, i):
    np.random.seed(i)

    ### initlalise population ###
    pop_env_setup = input_pars_dict['Population']
    if version > 1:
        pop = p2.Population(**pop_env_setup)
    else:
        pop = p1.Population(**pop_env_setup)
    individuals_df, household_contact_layer_arr, entity_type_to_ids, social_contact_layer_arr, master_school_dict = pop.initialise()

    if slim == False:
        # save household_contact_layer_arr as json
        with gzip.open('%s/%04d_household_contact_layer_arr.json.gz'%(outfolder, i), 'w') as outfile:
            json_str = json.dumps(household_contact_layer_arr, cls=NpEncoder) + "\n"
            json_bytes = json_str.encode('utf-8')
            outfile.write(json_bytes)

        # save social_contact_layer_arr as json
        with gzip.open('%s/%04d_social_contact_layer_arr.json.gz'%(outfolder, i), 'w') as outfile:
            json_str = json.dumps(social_contact_layer_arr, cls=NpEncoder) + "\n"
            json_bytes = json_str.encode('utf-8')
            outfile.write(json_bytes)

        # save entity_type_to_ids as json
        with gzip.open('%s/%04d_entity_type_to_ids.json.gz'%(outfolder, i), 'w') as outfile:
            json_str = json.dumps(entity_type_to_ids, cls=NpEncoder) + "\n"
            json_bytes = json_str.encode('utf-8')
            outfile.write(json_bytes)

    ### run simulation ###
    sim_env_setup = input_pars_dict['Simulation']
    if version > 1:
        sim = p2.Simulation(individuals_df, household_contact_layer_arr, entity_type_to_ids, social_contact_layer_arr, master_school_dict, **sim_env_setup)
    else:
        sim = p1.Simulation(individuals_df, household_contact_layer_arr, entity_type_to_ids, social_contact_layer_arr, master_school_dict, **sim_env_setup)
    sim_results = sim.execute(total_days, start_date=start_date, verbose=1)

    if version > 1:
        # version 2
        weekdays_arr, pmap_age, pmap_adults_at_risk, pmap_vacc_status, epi_seird_arr, epi_isoquar_arr, Reff_arr, asymp_infector_arr, setting_incidence_arr, exposed_day_infectee, simpop_day_of_symptom_onset, length_of_infectious_period, setting_infectee, hcf_sample_collection_day_arr, untested_non_covid_symp_lack_of_test, total_symp_testing_results, total_selftest_results, total_community_testing_results, total_exit_testing_results, total_daily_quarantine_testing_results, eod_hcf_tav_stocks, eod_test_arr, reported_daily_case_arr, untested_symp_lack_of_test_arr, vtype_infector_to_infectee, simpop_infection_status, simpop_disease_severity, simpop_ct_arr, simpop_postest_setting, simpop_isoquar_arr, border_crossing_stats, simpop_travel_days, prev_tested_entities, hcf_contact_layer_arr, simpop_agents_w_av, simpop_agents_av_benefit = sim_results

        # save results
        # save sparse arrays
        sparse.save_npz("%s/%04d_vtype_infector_to_infectee.npz"%(outfolder, i), vtype_infector_to_infectee.tocsr())
        sparse.save_npz("%s/%04d_simpop_infection_status.npz"%(outfolder, i), simpop_infection_status.tocsr())
        sparse.save_npz("%s/%04d_simpop_disease_severity.npz"%(outfolder, i), simpop_disease_severity.tocsr())
        sparse.save_npz("%s/%04d_simpop_ct_arr.npz"%(outfolder, i), simpop_ct_arr.tocsr())

        sparse.save_npz("%s/%04d_hcf_contact_layer_arr.npz"%(outfolder, i), hcf_contact_layer_arr.tocsr())
        sparse.save_npz("%s/%04d_simpop_agents_w_av.npz"%(outfolder, i), simpop_agents_w_av.tocsr())
        sparse.save_npz("%s/%04d_simpop_agents_av_benefit.npz"%(outfolder, i), simpop_agents_av_benefit.tocsr())

        if slim == True:
            np.savez("%s/%04d_sim_result_arrays.npz"%(outfolder, i), weekdays_arr=weekdays_arr,
                                                                     pmap_age=pmap_age,
                                                                     pmap_adults_at_risk=pmap_adults_at_risk,
                                                                     pmap_vacc_status=pmap_vacc_status,
                                                                     epi_seird_arr=epi_seird_arr,
                                                                     Reff_arr=Reff_arr,
                                                                     hcf_sample_collection_day_arr=hcf_sample_collection_day_arr,
                                                                     asymp_infector_arr=asymp_infector_arr,
                                                                     setting_incidence_arr=setting_incidence_arr,
                                                                     exposed_day_infectee=exposed_day_infectee,
                                                                     simpop_day_of_symptom_onset=simpop_day_of_symptom_onset,
                                                                     length_of_infectious_period=length_of_infectious_period,
                                                                     setting_infectee=setting_infectee,
                                                                     total_symp_testing_results=total_symp_testing_results,
                                                                     total_selftest_results=total_selftest_results,
                                                                     )
        else:
            np.savez("%s/%04d_sim_result_arrays.npz"%(outfolder, i), weekdays_arr=weekdays_arr,
                                                                     pmap_age=pmap_age,
                                                                     pmap_adults_at_risk=pmap_adults_at_risk,
                                                                     pmap_vacc_status=pmap_vacc_status,
                                                                     epi_seird_arr=epi_seird_arr,
                                                                     Reff_arr=Reff_arr,
                                                                     hcf_sample_collection_day_arr=hcf_sample_collection_day_arr,
                                                                     asymp_infector_arr=asymp_infector_arr,
                                                                     setting_incidence_arr=setting_incidence_arr,
                                                                     exposed_day_infectee=exposed_day_infectee,
                                                                     simpop_day_of_symptom_onset=simpop_day_of_symptom_onset,
                                                                     length_of_infectious_period=length_of_infectious_period,
                                                                     setting_infectee=setting_infectee,
                                                                     total_symp_testing_results=total_symp_testing_results,
                                                                     total_selftest_results=total_selftest_results,
                                                                     epi_isoquar_arr=epi_isoquar_arr,
                                                                     eod_hcf_tav_stocks=eod_hcf_tav_stocks,
                                                                     untested_non_covid_symp_lack_of_test=untested_non_covid_symp_lack_of_test,
                                                                     total_community_testing_results=total_community_testing_results,
                                                                     total_exit_testing_results=total_exit_testing_results,
                                                                     total_daily_quarantine_testing_results=total_daily_quarantine_testing_results,
                                                                     )

            sparse.save_npz("%s/%04d_eod_test_arr.npz"%(outfolder, i), eod_test_arr.tocsr())
            sparse.save_npz("%s/%04d_reported_daily_case_arr.npz"%(outfolder, i), reported_daily_case_arr.tocsr())
            sparse.save_npz("%s/%04d_untested_symp_lack_of_test_arr.npz"%(outfolder, i), untested_symp_lack_of_test_arr.tocsr())

            sparse.save_npz("%s/%04d_simpop_postest_setting.npz"%(outfolder, i), simpop_postest_setting.tocsr())
            sparse.save_npz("%s/%04d_simpop_isoquar_arr.npz"%(outfolder, i), simpop_isoquar_arr.tocsr())

            # save prev_tested_entities as json
            with gzip.open('%s/%04d_prev_tested_entities.json.gz'%(outfolder, i), 'w') as outfile:
                json_str = json.dumps(prev_tested_entities, cls=NpEncoder) + "\n"
                json_bytes = json_str.encode('utf-8')
                outfile.write(json_bytes)

    else:
        # version 1
        weekdays_arr, pmap_age, epi_seird_arr, epi_isoquar_arr, Reff_arr, asymp_infector_arr, setting_incidence_arr, exposed_day_infectee, simpop_day_of_symptom_onset, length_of_infectious_period, setting_infectee, hcf_sample_collection_day_arr, untested_non_covid_symp_lack_of_test, total_symp_testing_results, total_community_testing_results, total_exit_testing_results, total_daily_quarantine_testing_results, eod_test_arr, reported_daily_case_arr, untested_symp_lack_of_test_arr, vtype_infector_to_infectee, simpop_infection_status, simpop_disease_severity, simpop_ct_arr, simpop_postest_setting, simpop_isoquar_arr, border_crossing_stats, simpop_travel_days, prev_tested_entities, hcf_contact_layer_arr = sim_results

        # save results
        np.savez("%s/%04d_sim_result_arrays.npz"%(outfolder, i), weekdays_arr=weekdays_arr,
                                                                 pmap_age=pmap_age,
                                                                 epi_seird_arr=epi_seird_arr,
                                                                 epi_isoquar_arr=epi_isoquar_arr,
                                                                 Reff_arr=Reff_arr,
                                                                 asymp_infector_arr=asymp_infector_arr,
                                                                 setting_incidence_arr=setting_incidence_arr,
                                                                 exposed_day_infectee=exposed_day_infectee,
                                                                 simpop_day_of_symptom_onset=simpop_day_of_symptom_onset,
                                                                 length_of_infectious_period=length_of_infectious_period,
                                                                 setting_infectee=setting_infectee,
                                                                 hcf_sample_collection_day_arr=hcf_sample_collection_day_arr,
                                                                 untested_non_covid_symp_lack_of_test=untested_non_covid_symp_lack_of_test,
                                                                 total_symp_testing_results=total_symp_testing_results,
                                                                 total_community_testing_results=total_community_testing_results,
                                                                 total_exit_testing_results=total_exit_testing_results,
                                                                 total_daily_quarantine_testing_results=total_daily_quarantine_testing_results,)

        # save sparse arrays
        sparse.save_npz("%s/%04d_eod_test_arr.npz"%(outfolder, i), eod_test_arr.tocsr())
        sparse.save_npz("%s/%04d_reported_daily_case_arr.npz"%(outfolder, i), reported_daily_case_arr.tocsr())
        sparse.save_npz("%s/%04d_untested_symp_lack_of_test_arr.npz"%(outfolder, i), untested_symp_lack_of_test_arr.tocsr())
        sparse.save_npz("%s/%04d_vtype_infector_to_infectee.npz"%(outfolder, i), vtype_infector_to_infectee.tocsr())

        sparse.save_npz("%s/%04d_simpop_infection_status.npz"%(outfolder, i), simpop_infection_status.tocsr())
        sparse.save_npz("%s/%04d_simpop_disease_severity.npz"%(outfolder, i), simpop_disease_severity.tocsr())
        sparse.save_npz("%s/%04d_simpop_ct_arr.npz"%(outfolder, i), simpop_ct_arr.tocsr())
        sparse.save_npz("%s/%04d_simpop_postest_setting.npz"%(outfolder, i), simpop_postest_setting.tocsr())

        sparse.save_npz("%s/%04d_simpop_isoquar_arr.npz"%(outfolder, i), simpop_isoquar_arr.tocsr())

        sparse.save_npz("%s/%04d_hcf_contact_layer_arr.npz"%(outfolder, i), hcf_contact_layer_arr.tocsr())

        # save prev_tested_entities as json
        with gzip.open('%s/%04d_prev_tested_entities.json.gz'%(outfolder, i), 'w') as outfile:
            json_str = json.dumps(prev_tested_entities, cls=NpEncoder) + "\n"
            json_bytes = json_str.encode('utf-8')
            outfile.write(json_bytes)

    return

def simulate(params):
    """
    Wrapper to run PATAT simulation
    """
    pars_file = pd.read_excel(params.input)
    # read input parameters file
    input_pars_dict = simulate_read_excel(pars_file)

    # create output folders
    if params.outdir == None:
        outfname = re.sub("[^A-Za-z0-9\_\-]", "", re.sub('\.xls(x)*$', '', params.input))
        date = dt.date.today().strftime('%Y-%m-%d')
        outfolder = "./patat-sim_v%i_output_%s_%s"%(params.version, date, outfname)
    else:
        outfolder = params.outdir
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    # run bootstrap simulations in parallel
    pool = mp.Pool(processes=params.ncpu)
    results = [pool.apply_async(run_sim, args=(outfolder, input_pars_dict, params.ndays, params.start_date, params.version, params.slim, i,)) for i in np.arange(params.bootstrap)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()

    return

def read_result_folder(res_folder_name, sim_i):
    """
    Wrapper to read PATAT outputs
    """
    with np.load("%s/%04d_sim_result_arrays.npz"%(res_folder_name, sim_i)) as sim_result_arrays:

        weekdays_arr = sim_result_arrays['weekdays_arr']
        pmap_age = sim_result_arrays['pmap_age']
        epi_seird_arr = sim_result_arrays['epi_seird_arr']
        epi_isoquar_arr = sim_result_arrays['epi_isoquar_arr']
        Reff_arr = sim_result_arrays['Reff_arr']
        asymp_infector_arr = sim_result_arrays['asymp_infector_arr']
        setting_incidence_arr = sim_result_arrays['setting_incidence_arr']
        exposed_day_infectee = sim_result_arrays['exposed_day_infectee']
        simpop_day_of_symptom_onset = sim_result_arrays['simpop_day_of_symptom_onset']
        length_of_infectious_period = sim_result_arrays['length_of_infectious_period']
        setting_infectee = sim_result_arrays['setting_infectee']
        hcf_sample_collection_day_arr = sim_result_arrays['hcf_sample_collection_day_arr']
        untested_non_covid_symp_lack_of_test = sim_result_arrays['untested_non_covid_symp_lack_of_test']
        total_symp_testing_results = sim_result_arrays['total_symp_testing_results']
        total_community_testing_results = sim_result_arrays['total_community_testing_results']
        total_exit_testing_results = sim_result_arrays['total_exit_testing_results']
        total_daily_quarantine_testing_results = sim_result_arrays['total_daily_quarantine_testing_results']

    # read json files
    with gzip.open("%s/%04d_household_contact_layer_arr.json.gz"%(res_folder_name, sim_i), "r") as f:
        data = f.read()
        household_contact_layer_arr = json.loads(data.decode('utf-8'))

    with gzip.open("%s/%04d_social_contact_layer_arr.json.gz"%(res_folder_name, sim_i), "r") as f:
        data = f.read()
        social_contact_layer_arr = json.loads(data.decode('utf-8'))

    with gzip.open("%s/%04d_entity_type_to_ids.json.gz"%(res_folder_name, sim_i), "r") as f:
        data = f.read()
        entity_type_to_ids = {}
        temp_dict = json.loads(data.decode('utf-8'))
        for k, v in temp_dict.items():
            entity_type_to_ids[int(k)] = np.array(v)

    with gzip.open('%s/%04d_prev_tested_entities.json.gz'%(res_folder_name, sim_i), "r") as f:
        data = f.read()
        prev_tested_entities = {}
        temp_dict = json.loads(data.decode('utf-8'))
        for k, sub_dict in temp_dict.items():
            for sub_k, v in sub_dict.items():
                try:
                    prev_tested_entities[int(k)][int(sub_k)] = np.array(v)
                except:
                    prev_tested_entities[int(k)] = {int(sub_k):np.array(v)}

    # read sparse npz files
    eod_test_arr = sparse.load_npz("%s/%04d_eod_test_arr.npz"%(res_folder_name, sim_i))
    reported_daily_case_arr = sparse.load_npz("%s/%04d_reported_daily_case_arr.npz"%(res_folder_name, sim_i))
    untested_symp_lack_of_test_arr = sparse.load_npz("%s/%04d_untested_symp_lack_of_test_arr.npz"%(res_folder_name, sim_i))
    vtype_infector_to_infectee =sparse.load_npz("%s/%04d_vtype_infector_to_infectee.npz"%(res_folder_name, sim_i))

    simpop_infection_status = sparse.load_npz("%s/%04d_simpop_infection_status.npz"%(res_folder_name, sim_i))
    simpop_disease_severity = sparse.load_npz("%s/%04d_simpop_disease_severity.npz"%(res_folder_name, sim_i))

    simpop_ct_arr = sparse.load_npz("%s/%04d_simpop_ct_arr.npz"%(res_folder_name, sim_i))
    simpop_postest_setting = sparse.load_npz("%s/%04d_simpop_postest_setting.npz"%(res_folder_name, sim_i))
    simpop_isoquar_arr = sparse.load_npz("%s/%04d_simpop_isoquar_arr.npz"%(res_folder_name, sim_i))

    hcf_contact_layer_arr = sparse.load_npz('%s/%04d_hcf_contact_layer_arr.npz'%(res_folder_name, sim_i))

    return weekdays_arr, pmap_age, epi_seird_arr, epi_isoquar_arr, Reff_arr, asymp_infector_arr, setting_incidence_arr, exposed_day_infectee, simpop_day_of_symptom_onset, length_of_infectious_period, setting_infectee, hcf_sample_collection_day_arr, untested_non_covid_symp_lack_of_test, total_symp_testing_results, total_community_testing_results, total_exit_testing_results, total_daily_quarantine_testing_results, eod_test_arr, reported_daily_case_arr, untested_symp_lack_of_test_arr, vtype_infector_to_infectee, simpop_infection_status, simpop_disease_severity, simpop_ct_arr, simpop_postest_setting, simpop_isoquar_arr, prev_tested_entities, household_contact_layer_arr, social_contact_layer_arr, entity_type_to_ids, hcf_contact_layer_arr

def run_gs(outfolder, params, run_id):
    # read outputs in result folder
    row_results = read_result_folder(params.resfolder, run_id)

    summary_dict = {}

    # unpack result arrays
    weekdays_arr, pmap_age, epi_seird_arr, epi_isoquar_arr, Reff_arr, asymp_infector_arr, setting_incidence_arr, exposed_day_infectee, simpop_day_of_symptom_onset, length_of_infectious_period, setting_infectee, hcf_sample_collection_day_arr, untested_non_covid_symp_lack_of_test, total_symp_testing_results, total_community_testing_results, total_exit_testing_results, total_daily_quarantine_testing_results, eod_test_arr, reported_daily_case_arr, untested_symp_lack_of_test_arr, vtype_infector_to_infectee, simpop_infection_status, simpop_disease_severity, simpop_ct_arr, simpop_postest_setting, simpop_isoquar_arr, prev_tested_entities, household_contact_layer_arr, social_contact_layer_arr, entity_type_to_ids, hcf_contact_layer_arr = row_results
    summary_dict['weekdays_arr'] = weekdays_arr

    ## compute prevalence for each variant
    for i, label in zip(np.arange(epi_seird_arr.shape[-1]), ['sus', 'wt_exp', 'wt_inf', 'mt_exp', 'mt_inf', 'wt_rec', 'mt_rec', 'dea']):
        summary_dict['%s_arr'%(label)] = epi_seird_arr[:,i]

    reported_agents_arr = reported_daily_case_arr.tocoo().col
    reported_type_arr = reported_daily_case_arr.tocoo().data
    reported_day_arr = reported_daily_case_arr.tocoo().row
    # compute which and how many hospitalised specimens were collected per day
    hospitalised_agents_arr = reported_agents_arr[reported_type_arr == 2]
    hospitalised_day_arr = reported_day_arr[reported_type_arr == 2]
    hospitalised_var_arr = np.asarray(simpop_infection_status[hospitalised_day_arr,hospitalised_agents_arr]).flatten()
    hospitalised_var_arr[hospitalised_var_arr==2] = 0
    hospitalised_var_arr[hospitalised_var_arr==4] = 1
    hospitalised_hcf_arr = hcf_contact_layer_arr.tocsc()[:,hospitalised_agents_arr].tocoo().row
    # save
    summary_dict['hospitalised_agents_arr'] = hospitalised_agents_arr
    summary_dict['hospitalised_day_arr'] = hospitalised_day_arr
    summary_dict['hospitalised_var_arr'] = hospitalised_var_arr
    summary_dict['hospitalised_hcf_arr'] = hospitalised_hcf_arr

    # get infectious prevalence
    sim_tot_days = weekdays_arr.shape[0]
    day_to_inf_prev_arr = np.zeros((sim_tot_days, 2), dtype=float)
    agents_w_ct_arr = simpop_ct_arr.tocoo().col
    agents_ct_arr = simpop_ct_arr.tocoo().data
    agents_ct_day_arr = simpop_ct_arr.tocoo().row
    # get infectious agents
    mask = agents_ct_arr <= params.inf_Ct
    infectious_agents_arr = agents_w_ct_arr[mask]
    infectious_day_arr = agents_ct_day_arr[mask]
    # get their infectious status
    for day in np.unique(infectious_day_arr):
        day_mask = infectious_day_arr == day
        infectious_status_arr = simpop_infection_status.tocsc()[day,infectious_agents_arr[day_mask]].tocoo().data
        n_wt = len(infectious_status_arr[infectious_status_arr==2])
        n_mt = len(infectious_status_arr[infectious_status_arr==4])
        day_to_inf_prev_arr[day,:] = [n_wt, n_mt]

    summary_dict['wt_infectious_arr'] = day_to_inf_prev_arr[:,0]
    summary_dict['mt_infectious_arr'] = day_to_inf_prev_arr[:,1]

    ## diagnosis
    # get variant, agents and the day they were diagnosed
    diagnosed_var_arr, diagnosed_agents_arr = np.where(hcf_sample_collection_day_arr >= 0)
    diagnosed_day_arr = hcf_sample_collection_day_arr[diagnosed_var_arr, diagnosed_agents_arr]
    # get Ct values of diagnosed
    diagnosed_ct_arr = simpop_ct_arr.todok()[diagnosed_day_arr, diagnosed_agents_arr].toarray()[0]
    # filter by max Ct cut-off
    Ct_mask = diagnosed_ct_arr <= params.max_Ct
    diagnosed_var_arr = diagnosed_var_arr[Ct_mask]
    diagnosed_agents_arr = diagnosed_agents_arr[Ct_mask]
    diagnosed_day_arr = diagnosed_day_arr[Ct_mask]
    # get healthcare facilities where agents were diagnosed
    diagnosed_hcf_arr = hcf_contact_layer_arr.tocsc()[:,diagnosed_agents_arr].tocoo().row
    summary_dict['diagnosed_var_arr'] = diagnosed_var_arr
    summary_dict['diagnosed_agents_arr'] = diagnosed_agents_arr
    summary_dict['diagnosed_day_arr'] = diagnosed_day_arr
    summary_dict['diagnosed_hcf_arr'] = diagnosed_hcf_arr

    # get total number of HCFs
    all_hcf_n = len(np.unique(hcf_contact_layer_arr.tocoo().row))
    summary_dict['all_hcf_n'] = all_hcf_n

    ## get false positive samples
    falpos_agents_arr = []
    falpos_day_arr = []
    for day in np.unique(diagnosed_day_arr):
        postest_agents = simpop_postest_setting[day].tocoo().col
        diagnosed_agents = diagnosed_agents_arr[diagnosed_day_arr==day]
        # get agents who tested false positive today
        falpos_agents = np.setdiff1d(postest_agents, diagnosed_agents)
        falpos_agents_arr += list(falpos_agents)
        falpos_day_arr += [day] * len(falpos_agents)
    falpos_agents_arr = np.array(falpos_agents_arr)
    falpos_day_arr = np.array(falpos_day_arr)
    # get healthcare facilities where false positive agents were diagnosed
    falpos_hcf_arr = hcf_contact_layer_arr.tocsc()[:,falpos_agents_arr].tocoo().row
    summary_dict['falpos_agents_arr'] = falpos_agents_arr
    summary_dict['falpos_day_arr'] = falpos_day_arr
    summary_dict['falpos_hcf_arr'] = falpos_hcf_arr

    # write summary dict as json
    with gzip.open('%s/%03d_gs_summary.json.gz'%(outfolder, run_id), 'w') as outfile:
        json_str = json.dumps(summary_dict, cls=NpEncoder) + "\n"
        json_bytes = json_str.encode('utf-8')
        outfile.write(json_bytes)

    # get all hcfs
    all_hcf_arr = np.arange(all_hcf_n)
    # compute number of tertiary HCFs
    tertiary_hcf_N = np.around(params.tertiary_hcf_prop * len(all_hcf_arr)).astype(np.int32)
    comm_hcf_N = len(all_hcf_arr) - tertiary_hcf_N
    print ('number of tertiary healthcare facilities = %i'%(tertiary_hcf_N))
    print ('number of community healthcare facilities = %i'%(comm_hcf_N))

    if tertiary_hcf_N ==0:
        raise Exception('no tertiary healthcare faciliites. increase tertiary_hcf_prop.')

    # sim_N x days x seq_mode x variant type - array to save collected variants
    var_seq_arr = np.zeros((params.gs_sim_N, sim_tot_days, 8, 2), dtype=float)

    for sim_i in np.arange(params.gs_sim_N):

        # compute number of positive specimens collected from symptomatic testing patients per day
        pos_specimen_n_arr = np.array([len(diagnosed_agents_arr[diagnosed_day_arr == day]) + len(falpos_agents_arr[falpos_day_arr == day]) for day in np.arange(sim_tot_days)])
        # compute number of positive specimens collected from hospitals assuming only hospital_diagnosis_rate of all hospitalised cases are diagnosed
        diagnosed_hospitalised_agents = np.random.choice(hospitalised_agents_arr, np.around(params.hospital_diagnosis_rate * len(hospitalised_agents_arr)).astype(np.int32), replace=False)
        diagnosed_hospitalised_mask = np.isin(hospitalised_agents_arr, diagnosed_hospitalised_agents)

        diagnosed_hospitalised_agents = hospitalised_agents_arr[diagnosed_hospitalised_mask]
        diagnosed_hospitalised_var = hospitalised_var_arr[diagnosed_hospitalised_mask]
        diagnosed_hospitalised_day = hospitalised_day_arr[diagnosed_hospitalised_mask]
        diagnosed_hospitalised_hcf = hospitalised_hcf_arr[diagnosed_hospitalised_mask]
        pos_specimen_n_arr = pos_specimen_n_arr + np.array([len(diagnosed_hospitalised_agents[diagnosed_hospitalised_day == day]) for day in np.arange(sim_tot_days)])

        # get number of specimen to sample per day
        seq_N_per_day_arr = np.around(pos_specimen_n_arr * params.seq_prop/100).astype(np.int32)

        # randomly distribute community HCFs to tertiary facilities
        rand_gamma_dist = np.random.gamma(params.comm_hcf_gshape, params.comm_hcf_gscale, size=tertiary_hcf_N)
        rand_gamma_dist = rand_gamma_dist/rand_gamma_dist.sum()
        comm_hcf_per_tert_hcf = np.around(rand_gamma_dist * (len(all_hcf_arr) - tertiary_hcf_N)).astype(np.int32)
        sum_diff = comm_hcf_per_tert_hcf.sum() - comm_hcf_N
        if sum_diff < 0:
            # less than total comm hcf
            for i in np.arange(np.abs(sum_diff)):
                comm_hcf_per_tert_hcf[(np.random.choice(np.arange(tertiary_hcf_N)))] += 1
        elif sum_diff > 0:
            # more than total comm hcf
            for i in np.arange(np.abs(sum_diff)):
                selected_hcf = np.random.choice(np.arange(tertiary_hcf_N))
                selected_hcf = np.random.choice(np.arange(tertiary_hcf_N)[comm_hcf_per_tert_hcf > 0])
                comm_hcf_per_tert_hcf[selected_hcf] -= 1

        # randomly select tertiary HCFs
        tert_hcf_arr = np.sort(np.random.choice(all_hcf_arr, tertiary_hcf_N, replace=False))
        # assign comm HCFs to tertiary HCFs (with implicit distance basedo numbering)
        comm_hcf_arr = np.sort(np.setdiff1d(all_hcf_arr, tert_hcf_arr))
        tert_hcf_to_comm_hcf = {}
        prev_n = 0
        for i, n in enumerate(comm_hcf_per_tert_hcf):
            tert_hcf_to_comm_hcf[tert_hcf_arr[i]] = np.array(list(comm_hcf_arr[prev_n:prev_n+n]) + [tert_hcf_arr[i]])
            prev_n += n

        i_to_selected_tert_hcfs = {}
        for i, tert_hcf_f in enumerate([0, 0.1, 0.25, 0.5]):
            if tert_hcf_f == 0:
                selected_tert_hcf = np.array([np.random.choice(tert_hcf_arr)])
            else:
                selected_tert_hcf = np.random.choice(tert_hcf_arr, np.around(tert_hcf_f * tertiary_hcf_N).astype(np.int32), replace=False)
            i_to_selected_tert_hcfs[i] = selected_tert_hcf

        for day in np.arange(sim_tot_days):
            seq_N_today = seq_N_per_day_arr[day]
            if seq_N_today > 0:

                #### seq_mode 0: centralised sequencing ####
                # in this sequencing mode, all community facilities will send their samples to one facility only
                # get all specimens collected today and the variant type
                specimens_collected = list(diagnosed_var_arr[diagnosed_day_arr == day]) + ([-1] * len(falpos_day_arr[falpos_day_arr == day])) + list(diagnosed_hospitalised_var[diagnosed_hospitalised_day==day])
                if len(specimens_collected) == 0:
                    continue
                np.random.shuffle(specimens_collected)
                # sample specimens to be sequenced
                if len(specimens_collected) > seq_N_today:
                    sequenced_specimens = np.random.choice(specimens_collected, seq_N_today, replace=False)
                else:
                    sequenced_specimens = np.array(specimens_collected[:])
                # filter out false positives
                sequenced_specimens = sequenced_specimens[sequenced_specimens>-1]
                # sequencing success
                if len(sequenced_specimens) > 0:
                    sequenced_specimens = sequenced_specimens[np.random.random(len(sequenced_specimens))<params.seq_success_prob]
                if len(sequenced_specimens) > 0:
                    # save to var_seq_arr
                    n_wt = len(sequenced_specimens[sequenced_specimens==0])
                    n_mt = len(sequenced_specimens[sequenced_specimens==1])
                    var_seq_arr[sim_i,day,0,:] = [n_wt, n_mt]

                #### seq_mode 1: decentralised random gamma distribution of community to tertiary facilities ####
                # distribute the number of specimens today based on how many community facilities
                seq_N_today_per_tert_hcf = np.around(seq_N_today * comm_hcf_per_tert_hcf/comm_hcf_per_tert_hcf.sum()).astype(np.int32)
                sum_diff = seq_N_today_per_tert_hcf.sum() - seq_N_today
                if sum_diff < 0:
                    # less than total comm hcf
                    for i in np.arange(np.abs(sum_diff)):
                        seq_N_today_per_tert_hcf[(np.random.choice(np.arange(tertiary_hcf_N)))] += 1
                elif sum_diff > 0:
                    # more than total comm hcf
                    for i in np.arange(np.abs(sum_diff)):
                        selected_hcf = np.random.choice(np.arange(tertiary_hcf_N)[seq_N_today_per_tert_hcf > 0])
                        seq_N_today_per_tert_hcf[selected_hcf] -= 1

                # get all specimens collected by each tertiary facility
                for i, curr_tert_hcf in enumerate(tert_hcf_arr):
                    curr_comm_hcfs = tert_hcf_to_comm_hcf[curr_tert_hcf]
                    seq_N_today_tert_hcf = seq_N_today_per_tert_hcf[i]

                    specimens_collected = list(diagnosed_var_arr[(diagnosed_day_arr == day)&(np.isin(diagnosed_hcf_arr, curr_comm_hcfs))])
                    specimens_collected += [-1] * len(falpos_day_arr[(falpos_day_arr == day)&(np.isin(falpos_hcf_arr, curr_comm_hcfs))])
                    specimens_collected += list(diagnosed_hospitalised_var[(diagnosed_hospitalised_day==day)&(np.isin(diagnosed_hospitalised_hcf, curr_comm_hcfs))])
                    if len(specimens_collected) == 0:
                        continue
                    np.random.shuffle(specimens_collected)

                    # sample specimens to be sequenced
                    if len(specimens_collected) > seq_N_today_tert_hcf:
                        sequenced_specimens = np.random.choice(specimens_collected, seq_N_today_tert_hcf, replace=False)
                    else:
                        sequenced_specimens = np.array(specimens_collected[:])
                    # filter out false positives
                    sequenced_specimens = sequenced_specimens[sequenced_specimens>-1]
                    if len(sequenced_specimens) > 0:
                        sequenced_specimens = sequenced_specimens[np.random.random(len(sequenced_specimens))<params.seq_success_prob]
                    if len(sequenced_specimens) > 0:
                        # save to var_seq_arr
                        n_wt = len(sequenced_specimens[sequenced_specimens==0])
                        n_mt = len(sequenced_specimens[sequenced_specimens==1])

                        var_seq_arr[sim_i,day,1,0] += n_wt
                        var_seq_arr[sim_i,day,1,1] += n_mt

                ### seq_mode 2: only sequence whatever samples that are collected at tertiary facilities ###
                specimens_collected = list(diagnosed_var_arr[(diagnosed_day_arr == day)&(np.isin(diagnosed_hcf_arr, tert_hcf_arr))])
                specimens_collected += [-1] * len(falpos_day_arr[(falpos_day_arr == day)&(np.isin(falpos_hcf_arr, tert_hcf_arr))])
                specimens_collected += list(diagnosed_hospitalised_var[(diagnosed_hospitalised_day==day)])
                if len(specimens_collected) == 0:
                    continue
                np.random.shuffle(specimens_collected)
                # sample specimens to be sequenced
                if len(specimens_collected) > seq_N_today:
                    sequenced_specimens = np.random.choice(specimens_collected, seq_N_today, replace=False)
                else:
                    sequenced_specimens = np.array(specimens_collected[:])
                # filter out false positives
                sequenced_specimens = sequenced_specimens[sequenced_specimens>-1]
                if len(sequenced_specimens) > 0:
                    sequenced_specimens = sequenced_specimens[np.random.random(len(sequenced_specimens))<params.seq_success_prob]
                if len(sequenced_specimens) > 0:
                    # save to var_seq_arr
                    n_wt = len(sequenced_specimens[sequenced_specimens==0])
                    n_mt = len(sequenced_specimens[sequenced_specimens==1])
                    var_seq_arr[sim_i,day,2,:] = [n_wt, n_mt]

                ### seq_mode 3: only sequence hospitalised samples that are collected at tertiary facilities ###
                specimens_collected = list(diagnosed_hospitalised_var[(diagnosed_hospitalised_day==day)])
                if len(specimens_collected) == 0:
                    continue
                np.random.shuffle(specimens_collected)
                # sample specimens to be sequenced
                if len(specimens_collected) > seq_N_today:
                    sequenced_specimens = np.random.choice(specimens_collected, seq_N_today, replace=False)
                else:
                    sequenced_specimens = np.array(specimens_collected[:])
                # filter out false positives
                sequenced_specimens = sequenced_specimens[sequenced_specimens>-1]
                if len(sequenced_specimens) > 0:
                    sequenced_specimens = sequenced_specimens[np.random.random(len(sequenced_specimens))<params.seq_success_prob]
                if len(sequenced_specimens) > 0:
                    # save to var_seq_arr
                    n_wt = len(sequenced_specimens[sequenced_specimens==0])
                    n_mt = len(sequenced_specimens[sequenced_specimens==1])
                    var_seq_arr[sim_i,day,3,:] = [n_wt, n_mt]

                ### seq_mode 4 onwards (5, 6, 7): different number of hospitals will only sequence the samples they collected
                for i, tert_hcf_f in enumerate([0, 0.1, 0.25, 0.5]):
                    selected_tert_hcf = i_to_selected_tert_hcfs[i]

                    specimens_collected = list(diagnosed_var_arr[(diagnosed_day_arr == day)&(np.isin(diagnosed_hcf_arr, selected_tert_hcf))])
                    specimens_collected += [-1] * len(falpos_day_arr[(falpos_day_arr == day)&(np.isin(falpos_hcf_arr, selected_tert_hcf))])
                    specimens_collected += list(diagnosed_hospitalised_var[(diagnosed_hospitalised_day==day)&(np.isin(diagnosed_hospitalised_hcf,selected_tert_hcf))])
                    if len(specimens_collected) == 0:
                        continue
                    np.random.shuffle(specimens_collected)
                    # sample specimens to be sequenced
                    if len(specimens_collected) > seq_N_today:
                        sequenced_specimens = np.random.choice(specimens_collected, seq_N_today, replace=False)
                    else:
                        sequenced_specimens = np.array(specimens_collected[:])
                    # filter out false positives
                    sequenced_specimens = sequenced_specimens[sequenced_specimens>-1]
                    if len(sequenced_specimens) > 0:
                        sequenced_specimens = sequenced_specimens[np.random.random(len(sequenced_specimens))<params.seq_success_prob]
                    if len(sequenced_specimens) > 0:
                        # save to var_seq_arr
                        n_wt = len(sequenced_specimens[sequenced_specimens==0])
                        n_mt = len(sequenced_specimens[sequenced_specimens==1])
                        var_seq_arr[sim_i,day,4+i,:] = [n_wt, n_mt]

    np.savez("%s/%03d_seqprop%03d_gs-sim_arr.npz"%(outfolder, run_id, params.seq_prop), var_seq_arr=var_seq_arr)

    return

def genomesurv(params):
    np.seterr(all='raise')

    # consolidate all sim_i in folder
    run_id_arr = np.sort([int(re.search("^(\d+)_sim_result_arrays.npz", fname).group(1)) for fname in os.listdir(params.resfolder) if re.search("^(\d+)_sim_result_arrays.npz", fname)])

    # create output folders
    if params.outdir == None:
        date = dt.date.today().strftime('%Y-%m-%d')
        outfolder = "./patat-sim_gs_%s_%s"%(date, re.sub("(^.+/|/$)", "", params.resfolder))
    else:
        outfolder = params.outdir
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    # run simulations in parallel
    pool = mp.Pool(processes=params.ncpu)
    results = [pool.apply_async(run_gs, args=(outfolder, params, run_id,)) for run_id in run_id_arr]
    output = [p.get() for p in results]
    pool.close()
    pool.join()

    return

def make_parser():
    """
    Make argument parser
    """
    version = "2.0"

    parser = argparse.ArgumentParser(description='PATAT simulator (v%s)'%(version))
    subparsers = parser.add_subparsers()

    # simulation
    sim_parser = subparsers.add_parser('simulate', description='run PATAT simulation')
    sim_parser.add_argument('--input', type = str,  help='input parameter file (excel format)')
    sim_parser.add_argument('--ndays', type = int, default = 30, help='length of period to simulate in days')
    sim_parser.add_argument('--start_date', type = str, default = '2021-01-01', help = 'start date of simulation')
    sim_parser.add_argument('--bootstrap', type = int, default=1, help='number of boostrap runs')
    sim_parser.add_argument('--ncpu', type = int, default = 1, help='number of threads to run bootstrap runs in parallel')
    sim_parser.add_argument('--slim', default=False, action='store_true', help='save only essential simulation outputs.')
    sim_parser.add_argument('--outdir', type = str, help='optional output directory path')
    sim_parser.add_argument('--version', type = int, default=1, help='version of PATAT to run.')
    sim_parser.set_defaults(func=simulate)

    # genomic surveillance
    gs_parser = subparsers.add_parser('gs', description='run genomic surveillance simulation')
    gs_parser.add_argument('--resfolder', type = str, help='PATAT output result folder path')
    gs_parser.add_argument('--max_Ct', type = int, default = 30, help='max Ct value of specimen that could be sequenced')
    gs_parser.add_argument('--inf_Ct', type = int, default = 30, help='infectious Ct value threshold')
    gs_parser.add_argument('--seq_prop', type = int, default = 10, help='Percentage of collected specimens to sequence')
    gs_parser.add_argument('--seq_success_prob', type=float, default=0.8, help='seqeuncing success probability')
    gs_parser.add_argument('--tertiary_hcf_prop', type=float, default=0.2, help='proportion of healthcare facilities that are tertiary facilities')
    gs_parser.add_argument('--comm_hcf_gshape', type=float, default=2.0, help='gamma distribution shape factor for how community healthcare facilities are linked to tertiary facilities')
    gs_parser.add_argument('--comm_hcf_gscale', type=float, default=2.0, help='gamma distribution scale factor for how community healthcare facilities are linked to tertiary facilities')
    gs_parser.add_argument('--hospital_diagnosis_rate', type=float, default=0.3, help='proportion of hospitalized cases that were diagnosed for COVID-19')
    gs_parser.add_argument('--gs_sim_N', type=int, default = 100, help='number of boostrap genomic surveillance simulations to perform')
    gs_parser.add_argument('--ncpu', type = int, default = 1, help='number of threads for parallel computation')
    gs_parser.add_argument('--outdir', type = str, help='optional output directory path')
    gs_parser.set_defaults(func=genomesurv)

    return parser

def main():
    # parse arguments
    parser = make_parser()
    params = parser.parse_args()
    # run function
    if params == argparse.Namespace():
        parser.print_help()
        return_code = 0
    else:
        return_code = params.func(params)

    sys.exit(return_code)

if __name__ == '__main__':
    main()
