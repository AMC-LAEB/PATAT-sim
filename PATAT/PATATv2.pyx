#cython: language_level=3
import re
import numpy as np
np.seterr(invalid='raise')

import pandas as pd
from datetime import datetime

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np

from scipy import sparse
from scipy.interpolate import CubicHermiteSpline
from scipy.stats import lognorm, truncnorm, gamma, expon

# define c type alias
ctypedef long int32
ctypedef signed char int8

# version 0.1 (23-Feb-2022)

"""
Common (global) parameters
"""

cdef np.ndarray global_age_bins = np.arange(0, 100, 5, dtype=np.int32) # corresponding age bins

ctypedef enum Vaccination_Status:
    unvaccinated,
    partial_vacc,
    full_vacc,
    boosted_vacc

ctypedef enum Infection_Status:
    susceptible,
    exposed_wt, # incubation
    infected_wt, # infectious
    exposed_mt, # incubation
    infected_mt, # infectious
    recovered_wt,
    recovered_mt,
    death

ctypedef enum Disease_Severity:
    healthy, # susceptible
    asymptomatic, # infected
    presymptomatic, # infected
    mild, # infected
    severe # assumed hospitalised

ctypedef enum Social_Entity:
    overseas, # 0
    household, # 1
    school, # 2
    school_class, # 3
    workplace_formal, # 4
    workplace_informal, # 5
    community, # 6
    church, # 7
    bars # 8

ctypedef enum IsoQuar_Type:
    isolation,
    quarantine,
    hospitalised, # hospitalisation of agents presenting severe symptoms
    self_isolation # self-isolation of agents presenting mild symptoms

ctypedef enum Disease_Periods:
    exposed_T,
    presymp_T,
    asymp_rec_T,
    mild_rec_T,
    symp_to_sev_T,
    sev_to_recdea_T,

### --- functions --- ###
def neg_binomial(float mean, float shape, int32 n, float step=0.1):
    '''
    Return random negative binomial samples; step size < 1 is used to return non-integer outputs
    '''
    cdef float nbn_n = shape # correct?
    cdef float nbn_p = shape/(mean/step + shape)
    cdef np.ndarray samples = np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n)*step
    return samples

def get_rand_lognormal_dur(np.ndarray par_arr, int32 N=-1, np.ndarray quantile_arr=np.array([]), int32 min_val=-1):
    cdef float par1 = par_arr[0]
    cdef float par2 = par_arr[1]
    cdef float mean  = np.log(par1**2 / np.sqrt(par2**2 + par1**2)) # Computes the mean of the underlying normal distribution
    cdef float sigma = np.sqrt(np.log(par2**2/par1**2 + 1)) # Computes sigma for the underlying normal distribution

    cdef object distgen = lognorm(s = sigma, scale = np.exp(mean))
    cdef np.ndarray rand_samples, samp_quantile
    if len(quantile_arr) > 0:
        N = len(quantile_arr)
        # quantile array is given, so sample from the same quantile
        rand_samples = np.around(distgen.ppf(quantile_arr), 0).astype(np.int32)
        if min_val > -1: # minimum positive value required
            rand_samples[rand_samples<min_val] = min_val
        return rand_samples
    else:
        # if quantile_arr is not given, generate random samples and corresponding quantile of samples
        rand_samples = distgen.rvs(random_state=np.random.default_rng(), size=N)
        rand_samples = np.around(rand_samples, 0).astype(np.int32) # round samples to whole numbers
        if min_val > -1: # minimum positive value required
            rand_samples[rand_samples<min_val] = min_val
        samp_quantile = distgen.cdf(rand_samples)
        return rand_samples, samp_quantile

def get_halfnormal_int_sample(float mean, float std, int32 size):
    cdef float a = -mean / std
    cdef float b = ((mean + 5 * std) - mean) / std
    return np.around(truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=np.random.default_rng())).astype(np.int32)

### --- Population class --- ###
@cython.no_gc_clear
cdef class Population:

    # declare variables
    # general population and household structure
    cdef int32 pop_size, min_prime_adult_age
    cdef float women_head_prop, women_prop, mean_household_size
    cdef object household_to_agents
    cdef np.ndarray age_structure, households_arr, population_arr
    cdef np.ndarray pmap_age, pmap_sex, pmap_household_head, pmap_households

    # schools
    cdef int32 entity_id_counter
    cdef object school_age_range, entity_type_to_ids, entity_to_agents, school_to_classes, school_to_teachers_n, school_type_to_schools
    cdef np.ndarray schooling_rate, mean_class_size, mean_school_size, student_teacher_ratio, school_gender_parity

    # employment
    cdef int32 min_working_age
    cdef float mean_employment_contacts_formal, mean_employment_contacts_informal
    cdef np.ndarray employment_rate, employment_rate_formal

    # additional contact layers
    cdef object church_layer_params

    # individual dict
    cdef object individuals_df

    def __init__(self, **kwargs):
        # population structure
        # Population size (number of individuals) (default 20 persons)
        self.pop_size = kwargs.get("pop_size", 1e5)
        # age structure; stratified by age in 5 year bins
        # source (zambia): https://www.zamstats.gov.zm/index.php/publications/category/7-labour
        self.age_structure = kwargs.get("age_structure", np.array([0.161,0.165,0.157,0.101,0.083,0.068,0.057,0.051,0.042,0.030,0.024,0.015,0.016,0.009,0.008,0.005,0.006,0.002,0.000,0.000]))
        # min prime adult age (at least one prime adult must be in found in each household)
        self.min_prime_adult_age = kwargs.get("min_prime_adult_age", 20)
        # proportion of women
        # source (zambia): https://www.zamstats.gov.zm/index.php/publications/category/7-labour
        self.women_prop = kwargs.get("women_prop", 0.51)
        self.women_head_prop = kwargs.get('women_head_prop', 0.27) # proportion of women who are head of households
        # household size distribution (https://population.un.org/Household/index.html#/countries/894)
        self.mean_household_size = kwargs.get("mean_household_size", 5.1)

        # employment
        # source (zambia): https://www.zamstats.gov.zm/index.php/publications/category/7-labour
        self.min_working_age = kwargs.get("min_working_age", 15) # minimum working age (anyone older than this age is considered to be eligible to be in the employment
        self.employment_rate = kwargs.get("employment_rate", np.array([0.39, 0.23])) # employment participation rate (men, women) in working-age population
        self.employment_rate_formal = kwargs.get("employment_rate_formal", np.array([0.36, 0.24])) # proportion of employed (men, women) in formal employment
        self.mean_employment_contacts_formal = kwargs.get("mean_employment_contacts_formal", 20) # mean number of contacts per person in a formal employment setting
        self.mean_employment_contacts_informal = kwargs.get("mean_employment_contacts_informal", 5) # mean number of contacts per person in an informal employment setting

        # (K-12) schools
        # Zambian primary school (0) 7-13, secondary school (1) 14-18
        self.school_age_range = kwargs.get("school_age_range", {0:[7,13], 1:[14,18]})
        # schooling rate (primary, secondary) - source: https://www.zamstats.gov.zm/index.php/publications/category/8-demorgraphy?download=364:zambia-demographic-and-health-survey-2018
        self.schooling_rate = kwargs.get("schooling_rate", np.array([0.79, 0.40]))
        self.school_gender_parity = kwargs.get("school_gender_parity", np.array([1., 0.9])) # gender parity (primary, secondary)
        self.mean_class_size = kwargs.get("mean_class_size", np.array([37, 37])) # mean class size (primary, secondary)
        self.mean_school_size = kwargs.get("mean_school_size", np.array([700, 700])) # mean school size (primary, secondary)
        self.student_teacher_ratio = kwargs.get("student_teacher_ratio", np.array([42, 42])) # # of students per teacher (primary, secondary)

        # additional contact layers (church)
        # age_sex_prop (empty = all age and gender), sampling_prop
        self.church_layer_params = kwargs.get("church_layer_params", {'bool':1, 'sampling_prop':0.7, 'entity_size':{'mean':500, 'sd':100}})

        return

    def initialise_individual(self, int32 agent_id, int32 agent_age_bin, int32 household_id, float women_prop, int32 household_head_bool):
        """
        Initialise individual
        """
        cdef object ind_info = {"agent_age": np.random.choice(range(global_age_bins[agent_age_bin], global_age_bins[agent_age_bin]+5)),
                                "agent_age_bin":global_age_bins[agent_age_bin],
                                "sex":int(np.random.random() < women_prop),
                                "household_id":household_id,
                                "household_head":household_head_bool,
                                "social_id_arr":[], # list of social contact layers (schools, workplaces, etc.)
                                "student_bool":0, # boolean to denote if individual is schooling
                                "school_bool":0, # boolean to denote if individual is part of school contact networks
                                "formal_employed_bool":0, # boolean to denote if individual is employed formally
                                "informal_employed_bool":0,
                                "non_teacher_employed_bool":0,
                                }
        self.individuals_df[agent_id] = ind_info

        try:
            self.household_to_agents[household_id].append(agent_id)
        except:
            self.household_to_agents[household_id] = [agent_id]

        return

    def initialise_population(self):
        """
        Initialise individuals and households
        """

        # create household size array
        cdef np.ndarray household_N_arr = np.random.poisson(self.mean_household_size, size=int(np.ceil(self.pop_size/self.mean_household_size)))
        household_N_arr = np.delete(household_N_arr, np.where(household_N_arr==0)[0]) # remove any household size entries that are zero
        # compute difference between sum of household_N_arr and required pop_size
        cdef int32 household_N_diff = household_N_arr.sum() - self.pop_size

        cdef int32 household_N_arr_sum
        # if difference is < 0, add more individuals until we have excess
        while household_N_diff < 0:
            household_N_arr = np.array(list(household_N_arr)+list(np.random.poisson(self.mean_household_size, size=int(np.ceil(abs(household_N_diff)/self.mean_household_size)))))
            household_N_arr = np.delete(household_N_arr, np.where(household_N_arr==0)[0])
            household_N_arr_sum = household_N_arr.sum()
            household_N_diff = household_N_arr_sum - self.pop_size

        # remove households with excess individuals
        while household_N_diff > 0:
            household_N_arr = household_N_arr[1:]
            household_N_arr_sum = household_N_arr.sum()
            household_N_diff = household_N_arr_sum - self.pop_size

        if household_N_diff < 0: # add remaining difference
            household_N_arr = np.array(list(household_N_arr)+[np.abs(household_N_diff)])

        #household_N_arr = np.sort(household_N_arr)[::-1] # reverse sort household_N_arr by size

        # create holder of household IDs
        self.households_arr = np.arange(len(household_N_arr), dtype=np.int32)

        print ("Populating households (n=%i) with %i agents..."%(len(household_N_arr), self.pop_size))

        # get population age bins based on given structure
        cdef np.ndarray rand_popN_by_agebins = np.random.multinomial(self.pop_size, pvals=self.age_structure).astype(np.int32)
        # calculate number of prime adults
        cdef int32 prime_adults_startidx = np.where(global_age_bins==self.min_prime_adult_age)[0][0]
        cdef np.ndarray prime_adults_popN_by_agebins = rand_popN_by_agebins[prime_adults_startidx:]
        cdef int32 prime_adults_n = prime_adults_popN_by_agebins.sum()

        if prime_adults_n < len(household_N_arr):
            raise Exception("Not enough prime adults (n=%i) to be household heads."%(prime_adults_n))

        cdef int32 household_id, household_N, agent_age_bin, i
        cdef int32 agent_id_counter = -1
        cdef np.ndarray household_prime_adult_agebin_arr = np.zeros(len(self.households_arr)).astype(np.int32)

        # first populate each household with a prime age adult
        for household_id in self.households_arr:
            prime_adults_popN_by_agebins = rand_popN_by_agebins[prime_adults_startidx:]
            # randomly select agent age (by idx)
            agent_age_bin = np.random.choice(np.arange(prime_adults_startidx, len(self.age_structure)), p=prime_adults_popN_by_agebins/prime_adults_popN_by_agebins.sum())

            household_prime_adult_agebin_arr[household_id] = agent_age_bin

            # update population age bind
            rand_popN_by_agebins[agent_age_bin] -= 1

        # compute proportion of non-head of households who are women
        cdef float women_non_head_prop = ((self.women_prop * self.pop_size) - (self.women_head_prop * len(self.households_arr))) / (self.pop_size - len(self.households_arr))

        # then we populate the rest of the household members
        cdef int32 divisor = np.ceil(len(household_N_arr)/5).astype(np.int32)
        for household_id, household_N in enumerate(household_N_arr):
            if (household_id+1)%divisor == 0:
                print ("...%.0f%%..."%(100*(household_id+1)/len(household_N_arr)))

            # add and initialise dedicated prime adult as head of household first
            agent_id_counter += 1
            self.initialise_individual(agent_id_counter, household_prime_adult_agebin_arr[household_id], household_id, self.women_head_prop, 1)

            for i in np.arange(household_N-1):
                agent_id_counter += 1
                # randomly select agent age (by idx)
                agent_age_bin = np.random.choice(np.arange(len(global_age_bins)), p=rand_popN_by_agebins/rand_popN_by_agebins.sum())
                self.initialise_individual(agent_id_counter, agent_age_bin, household_id, women_non_head_prop, 0)

                # update population age bind
                rand_popN_by_agebins[agent_age_bin] -= 1

        print ("...done.")

        return

    def create_schools(self, np.ndarray participant_ids, float mean_class_size, float mean_school_size, float student_teacher_ratio):
        """
        Create and populate schools
        """
        cdef int32 age, curr_age_N, entity_N_diff, entity_N_arr_sum
        cdef object curr_age_participants_id, entity_N_arr
        cdef np.ndarray age_of_participants = self.pmap_age[participant_ids]
        cdef object entity_ids_arr

        cdef object age_to_class_size_dict = {} # dictionary to store class ids for each age group
        cdef object age_to_participant_ids = {} # dictionary to store participants of given age
        cdef int32 max_col_len = 0
        cdef np.ndarray sorted_unique_age_arr = np.sort(np.unique(age_of_participants))

        for age in sorted_unique_age_arr:
            curr_age_participants_id = participant_ids[age_of_participants==age]
            curr_age_N = len(curr_age_participants_id) # number of individuals of current age
            age_to_participant_ids[age] = list(curr_age_participants_id)

            # get class size array for each age
            entity_N_arr = np.random.poisson(mean_class_size, size=int(np.ceil(curr_age_N/mean_class_size)))
            entity_N_arr = np.delete(entity_N_arr, np.where(entity_N_arr==0)[0]) # remove any class with zero individuals
            # difference between class size array sum and curr_age_N
            entity_N_diff = entity_N_arr.sum() - curr_age_N

            # if difference is < 0, add more individuals until we have excess
            while entity_N_diff < 0:
                entity_N_arr = np.array(list(entity_N_arr)+list(np.random.poisson(mean_class_size, size=int(np.ceil(abs(entity_N_diff)/mean_class_size)))))
                entity_N_arr = np.delete(entity_N_arr, np.where(entity_N_arr==0)[0])
                entity_N_arr_sum = entity_N_arr.sum()
                entity_N_diff = entity_N_arr_sum - curr_age_N

            # remove classes with excess individuals
            while entity_N_diff > 0:
                entity_N_arr = entity_N_arr[1:]
                entity_N_arr_sum = entity_N_arr.sum()
                entity_N_diff = entity_N_arr_sum - curr_age_N

            if entity_N_diff < 0: # add remaining difference
                entity_N_arr = np.array(list(entity_N_arr)+[np.abs(entity_N_diff)])

            # save class size array to each age
            age_to_class_size_dict[age] = entity_N_arr
            if len(entity_N_arr) > max_col_len:
                max_col_len = len(entity_N_arr)

        # create schools which has equitable distribution of classes spanning all ages
        cdef np.ndarray age_to_class_size_arr = np.zeros((len(sorted_unique_age_arr), max_col_len), dtype=np.int32)
        cdef int32 i, j, N, agent_id
        # create matrix (rows = age, col = classes)
        for i, age in enumerate(sorted_unique_age_arr):
            entity_N_arr = age_to_class_size_dict[age]
            age_to_class_size_arr[i,:len(entity_N_arr)] = entity_N_arr
        # determine breakpoints in class distribution to form schools
        cdef object flatten_age_arr, flatten_age_to_class_size_arr
        flatten_age_arr = []
        flatten_age_to_class_size_arr = []
        for j in range(max_col_len):
            for i in range(len(sorted_unique_age_arr)):
                N = age_to_class_size_arr[i,j]
                flatten_age_arr.append(sorted_unique_age_arr[i])
                flatten_age_to_class_size_arr.append(N)

        # create schools
        cdef int32 prev_i = 0
        cdef int32 prev_age_j, school_id, class_id, school_size
        cdef object school_age_to_participants_id, school_to_age_class_size_arr
        cdef object school_id_list = []

        for i in np.arange(1, len(flatten_age_to_class_size_arr)+1):
            school_size = sum(flatten_age_to_class_size_arr[prev_i:i])
            if school_size >= mean_school_size:
                # create new school
                self.entity_id_counter += 1 # create entity (school)
                school_id = self.entity_id_counter
                school_id_list.append(school_id)
                # save entity ID to social entity type
                try:
                    self.entity_type_to_ids[school].append(school_id)
                except:
                    self.entity_type_to_ids[school] = [school_id]
                # create school_to_classes
                self.school_to_classes[school_id] = []
                # save number of teachers for each school
                self.school_to_teachers_n[school_id] = np.ceil(school_size/student_teacher_ratio).astype(np.int32)

                school_age_to_participants_id = {}
                school_to_age_class_size_arr = {}
                for age, N in zip(flatten_age_arr[prev_i:i], flatten_age_to_class_size_arr[prev_i:i]):
                    if N > 0:
                        # get remaining participants of age
                        curr_age_participants_id = age_to_participant_ids[age][:N][:]
                        # update age_to_participant_ids removing participants added to school
                        age_to_participant_ids[age] = sorted(set(age_to_participant_ids[age])-set(curr_age_participants_id))

                        # add participants to school
                        try:
                            school_age_to_participants_id[age] += curr_age_participants_id
                        except:
                            school_age_to_participants_id[age] = curr_age_participants_id.copy()
                        # add class size for age
                        try:
                            school_to_age_class_size_arr[age].append(N)
                        except:
                            school_to_age_class_size_arr[age] = [N]

                # populate classes in chool
                for age, curr_age_participants_id in school_age_to_participants_id.items():
                    entity_N_arr = school_to_age_class_size_arr[age]
                    curr_age_participants_id = np.array(curr_age_participants_id)
                    # randomise participants
                    np.random.shuffle(curr_age_participants_id)

                    prev_age_j = 0
                    for N in entity_N_arr:
                        self.entity_id_counter += 1 # create entity (class)
                        class_id = self.entity_id_counter
                        # save entity ID to social entity type
                        try:
                            self.entity_type_to_ids[school_class].append(class_id)
                        except:
                            self.entity_type_to_ids[school_class] = [class_id]
                        # apped class_id to school
                        self.school_to_classes[school_id].append(class_id)

                        for agent_id in curr_age_participants_id[prev_age_j:prev_age_j+N]:
                            # assign agent to class
                            self.individuals_df[agent_id]["social_id_arr"].append(class_id)
                            try:
                                self.entity_to_agents[class_id].append(agent_id)
                            except:
                                self.entity_to_agents[class_id] = [agent_id]
                            self.individuals_df[agent_id]["student_bool"] = 1
                            self.individuals_df[agent_id]["school_bool"] = 1
                        prev_age_j += N
                prev_i = i

        # create new school
        self.entity_id_counter += 1 # create entity (school)
        school_id = self.entity_id_counter
        school_id_list.append(school_id)
        # save entity ID to social entity type
        try:
            self.entity_type_to_ids[school].append(school_id)
        except:
            self.entity_type_to_ids[school] = [school_id]
        # create school_to_classes
        self.school_to_classes[school_id] = []
        # school size
        school_size = sum(flatten_age_to_class_size_arr[prev_i:])
        # save number of teachers for each school
        self.school_to_teachers_n[school_id] = np.ceil(school_size/student_teacher_ratio).astype(np.int32)

        school_age_to_participants_id = {}
        school_to_age_class_size_arr = {}
        for age, N in zip(flatten_age_arr[prev_i:], flatten_age_to_class_size_arr[prev_i:]):
            if N > 0:
                # get remaining participants of age
                curr_age_participants_id = age_to_participant_ids[age][:N][:]
                # update age_to_participant_ids removing participants added to school
                age_to_participant_ids[age] = sorted(set(age_to_participant_ids[age])-set(curr_age_participants_id))

                # add participants to school
                try:
                    school_age_to_participants_id[age] += curr_age_participants_id
                except:
                    school_age_to_participants_id[age] = curr_age_participants_id.copy()
                # add class size for age
                try:
                    school_to_age_class_size_arr[age].append(N)
                except:
                    school_to_age_class_size_arr[age] = [N]

        # populate classes in chool
        for age, curr_age_participants_id in school_age_to_participants_id.items():
            entity_N_arr = school_to_age_class_size_arr[age]
            curr_age_participants_id = np.array(curr_age_participants_id)
            # randomise participants
            np.random.shuffle(curr_age_participants_id)

            prev_age_j = 0
            for N in entity_N_arr:
                self.entity_id_counter += 1 # create entity (class)
                class_id = self.entity_id_counter
                # save entity ID to social entity type
                try:
                    self.entity_type_to_ids[school_class].append(class_id)
                except:
                    self.entity_type_to_ids[school_class] = [class_id]
                # apped class_id to school
                self.school_to_classes[school_id].append(class_id)

                for agent_id in curr_age_participants_id[prev_age_j:prev_age_j+N]:
                    # assign agent to class
                    self.individuals_df[agent_id]["social_id_arr"].append(class_id)
                    try:
                        self.entity_to_agents[class_id].append(agent_id)
                    except:
                        self.entity_to_agents[class_id] = [agent_id]
                    self.individuals_df[agent_id]["student_bool"] = 1
                    self.individuals_df[agent_id]["school_bool"] = 1
                prev_age_j += N

        return school_id_list

    def initialise_schools(self):
        """
        Initialise schools and classes
        """
        print ("Creating K-12 schools...")

        cdef int32 sec_school_bool, N, sample_n, sex
        cdef float f_sex
        cdef np.ndarray target_individuals, subset_target_inds
        cdef object participant_ids, school_id_list
        cdef int32 N_schooling_children = 0

        # schooling rate differs for primary and secondary school
        self.school_type_to_schools = {}
        for sec_school_bool in range(2):
            # get IDs of all target individuals (i.e. schooling age individuals)
            target_individuals = np.argwhere((self.pmap_age>=self.school_age_range[sec_school_bool][0])&(self.pmap_age<=self.school_age_range[sec_school_bool][-1])).T[0]

            # multiply by rate of participation (schooling rate) to get number of people enrolled in entities
            N = np.around(self.schooling_rate[sec_school_bool] * len(target_individuals), 0).astype(np.int32)

            # random selection of participating individuals in target population by gender
            participant_ids = []
            for sex in range(2):
                f_sex = 1/( 1 + self.school_gender_parity[sec_school_bool] ) if sex < 1 else self.school_gender_parity[sec_school_bool]/( 1 + self.school_gender_parity[sec_school_bool] )
                sample_n = np.around(N * f_sex).astype(np.int32)
                subset_target_inds = target_individuals[self.pmap_sex[target_individuals]==sex] # subset for sex
                if len(subset_target_inds) > sample_n:
                    participant_ids += list(np.random.choice(subset_target_inds, sample_n, replace=False))
                else:
                    participant_ids += list(subset_target_inds)
            participant_ids = np.sort(participant_ids)

            N_schooling_children += len(participant_ids)

            # create schools
            school_id_list = self.create_schools(participant_ids, self.mean_class_size[sec_school_bool], self.mean_school_size[sec_school_bool], self.student_teacher_ratio[sec_school_bool])
            # save school list to type
            self.school_type_to_schools[sec_school_bool] = np.array(school_id_list)

        print ('Number of schooling children:', N_schooling_children)
        return

    def create_workplaces(self, np.ndarray participant_ids, float mean_workplace_contacts, int32 formal_bool):

        cdef int32 school_id, class_id, agent_id, N
        cdef np.ndarray curr_participants_id, temp_arr
        cdef object class_arr
        cdef object teachers_id = []

        if formal_bool > 0: # allocate teachers first when accounting for formal workplaces
            # choose teachers from prime adults
            curr_participants_id = participant_ids[self.pmap_age[participant_ids]>=self.min_prime_adult_age].copy()
            for school_id, N in self.school_to_teachers_n.items():
                # randomly get teachers_id
                temp_arr = np.random.choice(curr_participants_id, N, replace=False)
                curr_participants_id = np.setdiff1d(curr_participants_id, temp_arr)
                teachers_id += list(temp_arr)

                # assign teachers to school
                for agent_id in temp_arr:
                    self.individuals_df[agent_id]["formal_employed_bool"] = 1
                    self.individuals_df[agent_id]["school_bool"] = 1
                    self.individuals_df[agent_id]["social_id_arr"].append(school_id)
                    try:
                        self.entity_to_agents[school_id].append(agent_id)
                    except:
                        self.entity_to_agents[school_id] = [agent_id]

                # assign teachers to classes
                class_arr = self.school_to_classes[school_id]
                if len(class_arr) < N: # more teachers than classes
                    class_arr += list(np.random.choice(class_arr, N-len(class_arr)))
                elif len(class_arr) > N: # more classes than teachers
                    temp_arr = np.array(list(temp_arr) + list(np.random.choice(temp_arr, len(class_arr)-N)))

                for class_id, agent_id in zip(class_arr, temp_arr):
                    self.individuals_df[agent_id]["social_id_arr"].append(class_id)
                    try:
                        self.entity_to_agents[class_id].append(agent_id)
                    except:
                        self.entity_to_agents[class_id] = [agent_id]

            teachers_id = np.array(teachers_id) # numpy-ize
            # remove individuals who have been assigned to be teachers
            participant_ids = np.setdiff1d(participant_ids, teachers_id)

        # create workplaces
        # get workplace size array
        N = len(participant_ids)
        cdef np.ndarray entity_N_arr = np.random.poisson(mean_workplace_contacts, size=int(np.ceil(N/mean_workplace_contacts)))
        entity_N_arr = np.delete(entity_N_arr, np.where(entity_N_arr==0)[0]) # remove any class with zero individuals
        # difference between workplace size array sum and N
        cdef int32 entity_N_diff = entity_N_arr.sum() - N

        # if difference is < 0, add more individuals until we have excess
        cdef int32 entity_N_arr_sum
        while entity_N_diff < 0:
            entity_N_arr = np.array(list(entity_N_arr)+list(np.random.poisson(mean_workplace_contacts, size=int(np.ceil(abs(entity_N_diff)/mean_workplace_contacts)))))
            entity_N_arr = np.delete(entity_N_arr, np.where(entity_N_arr==0)[0])
            entity_N_arr_sum = entity_N_arr.sum()
            entity_N_diff = entity_N_arr_sum - N

        # remove classes with excess individuals
        while entity_N_diff > 0:
            entity_N_arr = entity_N_arr[1:]
            entity_N_arr_sum = entity_N_arr.sum()
            entity_N_diff = entity_N_arr_sum - N

        if entity_N_diff < 0: # add remaining difference
            entity_N_arr = np.array(list(entity_N_arr)+[np.abs(entity_N_diff)])

        # randomly shuffle participant_ids
        np.random.shuffle(participant_ids)
        cdef int32 prev_i = 0

        for N in entity_N_arr:
            self.entity_id_counter += 1

            # save entity ID to social entity type
            if formal_bool > 0:
                try:
                    self.entity_type_to_ids[workplace_formal].append(self.entity_id_counter)
                except:
                    self.entity_type_to_ids[workplace_formal] = [self.entity_id_counter]
            else:
                try:
                    self.entity_type_to_ids[workplace_informal].append(self.entity_id_counter)
                except:
                    self.entity_type_to_ids[workplace_informal] = [self.entity_id_counter]

            for agent_id in participant_ids[prev_i:prev_i+N]:
                self.individuals_df[agent_id]["non_teacher_employed_bool"] = 1
                if formal_bool > 0:
                    self.individuals_df[agent_id]["formal_employed_bool"] = 1
                else:
                    self.individuals_df[agent_id]["informal_employed_bool"] = 1

                self.individuals_df[agent_id]["social_id_arr"].append(self.entity_id_counter)
                try:
                    self.entity_to_agents[self.entity_id_counter].append(agent_id)
                except:
                    self.entity_to_agents[self.entity_id_counter] = [agent_id]
            prev_i += N

        return

    def initialise_workplaces(self):
        """
        Initialise workplaces
        """
        print ("Creating workplaces...")
        cdef int32 agent_id
        # get IDs of all target individuals (i.e. schooling age individuals)
        cdef np.ndarray target_individuals = np.argwhere(self.pmap_age>=self.min_working_age).T[0]
        # remove schooling youths
        cdef np.ndarray schooling_youths = np.argwhere((self.pmap_age>=self.min_working_age)&(self.pmap_age<=self.school_age_range[1][-1])).T[0]
        schooling_youths = np.array([agent_id for agent_id in schooling_youths if self.individuals_df[agent_id]["student_bool"]>0])
        target_individuals = np.setdiff1d(target_individuals, schooling_youths)

        cdef int32 N, women_bool
        cdef np.ndarray curr_target_individuals, household_head_inds
        cdef float rate

        cdef object temp_obj = []
        # for each sex
        for women_bool, rate in enumerate(self.employment_rate):
            # multiply by rate of participation (employment rate) to get number of people in employment
            curr_target_individuals = target_individuals[self.pmap_sex[target_individuals]==women_bool]
            N = np.around(rate * len(curr_target_individuals), 0).astype(np.int32)
            # head of households are assumed to be preferentially employed
            household_head_inds = curr_target_individuals[self.pmap_household_head[curr_target_individuals]>0]

            if len(household_head_inds) < N:
                # less household heads than N to sample
                temp_obj += list(household_head_inds)
                temp_obj += list(np.random.choice(np.setdiff1d(curr_target_individuals, household_head_inds), N-len(household_head_inds), replace=False))
            else:
                # more household heads than N to sample
                temp_obj += list(np.random.choice(household_head_inds, N, replace=False))

        cdef np.ndarray participant_ids = np.sort(temp_obj) # combined for both sexes

        # further sort participants between those employed in formal and informal setting
        temp_obj = []
        # for each sex
        for women_bool, rate in enumerate(self.employment_rate_formal):
            # multiply by proportion of working individuals in formal setting
            curr_target_individuals = participant_ids[self.pmap_sex[participant_ids]==women_bool]
            N = np.around(rate * len(curr_target_individuals), 0).astype(np.int32)
            temp_obj += list(np.random.choice(curr_target_individuals, N, replace=False))

        # get individuals working in formal setting
        cdef np.ndarray participant_ids_formal = np.sort(temp_obj) # combined for both sexes
        # create formal workplaces
        self.create_workplaces(participant_ids_formal, self.mean_employment_contacts_formal, 1)

        # get individuals working in informal setting
        cdef np.ndarray participant_ids_informal = np.setdiff1d(participant_ids, participant_ids_formal)
        # create informal workplaces
        self.create_workplaces(participant_ids_informal, self.mean_employment_contacts_informal, 0)

        return

    def initialise_church(self):
        """
        Initialise church contact layer
        """

        print ("Creating churches...")

        # get mean/sd size of a church
        cdef float mean_entity_size = self.church_layer_params['entity_size'][0]
        cdef float sd_entity_size = self.church_layer_params['entity_size'][1]

        # sample by number of households as the sampling unit
        cdef int32 sample_n = np.around(self.church_layer_params['sampling_prop'] * len(self.households_arr)).astype(np.int32)
        cdef np.ndarray sampled_household_arr = np.sort(np.random.choice(self.households_arr, sample_n, replace=False)) # ordered households as well - nearer households would tend to go to the same church
        # get number of sampled households
        cdef int32 N = len(sampled_household_arr)

        # get array of churches that would congregate X number of households
        cdef np.ndarray entity_N_arr = get_halfnormal_int_sample(mean_entity_size/self.mean_household_size, sd_entity_size/self.mean_household_size, size=int(np.ceil(N/(mean_entity_size/self.mean_household_size))))
        entity_N_arr = np.delete(entity_N_arr, np.where(entity_N_arr==0)[0]) # remove any church with zero households
        # difference between church size array sum and N
        cdef int32 entity_N_diff = entity_N_arr.sum() - N

        while entity_N_diff < 0:
            entity_N_arr = np.array(list(entity_N_arr) + list(get_halfnormal_int_sample(mean_entity_size/self.mean_household_size, sd_entity_size/self.mean_household_size, size=int(np.ceil(abs(entity_N_diff)/(mean_entity_size/self.mean_household_size))))))
            entity_N_arr = np.delete(entity_N_arr, np.where(entity_N_arr==0)[0])
            entity_N_arr_sum = entity_N_arr.sum()
            entity_N_diff = entity_N_arr_sum - N

        # remove churches with excess households
        while entity_N_diff > 0:
            entity_N_arr = entity_N_arr[1:]
            entity_N_arr_sum = entity_N_arr.sum()
            entity_N_diff = entity_N_arr_sum - N
        if entity_N_diff < 0: # add remaining difference
            entity_N_arr = np.array(list(entity_N_arr)+[np.abs(entity_N_diff)])

        cdef int32 n, agent_id
        cdef np.ndarray curr_households, participant_ids
        cdef int32 prev_i = 0

        for n in entity_N_arr:

            self.entity_id_counter += 1

            # save entity ID to social entity type
            try:
                self.entity_type_to_ids[church].append(self.entity_id_counter)
            except:
                self.entity_type_to_ids[church] = [self.entity_id_counter]

            curr_households = sampled_household_arr[prev_i:prev_i+n]

            participant_ids = self.population_arr[np.isin(self.pmap_households, curr_households)]
            self.entity_to_agents[self.entity_id_counter] = list(participant_ids)

            for agent_id in participant_ids:
                self.individuals_df[agent_id]["social_id_arr"].append(self.entity_id_counter)

            prev_i += n

        return

    def initialise(self):
        """
        Initialise population and outbreak
        """

        cdef int32 agent_id, entity_id

        print ("Initialising population...")
        # initialise individual agents and households
        self.individuals_df = {} # initialise dictionary to store information of agents
        self.household_to_agents = {}
        self.initialise_population()
        self.population_arr = np.arange(self.pop_size) # holder for all individuals agent_id
        # individual array of age
        self.pmap_age = np.array([self.individuals_df[agent_id]["agent_age"] for agent_id in self.population_arr], dtype=np.int32)
        # individual array of sex
        self.pmap_sex = np.array([self.individuals_df[agent_id]["sex"] for agent_id in self.population_arr], dtype=np.int32)
        # inidividual array of household heads
        self.pmap_household_head = np.array([self.individuals_df[agent_id]["household_head"] for agent_id in self.population_arr], dtype=np.int32)
        # inidividual array of households
        self.pmap_households = np.array([self.individuals_df[agent_id]['household_id'] for agent_id in self.population_arr], dtype=np.int32)

        # create social contact layers
        self.entity_id_counter = -1 # entity id counter
        self.entity_type_to_ids = {} # dictionary to store array of entity IDs for each type
        self.entity_to_agents = {}

        # create schools
        self.school_to_classes = {} # dictionary to store school to classes
        self.school_to_teachers_n = {} # store number of teachers in each school
        self.initialise_schools()

        cdef object master_school_dict = {}
        cdef int8 sec_school_bool
        cdef int32 school_id
        cdef object class_list
        for sec_school_bool in range(2):
            for school_id in self.school_type_to_schools[sec_school_bool]:
                class_list = self.school_to_classes[school_id]
                try:
                    master_school_dict[sec_school_bool][school_id] = class_list
                except:
                    master_school_dict[sec_school_bool] = {school_id:class_list}

        # create workplaces
        self.initialise_workplaces()

        # create church layer
        if self.church_layer_params['bool'] > 0:
            self.initialise_church()

        # create social contact layer array as a sparse matrix
        print ("Creating social contact layer array...")
        # row indices
        cdef list row_ind = []
        # column indices
        cdef list col_ind = []
        # data to be stored in COO sparse matrix
        cdef list data = []

        cdef np.ndarray temp_arr
        for entity_id in np.arange(self.entity_id_counter+1):
            try:
                temp_arr = np.sort(self.entity_to_agents[entity_id])
            except:
                continue
            row_ind += [entity_id] * len(temp_arr)
            col_ind += list(temp_arr)
            data += [1] * len(temp_arr)

        cdef object social_contact_layer_arr = {"row_ind":row_ind, "col_ind":col_ind, "data":data, "shape":(self.entity_id_counter+1, self.pop_size)}

        # create household contact layer array as a sparse matrix
        print ("Creating household contact layer array...")

        # row indices
        row_ind = []
        # column indices
        col_ind = []
        # data to be stored in COO sparse matrix
        data = []

        for entity_id in self.households_arr:
            temp_arr = np.sort(self.household_to_agents[entity_id])

            row_ind += [entity_id] * len(temp_arr)
            col_ind += list(temp_arr)
            data += [1] * len(temp_arr)

        cdef object household_contact_layer_arr = {"row_ind":row_ind, "col_ind":col_ind, "data":data, "shape":(len(self.households_arr), self.pop_size)}

        print ("...done.")

        return self.individuals_df, household_contact_layer_arr, self.entity_type_to_ids, social_contact_layer_arr, master_school_dict

### --- Outbreak simulation class --- ###
@cython.no_gc_clear
cdef class Simulation:
    # declare variables
    # inputs
    cdef object individuals_df, entity_type_to_ids, household_contact_layer_arr, social_contact_layer_arr
    cdef object school_type_to_schools, school_to_classes, school_to_teachers, school_to_student_size
    cdef np.ndarray population_arr, pmap_agebins, pmap_age, pmap_households
    cdef int32 pop_size, total_days, teachers_n

    # healthcare and lab facilities
    cdef np.ndarray hcf_visit_delay

    # dynamic contact layers (bars)
    cdef int32 min_age_visiting_bars
    cdef float bar_visit_per_week, mean_bars_contact_size

    # outbreak initialisation
    cdef float init_wt_prop, init_mt_prop, init_immune_prop
    cdef int8 all_wt_init_exposed_bool
    cdef int32 mt_intro_delay

    # contact parameters
    cdef float mean_community_contact_size, mean_rand_church_contact_size

    # transmission parameters
    cdef float beta, f_asymp, f_mutant_beta, f_mutant_severe_prob, cross_immunity, max_vload_f, infectious_ct_thres
    cdef float fixed_community_prev, f_death_prob_wt, f_death_prob_mt
    cdef object ind_trans_dist, f_setting, var_cross_immunity_arr, simpop_vload_factor, transmission_bool

    # testing
    cdef object testing_sensitivity, testing_sensitivity_input, testing_strategies, exit_testing_boolean
    cdef object test_symp_req
    cdef float testing_specificity
    cdef float pcr_sensitivity, pcr_specificity
    cdef int32 pcr_ct_thres, selftest_otc_n
    cdef int8 ignore_test_specificity, pcr_test_bool, selftest_otc_bool, selftest_asymp_contacts_bool, selftest_at_clinic_bool
    cdef object selftest_asymp_req
    cdef int32 selftest_period
    cdef float selftest_endpoint_adherence, selftest_otc_own_prob
    cdef object simpop_agents_w_av, simpop_agents_av_benefit

    # isolation and quarantine
    cdef int8 quarantine_hcf_bool, quarantine_social_bool, daily_test_quarantine_bool, test8
    cdef int32 max_isolation_period
    cdef object f_contact_rate_reduction, isoquar_compliance, endpoint_isoquar_adherence
    cdef np.ndarray isoquar_period, prev_completed_isoquar_day

    # border crossing
    cdef float cross_border_traders_percent_employment
    cdef np.ndarray cross_border_traders_travel_freq_prop, cross_border_traders_length_of_stay_prop
    cdef object curr_agents_across_the_border, agents_infected_across_the_border
    cdef object cross_border_exposure_prob, cross_border_travelers, simpop_travel_days
    cdef int8 cross_border_travel_bool

    # contact tracing
    cdef int32 contact_tracing_delay

    # infection/transmissions related parameters
    cdef np.ndarray age_rel_sus, p_symptomatic, p_severe, p_death
    cdef np.ndarray tau_latent, tau_presymp, tau_symp_to_severe, tau_severe_to_death
    cdef np.ndarray tau_recovery_asymp, tau_recovery_mild, tau_recovery_severe, peak_ct_par
    cdef np.ndarray simpop_disease_periods_arr
    cdef object fated_symp_severity_arr, fated_to_die_arr

    # execution arrays
    cdef np.ndarray curr_contact_f_arr, hcf_sample_collection_day_arr, Reff_arr, curr_virus_type_arr, asymp_infector_arr
    cdef np.ndarray total_symp_testing_results, total_selftest_results, total_community_testing_results, total_exit_testing_results, total_daily_quarantine_testing_results, setting_incidence_arr, simpop_day_of_symptom_onset, length_of_infectious_period, untested_non_covid_symp_lack_of_test
    cdef object curr_seird_arr, simpop_infection_status, simpop_disease_severity, simpop_ct_arr, curr_isoquar_arr, agents_to_quarantine, daily_test_quarantine_agents, curr_days_in_isoquar, simpop_isoquar_arr, simpop_postest_setting,  border_crossing_stats, vtype_infector_to_infectee, untested_covid_symp_lack_of_test_arr, reported_daily_case_arr, curr_selftest_arr
    cdef np.ndarray exposed_day_infectee, setting_infectee, selftest_given_out
    cdef int32 agent_to_track

    # tests allocation
    cdef int32 number_of_rdt_per_100k_per_day, curr_number_of_hcf_rdt, symp_rdt_dist_type, rdt_restock_period, rdt_restock_day
    cdef object dist_comm_rdt_allocation, curr_number_of_comm_rdt, prev_tested_entities
    cdef float prop_rdt_hcf_allocation
    cdef np.ndarray curr_hcf_test_stocks

    # healthcare facilities
    cdef float pop_to_hcf_ratio
    cdef np.ndarray dist_of_distance_to_hcf, hcf_visit_probability_dist, voluntary_hcf_visit_prob
    cdef float fixed_voluntary_hcf_visit_prob
    cdef object hcf_contact_layer_arr, hcf_spatial_dist

    # NPIs
    cdef object npi_trans_prob_reduction_f

    # Non-covid testing demand
    cdef object non_covid_testing_demand_arr

    # Vaccination-related parameters
    cdef int32 vacc_min_age
    cdef float vacc_age_exp_scale
    cdef np.ndarray prop_agents_vacc_arr, vacc_immunity_arr, vacc_severe_f_arr
    cdef np.ndarray pmap_vacc_status

    # Antiviral-related parameter
    cdef object antiviral_req
    cdef int32 number_of_av_per_100k_per_day, av_restock_period, symp_av_dist_type, av_period_before_next_course
    cdef float av_or_red_sev, av_rom_symp_period
    cdef np.ndarray curr_hcf_av_stocks

    # at risk adults
    cdef np.ndarray pmap_adults_at_risk
    cdef float risk_prev, f_risk_severe_prob

    def __init__(self, object individuals_df, object household_contact_layer_arr, object entity_type_to_ids, object social_contact_layer_arr, object master_school_dict, **kwargs):

        # key infection/transmission related parameters
        # From COVASIM
        # Mortality rates are based on O'Driscoll et al. (Nature. 2021 Feb;590(7844):140–5) for ages <90
        # and Brazeau et al. (https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-34-ifr/) for ages >90
        # All other probabilities from Verity et al. (Vol. 20, The Lancet Infectious Diseases. 2020. p. 669–77.)
        # and Ferguson et al. (Impact of  non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand.
        # London: Imperial College COVID-19 Response Team, March. 2020;16.)
        # age-structured relative susceptibility
        self.age_rel_sus = kwargs.get('age_rel_sus', np.array([0.34,0.34,0.67,0.67,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.24,1.24,1.47,1.47,1.47,1.47]).astype(float))
        # age-structured probability of becoming symptomatic
        self.p_symptomatic = kwargs.get('p_symptomatic', np.array([0.50,0.50,0.55,0.55,0.60,0.60,0.65,0.65,0.70,0.70,0.75,0.75,0.80,0.80,0.85,0.85,0.90,0.90,0.90,0.90]).astype(float))
        # age-structured probability of becoming severe (hospitalisation)
        self.p_severe = kwargs.get('p_severe', np.array([0.00050,0.00050,0.00165,0.00165,0.00720,0.00720,0.02080,0.02080,0.03430,0.03430,0.07650,0.07650,0.13280,0.13280,0.20655,0.20655,0.24570,0.24570,0.24570,0.24570]).astype(float))
        # age-structured probability of death given that they have been infected by SARS-CoV-2
        self.p_death = kwargs.get('p_death', np.array([0.00002,0.00002,0.00002,0.00002,0.00010,0.00010,0.00032,0.00032,0.00098,0.00098,0.00265,0.00265,0.00766,0.00766,0.02439,0.02439,0.08292,0.08292,0.16190,0.16190]).astype(float))

        # proportion of agents that were vaccinated (>=18y, [partial, full, boosted])
        self.prop_agents_vacc_arr = kwargs.get('prop_agents_vacc_arr', np.array([0.07, 0.12, 0.004]))
        # minimum age to get vaccinated
        # simplified tier vaccination by going from oldest would more likely get vaccinated first, and down with age
        self.vacc_min_age = kwargs.get('vacc_min_age', 18)
        # exponential scale
        self.vacc_age_exp_scale = kwargs.get('vacc_age_exp_scale', 0.25)
        # protection from infection due to vaccination (vtype x vacc_status)
        self.vacc_immunity_arr = kwargs.get('vacc_immunity_arr', np.array([[0.575, 0.746, 0.927], [0.270, 0.294, 0.628]]))
        # protection from severe disease due to vaccination (vtype x vacc_status)
        self.vacc_severe_f_arr = kwargs.get('vacc_severe_f_arr', np.array([[0.712, 0.973, 0.993], [0.70, 0.70, 0.943]]))

        # time log-normal parameters (in days)
        # latent and presymp period = incubation
        # cdef np.ndarray tau_latent = np.array([[4.5, 1.5], [4.5, 1.5]]).astype(float)
        #self.tau_presymp = np.array([[1.1, 0.9], [1.1, 0.9]]).astype(float)

        # delta (https://www.medrxiv.org/content/10.1101/2021.08.12.21261991v1.full-text),
        # omicron (similar proliferation times and clearance rates according to https://dash.harvard.edu/handle/1/37370587)
        self.tau_latent = kwargs.get('tau_latent', np.array([[4., 1.3], [4., 1.3]]).astype(float))
        self.tau_presymp = kwargs.get('tau_presymp', np.array([[1.8, 1.7], [1.8, 1.7]]).astype(float))

        # symptomatic infectious to severe (requires hospitalisation); Linton et al. (https://dx.doi.org/10.3390%2Fjcm9020538)
        self.tau_symp_to_severe = kwargs.get('tau_symp_to_severe', np.array([[6.6, 4.9], [6.6, 4.9]]).astype(float))
        # severe (hospitalised) to death; Linton et al. (https://dx.doi.org/10.3390%2Fjcm9020538), Verity et al. (https://doi.org/10.1016/S1473-3099(20)30243-7)
        self.tau_severe_to_death = kwargs.get('tau_severe_to_death', np.array([[8.6, 6.7], [8.6, 6.7]]).astype(float))
        # from covasim
        #self.tau_recovery_asymp = np.array([[8., 2.], [8., 2.]]).astype(float)
        #self.tau_recovery_mild = np.array([[8., 2.], [8., 2.]]).astype(float)
        # based on clearance from https://dash.harvard.edu/handle/1/37370587
        self.tau_recovery_asymp = kwargs.get('tau_recovery_asymp', np.array([[6.23, 0.53], [5.35, .37]]).astype(float))
        self.tau_recovery_mild = kwargs.get('tau_recovery_mild', np.array([[6.23, 0.53], [5.35, .37]]).astype(float))
        self.tau_recovery_severe = kwargs.get('tau_recovery_severe', np.array([[18.1, 6.3], [18.1, 6.3]]).astype(float))

        # viral load computations following (Quilty et al.) https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(20)30308-X/fulltext
        # period of viral shedding since exposure is normally distributed (mean, sd) - WT based on https://www.thelancet.com/journals/lanmic/article/PIIS2666-5247(20)30172-5/fulltext
        # cdef np.ndarray tau_viral_shedding = np.array([[17., 0.8], [17., 0.8]])
        # peak Ct values is also assumed to be normally distributed (mean, sd)

        # update (30/1/2022): new peak Ct values based on Delta and Omicron (https://dash.harvard.edu/handle/1/37370587)
        self.peak_ct_par = kwargs.get('peak_ct_par', np.array([[20.5, 0.79], [23.3, 0.58]]).astype(float))

        # inputs from Population class
        # set up dictionary of agents
        self.individuals_df = individuals_df
        # get pop_size
        self.pop_size = len(self.individuals_df)
        self.population_arr = np.arange(self.pop_size).astype(np.int32)
        # set up contact layers of agents
        self.entity_type_to_ids = entity_type_to_ids
        # convert contact layer matrices to scipy sparse matrix
        self.household_contact_layer_arr = sparse.coo_matrix((household_contact_layer_arr["data"], (household_contact_layer_arr["row_ind"], household_contact_layer_arr["col_ind"])), shape=household_contact_layer_arr["shape"]).tocsr()
        self.social_contact_layer_arr = sparse.coo_matrix((social_contact_layer_arr["data"], (social_contact_layer_arr["row_ind"], social_contact_layer_arr["col_ind"])), shape=social_contact_layer_arr["shape"]).tocsr()
        # get school to classes dictionaries
        cdef int8 sec_school_bool
        cdef int32 school_id
        cdef object class_list
        self.school_type_to_schools = {}
        self.school_to_classes = {}
        for sec_school_bool in np.arange(2):
            self.school_type_to_schools[sec_school_bool] = np.array(list(master_school_dict[sec_school_bool].keys()))
            for school_id, class_list in master_school_dict[sec_school_bool].items():
                self.school_to_classes[school_id] = np.array(class_list)

        # bar inputs
        self.min_age_visiting_bars = kwargs.get('min_age_visiting_bars', 18)
        self.bar_visit_per_week = kwargs.get('bar_visit_per_week', 2)
        self.mean_bars_contact_size = kwargs.get('mean_bars_contact_size', 15)

        # initial proportion of people who are currently infected with WT lineage virus
        self.init_wt_prop = kwargs.get("init_wt_prop", 0.001)
        # initialise all wt infections as exposed
        self.all_wt_init_exposed_bool = kwargs.get('all_wt_init_exposed_bool', 0)
        # initial proportion of population who are infected with MT lineage virus
        self.init_mt_prop = kwargs.get("init_mt_prop", 0.)
        # initial proportion of people who are immune
        self.init_immune_prop = kwargs.get("init_immune_prop", 0.)
        # delay in introduction of MT virus
        self.mt_intro_delay = kwargs.get('mt_intro_delay', 0)
        # amount of cross immunity against MT strain
        self.cross_immunity = kwargs.get("cross_immunity", 0.9) # amount of cross immunity of WT against MT strain

        # probability of transmission when a susceptible comes into contact with infected
        self.beta = kwargs.get("beta", 0.017)
        # multiplicative factors of mutant virus
        self.f_asymp = kwargs.get("f_asymp", 1.) # fold change to beta if infected agent is asymptomatic
        self.f_mutant_beta = kwargs.get("f_mutant_beta", 1.74) # fold change to beta if infected by mutant
        self.f_mutant_severe_prob = kwargs.get("f_mutant_severe_prob", 1.64) # general fold change to severe probability of mutant virus
        self.f_death_prob_wt = kwargs.get("f_death_prob_wt", 1.) # fold change to death probability if infected by WT
        self.f_death_prob_mt = kwargs.get("f_death_prob_mt", 1.) # fold change to death probability if infected by MT

        # within-host viral load parameters
        self.max_vload_f = kwargs.get('max_vload_f', 2.) # transmission factor when viral load is max
        self.infectious_ct_thres = kwargs.get('infectious_ct_thres', 30) # Ct value below which we assume agent is infectious

        # isolation and quarantine
        self.f_contact_rate_reduction = kwargs.get('f_contact_rate_reduction', {isolation: np.array([0., 0.2, 0., 0., 0., 0., 0., 0., 0.]),
                                                                                quarantine: np.array([0., 0.2, 0., 0., 0., 0., 0., 0., 0.]),
                                                                                hospitalised: np.array([0., 0.2, 0., 0., 0., 0., 0., 0., 0.]),
                                                                                self_isolation: np.array([0., 0.2, 0., 0., 0., 0., 0., 0., 0.])})

        self.hcf_visit_delay = kwargs.get("hcf_visit_delay", np.array([1.1, 0.9])) # delay in visiting a HCF after being symptomatic (mild)
        self.isoquar_period = kwargs.get('isoquar_period', np.array([10, 14, 10, 10])) # stipulated period of isoquar (before exit test)
        self.max_isolation_period = kwargs.get('max_isolation_period', 10) # max period of isolation should exit test fail
        self.isoquar_compliance = kwargs.get('isoquar_compliance', {isolation:.92, quarantine:1., hospitalised:1., self_isolation:.1}) # probability of adherence to isoquar (either fully compliant or not at all)
        self.endpoint_isoquar_adherence = kwargs.get('endpoint_isoquar_adherence', {isolation:1., quarantine:1., self_isolation:1.}) # endpoint adherence to isoquar status
        self.exit_testing_boolean = kwargs.get('exit_testing_boolean', {isolation:0, quarantine:0}) # boolean to perform exit testing
        self.quarantine_hcf_bool = kwargs.get('quarantine_hcf_bool', 0) # quarantine close contacts for those tested positive at HCF
        self.quarantine_social_bool = kwargs.get('quarantine_social_bool', 0) # boolean to quarantine social contacts other than household contacts
        self.contact_tracing_delay = kwargs.get('contact_tracing_delay', 0) # delay in contact tracing of close contacts to be quarantined
        self.daily_test_quarantine_bool = kwargs.get('daily_test_quarantine_bool', 0) # daily testing instead of quarantine

        # either quarantine or self-test close contacts
        if self.quarantine_hcf_bool > 0 and self.selftest_asymp_contacts_bool > 0:
            raise Exception("For now, either quarantine or self-test asymptomatic close contacts but not both.")

        # testing (assuming no testing delays)
        # community testing strategies
        self.testing_strategies = kwargs.get("testing_strategies", {overseas:{"boolean":0., "unlimit_percent_tested":0., "prior_72h_test":0,}, # cross border
                                                                    household:{"boolean":0., "unlimit_percent_tested":0., "density":"low", "test_days":[0,3], "quarantine_bool":0}, # households
                                                                    school_class:{"boolean":1., "unlimit_percent_tested":0., "density":"low", "test_days":[0,4], "odd_week_bool":range(2), "school_to_test":2, "quarantine_bool":0}, # school_class
                                                                    workplace_formal:{"boolean":0., "unlimit_percent_tested":0., "density":"low", "test_days":[0], "quarantine_bool":0}, # workplace formal
                                                                    community:{"boolean":0., "unlimit_percent_tested":0., "test_days":[5,6], "odd_week_bool":range(2), "quarantine_bool":0}, # community
                                                                    church:{"boolean":0., "unlimit_percent_tested":0., "density":"low", "test_days":[6], "quarantine_bool":0}, # church
                                                                    bars:{"boolean":0., "unlimit_percent_tested":0., "density":"low", "test_days":range(7), "quarantine_bool":0} # testing before bar visits
                                                                    })

        # requirements to get symptomatic tests
        self.test_symp_req = kwargs.get('test_symp_req', {'bool':0, 'age_range':np.array([60, 105]), 'exclude_children':0, 'max_vacc_level':partial_vacc, 'risk':1, 'logic_mode':0})

        # reference: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1003735
        self.testing_sensitivity_input = kwargs.get("testing_sensitivity_input", {(40,35):0., (35,29):0.209, (29,24):0.507, (24,0):0.958}) # dependent on range of CT values when indvidual is tested - range(start_ct, end_ct):probability
        #self.testing_sensitivity_input = kwargs.get("testing_sensitivity_input", {(40,35):0., (35,30):0.3, (30,27):0.65, (27,0):0.95}) # dependent on range of CT values when indvidual is tested - range(start_ct, end_ct):probability
        self.testing_specificity = kwargs.get("testing_specificity", 0.989)
        self.ignore_test_specificity = kwargs.get('ignore_test_specificity', 0)
        # boolean denoting that testing is done by qt-PCR
        self.pcr_test_bool = kwargs.get('pcr_test_bool', 0)
        # ct cutoff, sensitivity and specificity of PCR tests
        self.pcr_ct_thres = kwargs.get('pcr_ct_thres', 35)
        self.pcr_sensitivity = kwargs.get('pcr_sensitivity', 1.0)
        self.pcr_specificity = kwargs.get('pcr_specificity', 1.0)

        # transform sensitivity input to array
        self.testing_sensitivity = np.zeros(41, dtype=float)
        cdef object ct_range
        cdef float prob
        cdef int32 ct
        if self.pcr_test_bool > 0:
            # test by PCR
            self.testing_sensitivity[:self.pcr_ct_thres+1] = self.pcr_sensitivity
            self.testing_specificity = self.pcr_specificity
        else:
            for ct_range, prob in self.testing_sensitivity_input.items():
                for ct in np.arange(ct_range[0], ct_range[-1], -1).astype(np.int32):
                    self.testing_sensitivity[ct] = prob

        # transmission
        self.transmission_bool = kwargs.get('transmission_bool', {household:1, school:0, workplace_formal:1,
                                                                  workplace_informal:1, community:1, church:0,
                                                                  bars:0})
        # mean number of random community contacts a day
        self.mean_community_contact_size = kwargs.get('mean_community_contact_size', 20)
        # mean number of random church contacts apart from household members
        self.mean_rand_church_contact_size = kwargs.get('mean_rand_church_contact_size', 9)
        # negative binomial parameters (mean, shape) modelling individual transmissiblity distribution (i.e. superspreading effects)
        self.ind_trans_dist = kwargs.get("ind_trans_dist", {"mean":1.0, "shape":0.45})
        # mulitplier to beta at different settings
        self.f_setting = kwargs.get("f_setting", {household:3., school_class:.6, school:.6, workplace_formal:.6, workplace_informal:.6, community:.3, church:.6, bars:.6})
        # fixed community prevalance - special variable activating a fixed contribution of new infections into settings with transmission bool > 0
        self.fixed_community_prev = kwargs.get('fixed_community_prev', 0.)

        # cross border travel
        self.cross_border_travel_bool = kwargs.get('cross_border_travel_bool', 0)
        # percentage of agents employed that crosses the border
        self.cross_border_traders_percent_employment = kwargs.get('cross_border_traders_percent_employment', 0.25)
        # proportion of traders travelling 1/per day, 1/per week, 1/per month
        self.cross_border_traders_travel_freq_prop = kwargs.get('cross_border_traders_travel_freq_prop', np.array([0.25, 0.35, 0.40]))
        # proportion of traders that will be away for different length of time [1 day (i.e. return on the same day), 2-7 days, >1-4 weeks]
        self.cross_border_traders_length_of_stay_prop = kwargs.get('cross_border_traders_length_of_stay_prop', np.array([0.7, 0.25, 0.05]))
        # prevalence of infections arising from border crossing (WT and MT)
        self.cross_border_exposure_prob = kwargs.get('cross_border_exposure_prob', {0:0., 1:0.})

        # testing capacity and allocation
        # number of available RDT per 100,000 inhabitants; -1 = infinite number of tests available
        self.number_of_rdt_per_100k_per_day = kwargs.get('number_of_rdt_per_100k_per_day', -1)
        # period between each rdt replenishment (0 = week, 1 = month, 2 = quarter)
        self.rdt_restock_period = kwargs.get('rdt_restock_period', 1)
        # proportion of available RDT for healthcare facilities (testing sick folks)
        self.prop_rdt_hcf_allocation = kwargs.get('prop_rdt_hcf_allocation', -1)
        # test distribution type across different healthcare clinics
        self.symp_rdt_dist_type = kwargs.get('symp_rdt_dist_type', 0)
        # distribution of community allocated RDT
        self.dist_comm_rdt_allocation = kwargs.get('dist_comm_rdt_allocation', {overseas:0., household:0., school_class:0., workplace_formal:0., community:0., church:0., bars:0.})

        # HCF
        self.pop_to_hcf_ratio = kwargs.get("pop_to_hcf_ratio", 7000) # ratio population to healthcare facilities
        self.dist_of_distance_to_hcf = kwargs.get("dist_of_distance_to_hcf", np.array([0.048, 0.193, 0.119, 0.08, 0.074, 0.098, 0.068, 0.072, 0.056, 0.191])) # distribution of distance to nearest healthcare facility (1 km incrementally)
        self.hcf_visit_probability_dist = kwargs.get("hcf_visit_probability_dist", np.array([0.853, 0.808, 0.762, 0.717, 0.672, 0.626, 0.581, 0.536, 0.49, 0.445])) # probability of HCF visit to get tested by distance to healthcare facility
        self.fixed_voluntary_hcf_visit_prob = kwargs.get('fixed_voluntary_hcf_visit_prob', -1)
        self.hcf_spatial_dist = kwargs.get('hcf_spatial_dist', {'type':'uniform', 'params':None})

        # NPIs
        self.npi_trans_prob_reduction_f = kwargs.get('npi_trans_prob_reduction_f', {overseas:1., household:1., school:0.5, school_class:0.5, workplace_formal:0.5, workplace_informal:0.75, community:0.75, church:0.5, bars:1.})

        # non covid testing demand
        self.non_covid_testing_demand_arr = kwargs.get('non_covid_testing_demand_arr', None) # array of day-by-day non covid testing demand

        # self-test
        self.selftest_otc_bool = kwargs.get("selftest_otc_bool", 0) # boolean for availability of store-bought self-test (unlimited for now)
        self.selftest_otc_own_prob = kwargs.get("selftest_otc_own_prob", 0.8) # Likelihood of OTC self-test if agent does not want to go HCF for testing

        self.selftest_asymp_contacts_bool = kwargs.get('selftest_asymp_contacts_bool', 0) # boolean for close contacts of those tested positive to self-test
        # requirements to give out self-tests
        self.selftest_asymp_req = kwargs.get('selftest_asymp_req', {'age_range':np.array([60, 105]), 'exclude_children':0, 'max_vacc_level':partial_vacc, 'risk':1, 'logic_mode':0})
        # number of daily self-tests per person
        self.selftest_period = kwargs.get('selftest_period', 3)
        # endpoint-adherence to self-testing
        self.selftest_endpoint_adherence = kwargs.get('selftest_endpoint_adherence', 0.5)
        # self-test at clinic boolean (otherwise all tests will be given to do so at home)
        self.selftest_at_clinic_bool = kwargs.get('selftest_at_clinic_bool', 1)

        # antivirals
        self.number_of_av_per_100k_per_day = kwargs.get('number_of_av_per_100k_per_day', 100) # number of antiviral stock available per 100,000 people per day (-1 = unlimited)
        self.av_restock_period = kwargs.get('av_restock_period', 2) # period between each antiviral stock replenishment (0 = week, 1 = month, 2 = quarter)
        self.symp_av_dist_type = kwargs.get('symp_av_dist_type', 0) # distribution of antivirals to healthcare clinics (0 = proportional to population size linked to clinic, 1 = equal for all HCFs, 2 = most densely populated clinic only)
        self.av_or_red_sev = kwargs.get('av_or_red_sev', 0.54) # odds ratio of hospital admission (severe case) if antiviral is administered
        self.av_rom_symp_period = kwargs.get('av_rom_symp_period', 0.67) # ratio of means for time to symptom resolution if antiviral is administered
        # Logical mode (0 = age|risk|vacc, 1 = (age&vacc)|(risk&vacc), 2 = (age&vacc)|risk, 3 = age|(risk&vacc), 4 = (age&risk)|vacc, 5 = age&risk&vacc
        self.antiviral_req = kwargs.get('antiviral_req', {'age_range':np.array([60, 105]), 'exclude_children':0, 'max_vacc_level':partial_vacc, 'days_since_onset':5, 'risk':1, 'logic_mode':0})
        self.av_period_before_next_course = kwargs.get('av_period_before_next_course', 7) # period between disbursing a second course to the same agent

        # adults at risk
        self.risk_prev = kwargs.get("risk_prev", 0.2) # prevalence of adults (>18) with at least one (aggregated) risk factor to result in more severe disease
        self.f_risk_severe_prob = kwargs.get('f_risk_severe_prob', 1.4) # fold change to severe probability if person has at least one risk factor

        self.agent_to_track = -1

    def simulate_ct_array(self, int8 virus_type, np.ndarray agents_arr, np.ndarray latent_period, np.ndarray incubation_period, np.ndarray tot_inf_period):

        # randomly sample peak Ct value for agents
        cdef np.ndarray peak_ct_val = np.random.normal(self.peak_ct_par[virus_type][0], self.peak_ct_par[virus_type][-1], size=len(agents_arr))
        # for all infectious individuals, peak Ct value must be below infectious_ct_thres
        cdef np.ndarray mask = peak_ct_val >= self.infectious_ct_thres
        while len(peak_ct_val[mask]) > 0:
            peak_ct_val[mask] = np.random.normal(self.peak_ct_par[virus_type][0], self.peak_ct_par[virus_type][-1], size=len(peak_ct_val[mask]))
            mask = peak_ct_val >= self.infectious_ct_thres
        # ... and cannot be below 0.
        mask = peak_ct_val < 1.
        peak_ct_val[mask] = 1.

        cdef int32 i
        cdef float grad
        cdef np.ndarray x_arr, y_arr, dydx_arr, i_ct_arr
        cdef object interpolator
        # initialise agent_ct_arr
        cdef np.ndarray agent_ct_arr = np.zeros((len(agents_arr), max(tot_inf_period)+1), dtype=np.int32)

        for i in np.arange(len(agents_arr)):
            # no incubation period (i.e. asymptomatic agent)
            if self.asymp_infector_arr[virus_type, agents_arr[i]] > 0:
                x_arr = np.array([0, latent_period[i], latent_period[i]+1, tot_inf_period[i]])
                y_arr = np.array([40, 30, peak_ct_val[i], 40]) # assume Ct value starts and resolves at 40 during the start and end of the infection
                dydx_arr = np.zeros(len(x_arr), dtype=float)
                dydx_arr[1] = (peak_ct_val[i] - 40)/(latent_period[i]+1)
            # symptomatic agents
            else:
                x_arr = np.array([0, latent_period[i], incubation_period[i], tot_inf_period[i]])
                y_arr = np.array([40, 30, peak_ct_val[i], 40]) # assume that agent will be infectious by the start of presymptomatic period
                dydx_arr = np.zeros(len(x_arr), dtype=float)
                dydx_arr[1] = (peak_ct_val[i] - 40)/(latent_period[i]+1)

            # interpolate day by day Ct values since exposure by CubicHermiteSpline
            try:
                interpolator = CubicHermiteSpline(x_arr, y_arr, dydx_arr)
            except:
                print (x_arr, y_arr, dydx_arr)
                raise Exception

            i_ct_arr = np.around(interpolator(np.arange(tot_inf_period[i] + 1)), 0).astype(np.int32)
            # lowest Ct values should not be zero
            i_ct_arr[i_ct_arr<1] = 1
            agent_ct_arr[i,:len(i_ct_arr)] = i_ct_arr

        return agent_ct_arr

    def compute_viral_load_array(self, np.ndarray agents_arr, int8 virus_type, int32 day, object initialise_as=None):

        # get fated latent period = exposed
        cdef np.ndarray latent_period = self.simpop_disease_periods_arr[virus_type, agents_arr, exposed_T]
        # get incubation period = exposed + presymp
        cdef np.ndarray incubation_period = latent_period + self.simpop_disease_periods_arr[virus_type, agents_arr, presymp_T]
        # compute total infection period (from exposed to resolution)
        cdef np.ndarray tot_inf_period = self.simpop_disease_periods_arr[virus_type, agents_arr, :].sum(axis=1)

        # simulate agents Ct values
        cdef np.ndarray agent_ct_arr = self.simulate_ct_array(virus_type, agents_arr, latent_period, incubation_period, tot_inf_period)

        # save number of days infected individuals are infectious
        self.length_of_infectious_period[virus_type, agents_arr] = (np.logical_and(agent_ct_arr <= self.infectious_ct_thres, agent_ct_arr > 0.).sum(axis=1))

        # get start of infectious date (previous day in the last day of latend period; first day when CT <= infectious_ct_thres)
        cdef np.ndarray argwhere_idx = np.argwhere((agent_ct_arr <= self.infectious_ct_thres) & (agent_ct_arr > 0.)).T
        cdef np.ndarray row_idx = argwhere_idx[0]
        cdef np.ndarray unique_row_idx, first_idx_row_idx
        unique_row_idx, first_idx_row_idx = np.unique(row_idx, return_index=True)
        cdef np.ndarray infectious_start_day = argwhere_idx[1][first_idx_row_idx]
        # adjust fated latent (exposed) period based on simulated Ct values
        latent_period = infectious_start_day
        self.simpop_disease_periods_arr[virus_type, agents_arr, exposed_T] = infectious_start_day

        # set up upcoming infection status
        cdef int32 i
        cdef int32 max_inf_period = max(tot_inf_period) + 1
        cdef np.ndarray infection_status_arr = np.zeros((len(agents_arr), max_inf_period), dtype=np.int8)
        # exposed up till infectious start day
        cdef np.ndarray flatten_idx_arr = np.concatenate([np.arange(i * max_inf_period, (i * max_inf_period ) + infectious_start_day[i]) for i in unique_row_idx]).ravel() # flatten index
        infection_status_arr.flat[flatten_idx_arr] = exposed_mt if virus_type > 0 else exposed_wt
        # infected till recovery or death
        flatten_idx_arr = np.concatenate([np.arange((i * max_inf_period) + infectious_start_day[i], (i * max_inf_period) + tot_inf_period[i] + 1) for i in unique_row_idx]).ravel() # flatten index
        infection_status_arr.flat[flatten_idx_arr] = infected_mt if virus_type > 0 else infected_wt

        # set up upcoming disease severity status
        cdef np.ndarray disease_severity_arr = np.zeros((len(agents_arr), max_inf_period), dtype=np.int8)
        # prior to infectious start day, all agents are asymptomatic
        flatten_idx_arr = np.concatenate([np.arange(i * max_inf_period, (i * max_inf_period) + infectious_start_day[i]) for i in unique_row_idx]).ravel()
        disease_severity_arr.flat[flatten_idx_arr] = asymptomatic
        # Get the day where Ct is at its peak for all agents
        cdef np.ndarray day_of_peak_ct = np.argmin(np.where(agent_ct_arr>0, agent_ct_arr, np.inf), axis=1)

        # get mask of symptomatic agents
        symp_mask = self.asymp_infector_arr[virus_type, agents_arr] < 1
        # for prospective symptomatic agent
        cdef np.ndarray symp_agents = agents_arr[symp_mask]
        if len(symp_agents) > 0:
            # adjust fated presymptomatic period based on simulated Ct values
            incubation_period[symp_mask] = day_of_peak_ct[symp_mask]
            self.simpop_disease_periods_arr[virus_type, symp_agents, presymp_T] = day_of_peak_ct[symp_mask] - infectious_start_day[symp_mask]
            # assign presymptomatic
            flatten_idx_arr = np.concatenate([np.arange((i * max_inf_period) + infectious_start_day[i], (i * max_inf_period) + day_of_peak_ct[i]) for i in unique_row_idx[symp_mask]]).ravel()
            disease_severity_arr.flat[flatten_idx_arr] = presymptomatic

        # compute length of symptomatic (or asymptomatic as well but we won't use them) days for all agents
        cdef np.ndarray length_of_symptomatic_days = np.unique(agent_ct_arr.nonzero()[0], return_counts=True)[-1] - np.unique(disease_severity_arr.nonzero()[0], return_counts=True)[-1]

        # adjust fated symptomatic period based on simulated Ct values
        # get fated severity of all agents
        cdef np.ndarray fated_symp_severity = self.fated_symp_severity_arr[virus_type, agents_arr].toarray()[0]
        # if agent is fated to present severe symptoms
        cdef np.ndarray n_mild_symp_days, n_severe_symp_days
        cdef np.ndarray severe_symp_mask = fated_symp_severity == severe
        cdef np.ndarray severe_symp_agents = agents_arr[severe_symp_mask]
        if len(severe_symp_agents) > 0:
            # adjust period before and after development of severe symptoms
            n_mild_symp_days = self.simpop_disease_periods_arr[virus_type, severe_symp_agents, symp_to_sev_T]
            n_severe_symp_days = self.simpop_disease_periods_arr[virus_type, severe_symp_agents, sev_to_recdea_T]

            n_mild_symp_days = np.floor((n_mild_symp_days/(n_mild_symp_days + n_severe_symp_days)) * length_of_symptomatic_days[severe_symp_mask]).astype(np.int32)
            n_severe_symp_days = length_of_symptomatic_days[severe_symp_mask] - n_mild_symp_days
            # update
            self.simpop_disease_periods_arr[virus_type, severe_symp_agents, symp_to_sev_T] = n_mild_symp_days
            self.simpop_disease_periods_arr[virus_type, severe_symp_agents, sev_to_recdea_T] = n_severe_symp_days

            # assign period of regular (mild) symptom period before severe symptoms
            flatten_idx_arr = np.concatenate([np.arange((i * max_inf_period) + day_of_peak_ct[i], (i * max_inf_period) + day_of_peak_ct[i] + n_mild_symp_days[j]) for j, i in enumerate(unique_row_idx[severe_symp_mask])]).ravel()
            disease_severity_arr.flat[flatten_idx_arr] = mild
            # assign the rest of the infected period as severe symptoms
            '''flatten_idx_arr = np.concatenate([np.arange((i * max_inf_period) + day_of_peak_ct[i] + n_mild_symp_days[j], (i * max_inf_period) + day_of_peak_ct[i] + n_mild_symp_days[j] + n_severe_symp_days[j]) for j, i in enumerate(unique_row_idx[severe_symp_mask])]).ravel()
            disease_severity_arr.flat[flatten_idx_arr] = severe'''
            for i in unique_row_idx[severe_symp_mask]:
                disease_severity_arr[i,np.setdiff1d(infection_status_arr[i,:].nonzero(), disease_severity_arr[i,:].nonzero())] = severe

        # if agents are fated to present mild symptoms only
        cdef np.ndarray mild_symp_mask = fated_symp_severity == mild
        cdef np.ndarray mild_symp_agents = agents_arr[mild_symp_mask]
        if len(mild_symp_agents) > 0:
            # update
            n_mild_symp_days = tot_inf_period[mild_symp_mask] + 1 - day_of_peak_ct[mild_symp_mask]
            self.simpop_disease_periods_arr[virus_type, mild_symp_agents, mild_rec_T] = n_mild_symp_days
            # assign period of mild symptoms
            flatten_idx_arr = np.concatenate([np.arange((i * max_inf_period) + day_of_peak_ct[i], (i * max_inf_period) + day_of_peak_ct[i] + n_mild_symp_days[j]) for j, i in enumerate(unique_row_idx[mild_symp_mask])]).ravel()
            disease_severity_arr.flat[flatten_idx_arr] = mild

        # if agents are fated to be asymptomatics
        cdef np.ndarray asymp_agents = agents_arr[~symp_mask]
        cdef np.ndarray n_asymp_days
        if len(asymp_agents) > 0:
            # update
            n_asymp_days = tot_inf_period[~symp_mask] + 1 - infectious_start_day[~symp_mask]
            self.simpop_disease_periods_arr[virus_type, asymp_agents, asymp_rec_T] = n_asymp_days
            # assign infected period of asymp
            flatten_idx_arr = np.concatenate([np.arange((i * max_inf_period) + infectious_start_day[i], (i * max_inf_period) + infectious_start_day[i] + n_asymp_days[j]) for j, i in enumerate(unique_row_idx[~symp_mask])]).ravel()
            disease_severity_arr.flat[flatten_idx_arr] = asymptomatic

        # compute viral load factor assuming peak titer = 2x more
        # initalise ind_viral_load_f_arr
        cdef np.ndarray ind_viral_load_f_arr = np.zeros((len(agents_arr), max_inf_period), dtype=float)
        # viral load factor = 1 minimally
        ind_viral_load_f_arr[argwhere_idx[0], argwhere_idx[-1]] = 1.
        # min Ct values of each agent
        cdef np.ndarray min_ind_ct = agent_ct_arr[unique_row_idx, day_of_peak_ct]

        # save everything to simpop arrays
        cdef int32 agent_id
        cdef np.ndarray day_range, agent_v_load_f, agent_inf_status, agent_disease_sev, agent_ct_vals, inf_status_mask

        if initialise_as == None:
            if len(symp_agents) > 0:
                # save day of symptom onset for symptomatic agents
                self.simpop_day_of_symptom_onset[virus_type, symp_agents] = day + day_of_peak_ct[symp_mask]
        else:
            # if initialised infections
            # completed exposed (latent) phase by day 0
            if virus_type > 0: # mutant virus
                # end of exposed period > transition to infected
                self.curr_seird_arr[exposed_mt, agents_arr] = 0
                self.curr_seird_arr[infected_mt, agents_arr] = 1
            else:
                # end of exposed period > transition to infected
                self.curr_seird_arr[exposed_wt, agents_arr] = 0
                self.curr_seird_arr[infected_wt, agents_arr] = 1

            if initialise_as == "presymptomatic":
                # initialise all agent to start day 0 at the end of latent phase (pre-infectious)
                self.exposed_day_infectee[virus_type, agents_arr] = -latent_period
                if len(symp_agents) > 0:
                    # save day of symptom onset for symptomatic agents
                    self.simpop_day_of_symptom_onset[virus_type, symp_agents] = -latent_period[symp_mask] + day_of_peak_ct[symp_mask]
            else:
                # initalise symptomatic agents to start day 0 as infected
                # for asymptomatic agents
                if len(asymp_agents) > 0:
                    self.exposed_day_infectee[virus_type, asymp_agents] = -latent_period[~symp_mask]
                # for symptomatic agents
                if len(symp_agents) > 0:
                    self.exposed_day_infectee[virus_type, symp_agents] = -incubation_period[symp_mask]
                    # save day of symptom onset for symptomatic agents
                    self.simpop_day_of_symptom_onset[virus_type, symp_agents] = -incubation_period[symp_mask] + day_of_peak_ct[symp_mask]

        for i, agent_id in enumerate(agents_arr):
            # save all infection_state, disease_severity and ct arrays
            agent_ct_vals = agent_ct_arr[i,:tot_inf_period[i]+1]
            agent_inf_status = infection_status_arr[i,:tot_inf_period[i]+1]
            agent_disease_sev = disease_severity_arr[i,:tot_inf_period[i]+1]
            agent_v_load_f = ind_viral_load_f_arr[i,:tot_inf_period[i]+1]

            # linearly interpolate to get day by day viral load factor
            if np.abs(min_ind_ct[i]-self.infectious_ct_thres) > 0:
                agent_v_load_f[agent_ct_vals <= self.infectious_ct_thres] += (1 - np.abs(min_ind_ct[i] - agent_ct_vals[agent_ct_vals<=self.infectious_ct_thres])/np.abs(min_ind_ct[i]-self.infectious_ct_thres)) * (self.max_vload_f - 1)

            if initialise_as != None:
                # initialised infections
                if initialise_as == 'infected' and self.asymp_infector_arr[virus_type, agent_id] < 1:
                    inf_status_mask = agent_disease_sev > presymptomatic
                else:
                    if virus_type > 0:
                        inf_status_mask = agent_inf_status >= infected_mt
                    else:
                        inf_status_mask = agent_inf_status >= infected_wt

                # update agent arrays
                agent_ct_vals = agent_ct_vals[inf_status_mask]
                agent_inf_status = agent_inf_status[inf_status_mask]
                agent_disease_sev = agent_disease_sev[inf_status_mask]
                agent_v_load_f = agent_v_load_f[inf_status_mask]

            if day < 0:
                day_range = np.arange(0, len(agent_ct_vals))
            else:
                day_range = np.arange(day, day + len(agent_ct_vals))

            # mask day_range before self.total_days
            mask = (day_range < self.total_days) & (day_range >= 0)
            if len(day_range[mask]) == 0:
                continue

            # save agent Ct values
            self.simpop_ct_arr[day_range[mask], agent_id] = agent_ct_vals[mask]
            # save disease_severity_arr
            self.simpop_disease_severity[day_range[mask], agent_id] = agent_disease_sev[mask]
            # save v load factor
            self.simpop_vload_factor[day_range[mask], agent_id] = agent_v_load_f[mask]

            # add death or recovery status
            if self.fated_to_die_arr[virus_type, agent_id] > 0:
                agent_inf_status = np.append(agent_inf_status, death)
            else:
                if virus_type > 0:
                    agent_inf_status = np.append(agent_inf_status, recovered_mt)
                else:
                    agent_inf_status = np.append(agent_inf_status, recovered_wt)
            if day < 0:
                day_range = np.arange(0, len(agent_inf_status))
            else:
                day_range = np.arange(day, day + len(agent_inf_status))
            # mask day_range before self.total_days
            mask = (day_range < self.total_days) & (day_range >= 0)
            if len(day_range[mask]) == 0:
                continue
            # save infection_status_arr
            self.simpop_infection_status[day_range[mask], agent_id] = agent_inf_status[mask]

            """print (self.simpop_ct_arr[day_range[mask], agent_id].toarray().T)
            print (self.simpop_disease_severity[day_range[mask], agent_id].toarray().T)
            print (self.simpop_vload_factor[day_range[mask], agent_id].toarray().T)
            print (self.simpop_infection_status[day_range[mask], agent_id].toarray().T)"""

        return

    def assign_symptomatic_disease_severity_array(self, np.ndarray agents_arr, np.ndarray prob_idx, np.ndarray samp_quantile, int8 virus_type):

        # get probability agent will have severe disease outcomes
        cdef np.ndarray severe_prob = self.p_severe[prob_idx]
        if virus_type > 0: # multiply by severity prob factor if mutant virus
            severe_prob *= self.f_mutant_severe_prob

        # multiply by risk factor
        cdef np.ndarray agents_risk = self.pmap_adults_at_risk[agents_arr]
        if len(severe_prob[agents_risk>0]) > 0:
            severe_prob[agents_risk>0] *= self.f_risk_severe_prob

        # if vaccinated or had a previous infection, reduce severe prob by max protection
        cdef np.ndarray agents_vacc_status = self.pmap_vacc_status[agents_arr]
        severe_prob[agents_vacc_status > 0] *= 1 - self.vacc_severe_f_arr[virus_type, agents_vacc_status[agents_vacc_status > 0]-1]
        cdef np.ndarray cond_severe_prob = severe_prob/self.p_symptomatic[prob_idx] # conditional proabability of severe disease given symptomatic
        cdef np.ndarray severe_bool = np.random.random(len(agents_arr)) < cond_severe_prob #np.random.choice([0, 1], p=[1-severe_prob, severe_prob])

        ### fated to be a severe case ###
        cdef np.ndarray death_prob, cond_death_prob, death_bool, death_agents, rec_sev_agents
        cdef np.ndarray symp_to_severe_period, severe_to_death_period, severe_to_rec_period

        cdef np.ndarray severe_mask = severe_bool == True
        cdef np.ndarray severe_agents = agents_arr[severe_mask]
        if len(severe_agents) > 0:
            # save fated severity
            self.fated_symp_severity_arr[virus_type, severe_agents] = severe
            # get random onset to severe symptoms period
            symp_to_severe_period = get_rand_lognormal_dur(self.tau_symp_to_severe[virus_type], quantile_arr=samp_quantile[severe_mask], min_val=1)
            # save sympton onset to severe symptoms period
            self.simpop_disease_periods_arr[virus_type, severe_agents, symp_to_sev_T] = symp_to_severe_period

            # compute probablity of being fated to die
            death_prob = self.p_death[prob_idx[severe_mask]]
            # multiply by death prob factor
            if virus_type > 0:
                death_prob *= self.f_death_prob_mt
            else:
                death_prob *= self.f_death_prob_wt
            # conditional proabability of death given severe case
            cond_death_prob = death_prob/severe_prob[severe_mask]
            death_bool = np.random.random(len(agents_arr[severe_mask])) < cond_death_prob #np.random.choice([0, 1], p=[1-death_prob, death_prob])

            death_agents = severe_agents[death_bool]
            if len(death_agents) > 0:
                # save which agents are fated to die
                self.fated_to_die_arr[virus_type, death_agents] = 1
                # get fated period between developing severe symptoms and death
                severe_to_death_period = get_rand_lognormal_dur(self.tau_severe_to_death[virus_type], quantile_arr=samp_quantile[severe_mask][death_bool], min_val=1)
                # save period between severe symptoms to death
                self.simpop_disease_periods_arr[virus_type, death_agents, sev_to_recdea_T] = severe_to_death_period

            rec_sev_agents = severe_agents[~death_bool]
            if len(rec_sev_agents) > 0:
                # get fated severe recovery period if agent won't die
                severe_to_rec_period = get_rand_lognormal_dur(self.tau_recovery_severe[virus_type], quantile_arr=samp_quantile[severe_mask][~death_bool], min_val=1)
                # save period between severe symptoms to recovery
                self.simpop_disease_periods_arr[virus_type, rec_sev_agents, sev_to_recdea_T] = severe_to_rec_period

        cdef np.ndarray mild_agents = agents_arr[~severe_mask]
        cdef np.ndarray mild_rec_period
        if len(mild_agents) > 0:
            mild_rec_period = get_rand_lognormal_dur(self.tau_recovery_mild[virus_type], quantile_arr=samp_quantile[~severe_mask], min_val=1)
            # save period between symptom onset to recovery form mild agents
            self.simpop_disease_periods_arr[virus_type, mild_agents, mild_rec_T] = mild_rec_period
            # save fated severity
            self.fated_symp_severity_arr[virus_type, mild_agents] = mild

        return

    def assign_infection_vars_to_exposed_array(self, np.ndarray agents_arr, np.ndarray infector_arr, int32 day, Social_Entity setting, int8 root_virus_type=-1, object initialise_as=None):

        # get virus type of infectors
        cdef np.ndarray virus_type_arr = np.zeros(len(agents_arr), dtype=np.int8)
        # seed infection (i.e. initialised infected)
        cdef np.ndarray mask = infector_arr < 0
        if len(virus_type_arr[mask]) > 0:
            virus_type_arr[mask] = root_virus_type
        # for actual infectors
        if len(virus_type_arr[~mask]) > 0:
            virus_type_arr[~mask] = self.curr_virus_type_arr[infector_arr[~mask]]

        # assign virus_type to infected agent
        self.curr_virus_type_arr[agents_arr] = virus_type_arr

        # change infection status of newly infected individuals to exposed depending on current status of agents
        # agents who are currently susceptible are no longer so
        mask = self.curr_seird_arr[susceptible, agents_arr].toarray()[0] > 0
        cdef np.ndarray subset_agents = agents_arr[mask]
        if len(subset_agents) > 0:
            self.curr_seird_arr[susceptible, subset_agents] = 0
        # remaining agents must be recovered_wt
        subset_agents = agents_arr[~mask]
        if len(subset_agents) > 0:
            if set(self.curr_seird_arr[recovered_wt, agents_arr[~mask]].toarray()[0]) != set([1]):
                raise Exception ("Error, should be recovered_wt status.")
            self.curr_seird_arr[recovered_wt, agents_arr[~mask]] = 0

        # day of exposure (for inidividuals infected since start of simulation)
        if day > -1:
            self.exposed_day_infectee[virus_type_arr, agents_arr] = day

        # infected setting
        # for all transmissions occurring during simulated outbreak
        if infector_arr[0] > -2:
            self.setting_infectee[virus_type_arr, agents_arr] = setting

        ### compute what happens to agent after completing exposure phase  ###
        # compute if agent is fated to be symptomatic
        # get the age bin index of agent's age
        cdef np.ndarray prob_idx = np.floor(self.pmap_agebins[agents_arr]/5).astype(np.int32)
        # get probability of being a symptomatic case
        cdef np.ndarray symp_prob = self.p_symptomatic[prob_idx]
        # get boolean if agent will be symptomatic
        cdef np.ndarray symp_bool = np.random.random(len(agents_arr)) < symp_prob

        cdef int8 virus_type
        cdef np.ndarray vtype_agents_arr, vtype_infector_arr, vtype_symp_bool
        cdef np.ndarray exposed_period_fated, samp_quantile
        cdef np.ndarray presymp_period_fated, asymp_recovery_period, symp_mask

        for virus_type in np.arange(2):
            mask = virus_type_arr == virus_type

            vtype_agents_arr = agents_arr[mask]
            if len(vtype_agents_arr) == 0:
                continue
            vtype_infector_arr = infector_arr[mask]

            # save infector to infectee info
            if len(vtype_infector_arr[vtype_infector_arr > -1]) > 0:
                # save infector/infectee/vtype (1 = WT, 2 = MT, 3 = WT + MT) info
                self.vtype_infector_to_infectee[vtype_infector_arr[vtype_infector_arr > -1], vtype_agents_arr[vtype_infector_arr > -1]] += (virus_type + 1)

            # for all transmissions occurring during simulated outbreak
            if root_virus_type == -1:
                # add count to setting incidence array
                self.setting_incidence_arr[day, setting, virus_type] += len(vtype_agents_arr)

            # update to exposed
            if virus_type > 0:
                self.curr_seird_arr[exposed_mt, vtype_agents_arr] = 1
                # change cross immunity for agent
                self.var_cross_immunity_arr[:, vtype_agents_arr] = 0 # no reinfection by WT and MT after infected by MT
            else:
                self.curr_seird_arr[exposed_wt, vtype_agents_arr] = 1
                # change cross immunity for agent
                self.var_cross_immunity_arr[0, vtype_agents_arr] = 0 # no reinfection by WT
                # take the minimum in likelihood of infection by mutant based on infection- and vaccine-acquired immunity
                self.var_cross_immunity_arr[1, vtype_agents_arr] = np.minimum(self.var_cross_immunity_arr[1, vtype_agents_arr], 1 - self.cross_immunity)

            # get fated latent (exposed period) and presymptomatic period
            # randomly sample latent period and get the sample quantile to be used for subsequent sampling
            exposed_period_fated, samp_quantile = get_rand_lognormal_dur(self.tau_latent[virus_type], N=len(vtype_agents_arr), min_val=1)
            # save sampled exposed period
            self.simpop_disease_periods_arr[virus_type, vtype_agents_arr, exposed_T] = exposed_period_fated

            # for symptomatic agents
            symp_mask = symp_bool[mask] == True
            # randomly sample fated presymptomatic phase in the same sample quantile
            subset_agents = vtype_agents_arr[symp_mask] # symptomatic agents
            if len(subset_agents) > 0:
                presymp_period_fated = get_rand_lognormal_dur(self.tau_presymp[virus_type], quantile_arr=samp_quantile[symp_mask], min_val=1)
                # save presymptomatic period
                self.simpop_disease_periods_arr[virus_type, subset_agents, presymp_T] = presymp_period_fated
                # assign fated symptomatic disease progression
                self.assign_symptomatic_disease_severity_array(subset_agents, prob_idx[mask][symp_mask], samp_quantile[symp_mask], virus_type)

            # asymptomatic agents
            subset_agents = vtype_agents_arr[~symp_mask]
            if len(subset_agents) > 0:
                # save to which agents are going to be asymptomatic
                self.asymp_infector_arr[virus_type, subset_agents] = 1
                # save fated severity
                self.fated_symp_severity_arr[virus_type, subset_agents] = asymptomatic
                # randomly sample fated asymptomatic period in the same sample quantile
                asymp_recovery_period = get_rand_lognormal_dur(self.tau_recovery_asymp[virus_type], quantile_arr=samp_quantile[~symp_mask], min_val=1)
                # save asymptomatic recovery period
                self.simpop_disease_periods_arr[virus_type, subset_agents, asymp_rec_T] = asymp_recovery_period

            # simulate within-host viral load for agent
            self.compute_viral_load_array(vtype_agents_arr, virus_type, day, initialise_as)

        return

    def initialise_infections(self):

        # make everyone susceptible first
        self.curr_seird_arr[susceptible,:] = 1

        # get # of agents that are already immune
        cdef int32 n_immune = np.around(self.init_immune_prop * len(self.population_arr), 0).astype(np.int32)
        print ("No. of initial immune individuals = %i"%(n_immune))
        # randomly select immune agents without replacement
        cdef np.ndarray immune_agents = np.random.choice(self.population_arr, n_immune, replace=False)

        # initialise immune agents of WT virus
        cdef int32 agent_id
        # change disease progression for immune agents
        self.curr_seird_arr[susceptible, immune_agents] = 0
        self.curr_seird_arr[recovered_wt, immune_agents] = 1

        # change variant susceptibilty for agents previously infected by wild-type
        self.var_cross_immunity_arr[0, immune_agents] = 0 # no reinfection by WT
        self.var_cross_immunity_arr[1, immune_agents] = 1 - self.cross_immunity

        # initialise agents with risk factor to severe disease
        self.pmap_adults_at_risk = np.zeros(self.pop_size, dtype=np.int8)
        cdef int32 n_all_adults = len(self.population_arr[self.pmap_age>=18])
        self.pmap_adults_at_risk[self.population_arr[self.pmap_age>=18][np.random.random(n_all_adults)<=self.risk_prev]] = 1

        # randomly select agents who were vaccinated
        cdef Vaccination_Status vacc_status
        cdef object vacc_status_label
        cdef int32 n_vacc, curr_min_age, curr_max_age
        cdef float vacc_immunity
        cdef int8 vtype
        cdef np.ndarray curr_agents_arr, curr_unvacc_agents, curr_unvacc_age, curr_vacc_prob, curr_age_range, age_range_prob

        for vacc_status, vacc_status_label in zip([boosted_vacc, full_vacc, partial_vacc], ['boosted', 'full', 'partial']):

            n_vacc = np.around(self.prop_agents_vacc_arr[vacc_status-1] * n_all_adults, 0).astype(np.int32)
            if n_vacc == 0:
                continue

            # get all currently unvacinated adults (>= self.vacc_min_age)
            curr_unvacc_agents = self.population_arr[(self.pmap_age >= self.vacc_min_age)&(self.pmap_vacc_status == 0)]
            if n_vacc >= len(curr_unvacc_agents):
                curr_agents_arr = curr_unvacc_agents[:]
            else:
                # if older, greater chance to be vaccinated
                curr_unvacc_age = self.pmap_age[curr_unvacc_agents]
                # compute probability of getting vaccinated first by age (assuming exponential distribution)
                curr_min_age = min(curr_unvacc_age)
                curr_max_age = max(curr_unvacc_age)
                curr_age_range = np.arange(curr_min_age, curr_max_age+1)
                age_range_prob = expon.pdf((curr_age_range-curr_min_age)/(curr_max_age-curr_min_age), scale=self.vacc_age_exp_scale)[::-1]
                curr_vacc_prob = age_range_prob[np.searchsorted(curr_age_range, curr_unvacc_age)]
                curr_vacc_prob /= curr_vacc_prob.sum()

                curr_agents_arr = np.random.choice(curr_unvacc_agents, n_vacc, p=curr_vacc_prob, replace=False)

            curr_agents_arr = np.sort(curr_agents_arr)

            print ('%s vacc status: %i agents'%(vacc_status_label, n_vacc,))
            print (np.unique(sorted(self.pmap_age[curr_agents_arr]), return_counts=True))

            # change vaccination status
            self.pmap_vacc_status[curr_agents_arr] = vacc_status
            # change variant susceptiblity for vaccinated individuals
            for vtype in np.arange(2):
                # we assume that
                vacc_immunity = 1 - self.vacc_immunity_arr[vtype, vacc_status-1]
                self.var_cross_immunity_arr[vtype, curr_agents_arr[self.var_cross_immunity_arr[vtype, curr_agents_arr] > vacc_immunity]] = vacc_immunity

        # get # of agents that were infected by WT
        cdef int32 n_infected_wt = np.around(self.init_wt_prop * len(self.population_arr), 0).astype(np.int32)
        print ("No. of initial WT infected individuals = %i"%(n_infected_wt))
        # filter out agents already infected with wild-type
        curr_agents_arr = np.setdiff1d(self.population_arr, immune_agents)
        # randomly select infected agents without replacement
        # probability of infection depends on their protection against wild-type
        cdef np.ndarray infected_agents = np.random.choice(curr_agents_arr, n_infected_wt, replace=False, p=self.var_cross_immunity_arr[0, curr_agents_arr]/self.var_cross_immunity_arr[0, curr_agents_arr].sum())

        # randomly select agents in different phases of infection
        cdef int32 n_exposed_wt, n_complete_exposed_wt
        # WT infected agents who were exposed on day 0
        # all start at exposed

        if len(infected_agents) == 0: # if there are no infected agents
            return

        if self.all_wt_init_exposed_bool > 0:
            # change infection status of initial WT-infected individuals to exposed
            # note that setting == household is just a dummy placeholder
            self.assign_infection_vars_to_exposed_array(infected_agents, np.repeat(-2, len(infected_agents)), 0, household, root_virus_type=0)

        else:
            n_exposed_wt, n_complete_exposed_wt = np.sort(np.random.randint(low=0, high=n_infected_wt, size=2))
            if n_exposed_wt > 0:
                # WT infected agents who are exposed on day 0
                self.assign_infection_vars_to_exposed_array(infected_agents[:n_exposed_wt], np.repeat(-2, len(infected_agents[:n_exposed_wt])), 0, household, root_virus_type=0)
                #print ('exposed', exposed_wt)
                #print (self.curr_seird_arr[:,infected_agents[:n_exposed_wt]])

            if n_complete_exposed_wt - n_exposed_wt > 0:
                # WT infected agents who completed latent period by day 0
                self.assign_infection_vars_to_exposed_array(infected_agents[n_exposed_wt:n_complete_exposed_wt], np.repeat(-2, len(infected_agents[n_exposed_wt:n_complete_exposed_wt])), -1, household, root_virus_type=0, initialise_as='presymptomatic')
                #print ('inf-presymp', infected_wt)
                #print (self.curr_seird_arr[:,infected_agents[n_exposed_wt:n_complete_exposed_wt]])

            # WT infected agents who were infected on day 0
            self.assign_infection_vars_to_exposed_array(infected_agents[n_complete_exposed_wt:], np.repeat(-2, len(infected_agents[n_complete_exposed_wt:])), -1, household, root_virus_type=0, initialise_as='infected')
            #print ('infected', infected_wt)
            #print (self.curr_seird_arr[:,infected_agents[n_complete_exposed_wt:]])

        print ('...done.')
        return

    def introduce_mt_infections(self, int32 day): # by exposure
        # compute no. of agents infected by MT
        cdef int32 n_infected_mt = np.around(self.init_mt_prop*self.pop_size, 0).astype(np.int32)
        print ("No. of initial MT infected individuals introduced on day %i = %i"%(day, n_infected_mt))

        # randomly select infected agent from susceptibles and recovered_wt
        cdef np.ndarray mt_susceptible_arr = np.union1d(self.curr_seird_arr[susceptible,:].tocoo().col, self.curr_seird_arr[recovered_wt,:].tocoo().col)
        # probability of infection depends on their protection against wild-type
        cdef np.ndarray infected_agent_ids = np.random.choice(mt_susceptible_arr, n_infected_mt, replace=False, p=self.var_cross_immunity_arr[1, mt_susceptible_arr]/self.var_cross_immunity_arr[1, mt_susceptible_arr].sum())

        # initialise MT infected individuals
        self.assign_infection_vars_to_exposed_array(infected_agent_ids, np.repeat(-2, len(infected_agent_ids)), day, household, root_virus_type=1)

        return

    def complete_isoquar(self, np.ndarray agents_arr, IsoQuar_Type isoquar_status, int32 day, object isoquar_reason):

        # exit isoquar_status
        self.curr_isoquar_arr[isoquar_status, agents_arr] = 0
        # save completed isoquar day
        self.prev_completed_isoquar_day[isoquar_status, agents_arr] = day
        # reset individual contact rate multiplier
        self.curr_contact_f_arr[:,agents_arr] = 1.
        # reset curr_days_in_isoquar
        self.curr_days_in_isoquar[:,agents_arr] = 0
        self.simpop_isoquar_arr[day:,agents_arr] = 0

        return

    def start_isoquar(self, np.ndarray agents_arr, int32 day, np.ndarray isoquar_period_arr, IsoQuar_Type isoquar_status, object isoquar_reason, object comm_test_setting=None):

        # other than isolated agents who tested positive during exit testing, determine agents who would adhere to isoquar (either fully compliant or not at all)
        cdef np.ndarray mask
        cdef int32 temp_count

        # save to curr_isoquar_arr for type of isoquar
        self.curr_isoquar_arr[isoquar_status, agents_arr] = 1

        cdef int32 i, agent_id
        cdef np.ndarray day_range
        for i, agent_id in enumerate(agents_arr):
            # record days agent will spend in isoquar
            day_range = np.arange(day, day+isoquar_period_arr[i])
            day_range = day_range[day_range<self.total_days]
            self.curr_days_in_isoquar[day_range, agent_id] = isoquar_status + 1
            self.simpop_isoquar_arr[day_range, agent_id] = isoquar_status + 1

        # change curr_contact_f_arr for agent
        # get contact rate reduction factor depedning on isoquar_status
        cdef np.ndarray contact_rate_red_arr = self.f_contact_rate_reduction[isoquar_status]

        if isoquar_reason == 'comm_testing':
            contact_rate_red_arr[comm_test_setting] = 0.
            if comm_test_setting == school_class:
                contact_rate_red_arr[school] = 0.

        self.curr_contact_f_arr[:,agents_arr] = np.tile(contact_rate_red_arr, (len(agents_arr),1)).T

        return

    def identify_close_contacts_for_quar_selftest(self, np.ndarray agents_arr, int32 day):

        # quarantine or self-test immediate close contacts in contact with positively-tested agents
        cdef IsoQuar_Type isoquar_status
        cdef np.ndarray entity_id_arr, social_contacts_arr
        cdef np.ndarray mask, age_mask, vacc_mask, symp_mask, risk_mask
        cdef int32 idx

        # get all contacts living in the same households
        cdef np.ndarray immd_close_contacts_arr = self.household_contact_layer_arr[self.pmap_households[agents_arr]].tocoo().col
        immd_close_contacts_arr = np.setdiff1d(immd_close_contacts_arr, agents_arr)
        if len(immd_close_contacts_arr) == 0: # no immediate close contacts
            return

        # quarantine
        if self.quarantine_hcf_bool > 0:
            # household contacts will be informed on the same day
            if len(immd_close_contacts_arr) > 0 and day + 1 < self.total_days:
                 self.agents_to_quarantine[day + 1, immd_close_contacts_arr] = 1

            if self.quarantine_social_bool > 0:
                # get all social entities of positively tested agents
                entity_id_arr = self.social_contact_layer_arr[:,agents_arr].tocoo().row
                # get corresponding social contacts
                social_contacts_arr = self.social_contact_layer_arr[entity_id_arr].tocoo().col
                social_contacts_arr = np.setdiff1d(social_contacts_arr, immd_close_contacts_arr)
                social_contacts_arr = np.setdiff1d(social_contacts_arr, agents_arr)

                # social contacts may only be contacted after delay
                if len(social_contacts_arr) > 0:
                    # quarantine will only be meted if contact tracing delay < quarantine period
                    if self.contact_tracing_delay < self.isoquar_period[quarantine] and (day + 1) < self.total_days and (day + self.contact_tracing_delay) < self.total_days:
                        if self.contact_tracing_delay == 0:
                            self.agents_to_quarantine[day + 1, social_contacts_arr] = 1
                        else:
                            self.agents_to_quarantine[day + self.contact_tracing_delay, social_contacts_arr] = 1

        # self-test
        elif self.selftest_asymp_contacts_bool > 0:
            # check eligibility for self-test
            # exclude children
            if self.selftest_asymp_req['exclude_children'] > 0:
                immd_close_contacts_arr = immd_close_contacts_arr[self.pmap_age[immd_close_contacts_arr]>=18]
                if len(immd_close_contacts_arr) == 0: # no immediate close contacts
                    return

            # age - anyone within age range
            age_mask = (self.pmap_age[immd_close_contacts_arr]>=self.selftest_asymp_req['age_range'][0])&(self.pmap_age[immd_close_contacts_arr]<=self.selftest_asymp_req['age_range'][-1])
            # vaccination status - anyone below the required vaccination status
            vacc_mask = self.pmap_vacc_status[immd_close_contacts_arr]<=self.selftest_asymp_req['max_vacc_level']
            # symptom mask - any immd contacts who has symptoms (regardless of risk factors) will be given tests
            symp_mask = self.simpop_disease_severity[day, immd_close_contacts_arr].toarray()[0]>=mild
            # risk mask
            risk_mask = self.pmap_adults_at_risk[immd_close_contacts_arr] > 1 - self.selftest_asymp_req['risk']

            # Logical mode (0 = age|risk|vacc, 1 = (age&vacc)|(risk&vacc), 2 = (age&vacc)|risk, 3 = age|(risk&vacc), 4 = (age&risk)|vacc, 5 = age&risk&vacc
            if self.selftest_asymp_req['logic_mode'] ==  0:
                immd_close_contacts_arr = immd_close_contacts_arr[age_mask|risk_mask|vacc_mask|symp_mask]
            elif self.selftest_asymp_req['logic_mode'] ==  1:
                immd_close_contacts_arr = immd_close_contacts_arr[(age_mask&vacc_mask)|(risk_mask&vacc_mask)|symp_mask]
            elif self.selftest_asymp_req['logic_mode'] ==  2:
                immd_close_contacts_arr = immd_close_contacts_arr[(age_mask&vacc_mask)|risk_mask|symp_mask]
            elif self.selftest_asymp_req['logic_mode'] ==  3:
                immd_close_contacts_arr = immd_close_contacts_arr[age_mask|(risk_mask&vacc_mask)|symp_mask]
            elif self.selftest_asymp_req['logic_mode'] ==  4:
                immd_close_contacts_arr = immd_close_contacts_arr[(age_mask&risk_mask)|vacc_mask|symp_mask]
            elif self.selftest_asymp_req['logic_mode'] ==  5:
                immd_close_contacts_arr = immd_close_contacts_arr[(age_mask&risk_mask&vacc_mask)|symp_mask]
            else:
                immd_close_contacts_arr = immd_close_contacts_arr[age_mask|risk_mask|symp_mask]

            if len(immd_close_contacts_arr) == 0: # no immediate close contacts left
                return

            # assign close contact agents to self-testing for the given period of time (including today)
            for idx in np.arange(day, day+self.selftest_period):
                if idx < self.total_days:
                    self.curr_selftest_arr[idx, immd_close_contacts_arr] = 1 - (idx - day) * ((1 - self.selftest_endpoint_adherence)/(self.selftest_period-1))

        return

    def isoquar_agent_by_pos_test_at_hcf(self, np.ndarray agents_arr, int32 day, int8 hospitalised_bool=0):

        # complete agents' previous isoquar type if any
        cdef np.ndarray subset_agents, mask
        cdef IsoQuar_Type isoquar_status
        for isoquar_status in [isolation, quarantine, self_isolation]:
            mask = self.curr_isoquar_arr[isoquar_status, agents_arr].toarray()[0] > 0
            subset_agents = agents_arr[mask]
            if len(subset_agents) > 0:
                self.complete_isoquar(subset_agents, isoquar_status, day, isoquar_reason='complete_prev_for_posiso')

        # compute isoquar period
        cdef np.ndarray isoquar_period_arr

        if hospitalised_bool > 0:
            self.reported_daily_case_arr[day,agents_arr] = 2 # add counts to confirmed cases
            # hospitalised period will only end upon recovery or death
            isoquar_period_arr = np.unique(self.simpop_disease_severity[day:,agents_arr].tocoo().col, return_counts=True)[-1]
            self.start_isoquar(agents_arr, day, isoquar_period_arr, hospitalised, isoquar_reason='hospitalisation')

        else:
            # non-hospitalised (i.e. mild)
            isoquar_period_arr = np.array([self.isoquar_period[isolation]] * len(agents_arr))
            self.start_isoquar(agents_arr, day, isoquar_period_arr, isolation, isoquar_reason='hcf_mild_iso')

        return

    def compute_testing(self, np.ndarray agents_arr, int32 day):

        # separate agents to be tested into infected and uninfected
        cdef np.ndarray mask = np.ravel(self.curr_seird_arr.tocsc()[exposed_wt:recovered_wt,agents_arr].sum(axis=0)) > 0
        cdef np.ndarray infected_arr = agents_arr[mask]
        cdef np.ndarray uninfected_arr = np.setdiff1d(agents_arr, infected_arr)

        '''### debug ###
        cdef int32 agent_id, status_sum
        for agent_id, status_sum in zip(agents_arr, np.ravel(self.curr_seird_arr.tocsc()[exposed_wt:recovered_wt,agents_arr].sum(axis=0))):
            if status_sum > 0 and self.simpop_ct_arr[day, agent_id] == 0:
                print (self.curr_seird_arr[:,agent_id])
                raise Exception
        ### debug ###'''

        # for agents who are currently infected, get their Ct values today
        cdef np.ndarray infected_ct_arr = self.simpop_ct_arr[day,infected_arr].toarray()[0]
        if (infected_ct_arr == 0).any():
            raise Exception ("XUETETY CT!")
        # map RDT testing sentivity and test
        cdef np.ndarray prob_arr = np.random.random(len(infected_arr))
        mask = prob_arr < self.testing_sensitivity[infected_ct_arr]
        cdef np.ndarray postest_infected_arr = infected_arr[mask]
        cdef np.ndarray falseneg_infected_arr = np.setdiff1d(infected_arr, postest_infected_arr)

        # return positively tested agents who had been positively tested as well as testing stats
        cdef int32 TP, FP, TN, FN
        TP = len(postest_infected_arr)
        FN = len(infected_arr) - TP
        TN = len(uninfected_arr)
        FP = 0

        cdef np.ndarray testing_results = np.array([TP, FP, TN, FN])

        if self.ignore_test_specificity > 0:
            # no need to account for false positives
            return postest_infected_arr, np.array([TP, FP, TN, FN])

        # for agents who are uninfected, test against specificity
        mask = np.random.random(len(uninfected_arr)) < 1 - self.testing_specificity
        cdef np.ndarray postest_uninfected_arr = uninfected_arr[mask]
        FP = len(postest_uninfected_arr)
        TN -= FP

        testing_results = np.array([TP, FP, TN, FN])

        cdef np.ndarray postest_agents = np.union1d(postest_infected_arr, postest_uninfected_arr)
        return postest_agents, testing_results

    def symptomatic_testing(self, np.ndarray symp_agents_who_will_visit_hcf, np.ndarray vtype_of_symp_agents_who_will_visit_hcf, np.ndarray symp_agents_who_will_selftest, np.ndarray vtype_of_symp_agents_who_will_selftest, int32 day):

        cdef np.ndarray vtype_arr, testing_results, postest_agents, negtest_agents, untested_agents_lack_of_test, compliant_agents, untested_agents_vtype, all_agents_who_want_test, immd_close_contacts_arr, mask, prob_arr, quarantined_agents, isoquar_period_arr, agents_hcf_arr
        cdef object agents_not_getting_test

        ##### symptomatic testing #####
        # get non-covid testing demand today based on given Poisson mean
        cdef int32 non_covid_tests_n_today
        try:
            non_covid_tests_n_today = np.random.poisson(self.non_covid_testing_demand_arr[day])
        except:
            non_covid_tests_n_today = 0

        cdef np.ndarray noninfected_agents_arr
        cdef np.ndarray non_covid_test_agents = np.array([])
        if non_covid_tests_n_today > 0:
            # randomly select non-infected agents (i.e. susceptible, recovered_wt, recovered_mt) as those who are seeking non-covid related tests
            noninfected_agents_arr = self.curr_seird_arr[susceptible,:].tocoo().col
            noninfected_agents_arr = np.union1d(noninfected_agents_arr, self.curr_seird_arr[recovered_wt,:].tocoo().col)
            noninfected_agents_arr = np.union1d(noninfected_agents_arr, self.curr_seird_arr[recovered_mt,:].tocoo().col)

            # keep noninfected agents to tho
            if len(noninfected_agents_arr) > 0:
                # mask agents who are currently overseas
                mask = self.curr_agents_across_the_border[noninfected_agents_arr] < 1
                noninfected_agents_arr = noninfected_agents_arr[mask]

            # remove agents who are currently in isolation, hospitalised (hard no) or self-isolation (some may still be after recovering from disease)
            if len(noninfected_agents_arr) > 0:
                mask = self.curr_isoquar_arr[isolation, noninfected_agents_arr].toarray()[0] < 1
                noninfected_agents_arr = noninfected_agents_arr[mask]

            if len(noninfected_agents_arr) > 0:
                mask = self.curr_isoquar_arr[hospitalised, noninfected_agents_arr].toarray()[0] < 1
                noninfected_agents_arr = noninfected_agents_arr[mask]

            if len(noninfected_agents_arr) > 0:
                mask = self.curr_isoquar_arr[self_isolation, noninfected_agents_arr].toarray()[0] < 1
                noninfected_agents_arr = noninfected_agents_arr[mask]

            # if there are more non-covid tests available than non-covid demand agents, then all noninfected agents will form the non-covid test demand
            if non_covid_tests_n_today >= len(noninfected_agents_arr):
                non_covid_test_agents = noninfected_agents_arr
            # elif there are less tests available...
            else:
                non_covid_test_agents = np.random.choice(noninfected_agents_arr, non_covid_tests_n_today, replace=False)

            # filter by those who willing to go seek testing (since nc_demand is now estimated from assuming 100% of all infected symptomatic agents would go get tested)
            prob_arr = np.random.random(len(non_covid_test_agents))
            mask = prob_arr < self.voluntary_hcf_visit_prob[non_covid_test_agents]
            # remaining agents will visit HCF for symptomatic testing
            non_covid_test_agents = non_covid_test_agents[mask]

        # if self-testing at OTC is available
        cdef np.ndarray age_mask, risk_mask
        cdef np.ndarray agents_st_arr, agents_who_cont_to_hcf_testing, agents_who_wont_hcf_test
        cdef int32 n
        if self.selftest_otc_bool > 0:
            ###
            # among non-covid-infected persons, there will be a proportion who will just go for symptomatic testing at healthcare facilities
            if len(non_covid_test_agents) > 0:
                n = np.around(self.fixed_voluntary_hcf_visit_prob * len(non_covid_test_agents)).astype(np.int32)
                agents_st_arr = np.random.choice(non_covid_test_agents, n, replace=False)
                # remove these non-infected agents who prefer to self-test
                non_covid_test_agents = non_covid_test_agents[~np.isin(non_covid_test_agents, agents_st_arr)]
                # the rest will perform a pre-self-test
                postest_agents, testing_results = self.compute_testing(agents_st_arr, day)
                if len(postest_agents) > 0:
                    # save symptomatic testing
                    self.simpop_postest_setting[day, postest_agents] = - 4
                    self.selftest_otc_n += len(postest_agents)

                    # assume that only those at-risk would continue to a reflexive test at healthcare facilities
                    age_mask = (self.pmap_age[postest_agents] >= self.test_symp_req['age_range'][0])&(self.pmap_age[postest_agents] <= self.test_symp_req['age_range'][-1])
                    risk_mask = self.pmap_adults_at_risk[postest_agents] > 1 - self.test_symp_req['risk'] # risk mask
                    agents_who_cont_to_hcf_testing = postest_agents[age_mask|risk_mask]
                    if len(agents_who_cont_to_hcf_testing) > 0:
                        if len(non_covid_test_agents) > 0:
                            non_covid_test_agents = np.concatenate([non_covid_test_agents, agents_who_cont_to_hcf_testing])
                        else:
                            non_covid_test_agents = agents_who_cont_to_hcf_testing

                    # agents who won't continue to reflex test will isolate
                    agents_who_wont_hcf_test = postest_agents[~(age_mask|risk_mask)]
                    if len(agents_who_wont_hcf_test) > 0:
                        prob_arr = np.random.random(len(agents_who_wont_hcf_test))
                        mask = prob_arr < self.isoquar_compliance[isolation]
                        # compliant positively tested agents will visit HCF and be isolated
                        compliant_agents = agents_who_wont_hcf_test[mask]
                        if len(compliant_agents) > 0:
                            # visit HCF and isolate compliant positively tested agents
                            self.isoquar_agent_by_pos_test_at_hcf(compliant_agents, day)

            ###
            # among symptomatic infected persons who do not want to go HCF for testing but are willing to do OTC self-tests
            postest_agents, testing_results = self.compute_testing(symp_agents_who_will_selftest, day)
            if len(postest_agents) > 0:
                # save symptomatic testing
                self.simpop_postest_setting[day, postest_agents] = - 4
                self.selftest_otc_n += len(postest_agents)

                # assume that only those at-risk would continue to a reflexive test at healthcare facilities
                age_mask = (self.pmap_age[postest_agents] >= self.test_symp_req['age_range'][0])&(self.pmap_age[postest_agents] <= self.test_symp_req['age_range'][-1])
                risk_mask = self.pmap_adults_at_risk[postest_agents] > 1 - self.test_symp_req['risk'] # risk mask
                agents_who_cont_to_hcf_testing = postest_agents[age_mask|risk_mask]

                if len(agents_who_cont_to_hcf_testing) > 0:
                    mask = np.isin(symp_agents_who_will_selftest, agents_who_cont_to_hcf_testing)
                    if len(symp_agents_who_will_visit_hcf) > 0:
                        symp_agents_who_will_visit_hcf = np.concatenate([symp_agents_who_will_visit_hcf, symp_agents_who_will_selftest[mask]], axis=0, dtype=np.int32)
                        vtype_of_symp_agents_who_will_visit_hcf = np.concatenate([vtype_of_symp_agents_who_will_visit_hcf, vtype_of_symp_agents_who_will_selftest[mask]], axis=0, dtype=np.int32)
                    else:
                        symp_agents_who_will_visit_hcf = symp_agents_who_will_selftest[mask]
                        vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_selftest[mask]

                # agents who won't continue to reflex test will isolate
                agents_who_wont_hcf_test = postest_agents[~(age_mask|risk_mask)]
                if len(agents_who_wont_hcf_test) > 0:
                    prob_arr = np.random.random(len(agents_who_wont_hcf_test))
                    mask = prob_arr < self.isoquar_compliance[isolation]
                    # compliant positively tested agents will visit HCF and be isolated
                    compliant_agents = agents_who_wont_hcf_test[mask]
                    if len(compliant_agents) > 0:
                        # visit HCF and isolate compliant positively tested agents
                        self.isoquar_agent_by_pos_test_at_hcf(compliant_agents, day)

        cdef np.ndarray vacc_mask
        if self.test_symp_req['bool'] > 0:
            # only agents who meet test distribution requirements get tests
            # check eligibility for test
            if self.test_symp_req['exclude_children'] > 0:
                # exclude children
                # infected agents
                if len(symp_agents_who_will_visit_hcf) > 0:
                    age_mask = self.pmap_age[symp_agents_who_will_visit_hcf]>=18
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[age_mask]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[age_mask]
                # non-infected agents
                if len(non_covid_test_agents) > 0:
                    age_mask = self.pmap_age[non_covid_test_agents]>=18
                    non_covid_test_agents = non_covid_test_agents[age_mask]

            if len(symp_agents_who_will_visit_hcf) > 0:
                # filter by age and vaccination status
                age_mask = (self.pmap_age[symp_agents_who_will_visit_hcf] >= self.test_symp_req['age_range'][0])&(self.pmap_age[symp_agents_who_will_visit_hcf] <= self.test_symp_req['age_range'][-1])
                vacc_mask = self.pmap_vacc_status[symp_agents_who_will_visit_hcf] <= self.test_symp_req['max_vacc_level']
                risk_mask = self.pmap_adults_at_risk[symp_agents_who_will_visit_hcf] > 1 - self.test_symp_req['risk'] # risk mask

                # Logical mode; 0 = age|risk|vacc, 1 = (age&vacc)|(risk&vacc), 2 = (age&vacc)|risk, 3 = age|(risk&vacc), 4 = (age&risk)|vacc, 5 = age&risk&vacc, 6 = age|risk
                if self.test_symp_req['logic_mode'] ==  0:
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[age_mask|risk_mask|vacc_mask]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[age_mask|risk_mask|vacc_mask]
                elif self.test_symp_req['logic_mode'] ==  1:
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[(age_mask&vacc_mask)|(risk_mask&vacc_mask)]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[(age_mask&vacc_mask)|(risk_mask&vacc_mask)]
                elif self.test_symp_req['logic_mode'] ==  2:
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[(age_mask&vacc_mask)|risk_mask]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[(age_mask&vacc_mask)|risk_mask]
                elif self.test_symp_req['logic_mode'] ==  3:
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[age_mask|(risk_mask&vacc_mask)]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[age_mask|(risk_mask&vacc_mask)]
                elif self.test_symp_req['logic_mode'] ==  4:
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[(age_mask&risk_mask)|vacc_mask]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[(age_mask&risk_mask)|vacc_mask]
                elif self.test_symp_req['logic_mode'] ==  5:
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[(age_mask&risk_mask&vacc_mask)]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[(age_mask&risk_mask&vacc_mask)]
                else:
                    symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[age_mask|risk_mask]
                    vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[age_mask|risk_mask]

            # filter by age and vaccination status
            if len(non_covid_test_agents) > 0:
                age_mask = (self.pmap_age[non_covid_test_agents] >= self.test_symp_req['age_range'][0])&(self.pmap_age[non_covid_test_agents] <= self.test_symp_req['age_range'][-1])
                vacc_mask = self.pmap_vacc_status[non_covid_test_agents] <= self.test_symp_req['max_vacc_level']
                risk_mask = self.pmap_adults_at_risk[non_covid_test_agents] > 1 - self.test_symp_req['risk'] # risk mask

                # Logical mode; 0 = age|risk|vacc, 1 = (age&vacc)|(risk&vacc), 2 = (age&vacc)|risk, 3 = age|(risk&vacc), 4 = (age&risk)|vacc, 5 = age&risk&vacc
                if self.test_symp_req['logic_mode'] ==  0:
                    non_covid_test_agents = non_covid_test_agents[age_mask|risk_mask|vacc_mask]
                elif self.test_symp_req['logic_mode'] ==  1:
                    non_covid_test_agents = non_covid_test_agents[(age_mask&vacc_mask)|(risk_mask&vacc_mask)]
                elif self.test_symp_req['logic_mode'] ==  2:
                    non_covid_test_agents = non_covid_test_agents[(age_mask&vacc_mask)|risk_mask]
                elif self.test_symp_req['logic_mode'] ==  3:
                    non_covid_test_agents = non_covid_test_agents[age_mask|(risk_mask&vacc_mask)]
                elif self.test_symp_req['logic_mode'] ==  4:
                    non_covid_test_agents = non_covid_test_agents[(age_mask&risk_mask)|vacc_mask]
                elif self.test_symp_req['logic_mode'] ==  5:
                    non_covid_test_agents = non_covid_test_agents[(age_mask&risk_mask&vacc_mask)]
                else:
                    non_covid_test_agents = non_covid_test_agents[age_mask|risk_mask]

        cdef int32 hcf_id
        cdef np.ndarray agents_at_hcf
        # adjustments if there are finite number of tests available
        if self.curr_number_of_hcf_rdt > -1:
            ### no more tests available this week ###
            if self.curr_number_of_hcf_rdt == 0:
                # nobody will get a test
                untested_agents_lack_of_test = symp_agents_who_will_visit_hcf
                untested_agents_vtype = vtype_of_symp_agents_who_will_visit_hcf

                # save number of non-covid agents seeking tests but failed because of shortage
                self.untested_non_covid_symp_lack_of_test[day] = len(non_covid_test_agents)
                non_covid_test_agents = np.array([])

            else:
                # get all agents who want to be tested and the clinics they will visit
                all_agents_who_want_test = np.union1d(symp_agents_who_will_visit_hcf, non_covid_test_agents).astype(np.int32)
                try:
                    agents_hcf_arr = self.hcf_contact_layer_arr.tocsc()[:,all_agents_who_want_test].tocoo().row
                except:
                    print (all_agents_who_want_test)
                    raise Exception("WHAT?")

                agents_not_getting_test = []
                for hcf_id in np.arange(len(self.curr_hcf_test_stocks)):
                    agents_at_hcf = all_agents_who_want_test[agents_hcf_arr==hcf_id]

                    if self.curr_hcf_test_stocks[hcf_id] == 0:
                        # no more test stock at clinic
                        agents_not_getting_test += list(agents_at_hcf)
                        continue

                    # if there are more agents seeking tests than there are available tests
                    if len(agents_at_hcf) > self.curr_hcf_test_stocks[hcf_id]:
                        agents_not_getting_test += list(np.random.choice(agents_at_hcf, len(agents_at_hcf) - self.curr_hcf_test_stocks[hcf_id], replace=False))
                        self.curr_hcf_test_stocks[hcf_id] = 0
                    else:
                        # enough tests for all agents
                        self.curr_hcf_test_stocks[hcf_id] -= len(agents_at_hcf)

                self.curr_number_of_hcf_rdt = self.curr_hcf_test_stocks.sum()

                # filter out infected agents who will NOT be tested
                mask = np.isin(symp_agents_who_will_visit_hcf, agents_not_getting_test)
                untested_agents_lack_of_test = symp_agents_who_will_visit_hcf[mask]
                untested_agents_vtype = vtype_of_symp_agents_who_will_visit_hcf[mask]

                # filter out noninfected agents who will NOT be tested
                mask = np.isin(non_covid_test_agents, agents_not_getting_test)
                # save number of non-covid agents seeking tests but failed because of shortage
                self.untested_non_covid_symp_lack_of_test[day] = len(non_covid_test_agents[mask])
                non_covid_test_agents = non_covid_test_agents[~mask]

            if len(untested_agents_lack_of_test) > 0:
                # save which COVID symptomatic agents that were not tested due to a lack of test
                self.untested_covid_symp_lack_of_test_arr[day, untested_agents_lack_of_test] = 1

                # agents who are not tested due to shortage: their samples are assumed to not be collected
                self.hcf_sample_collection_day_arr[untested_agents_vtype, untested_agents_lack_of_test] = -1

                # remaining agents will be tested
                mask = np.isin(symp_agents_who_will_visit_hcf, untested_agents_lack_of_test)
                symp_agents_who_will_visit_hcf = symp_agents_who_will_visit_hcf[~mask]
                vtype_of_symp_agents_who_will_visit_hcf = vtype_of_symp_agents_who_will_visit_hcf[~mask]

        if len(symp_agents_who_will_visit_hcf) > 0:
            # test covid symptomatic agents at HCF
            postest_agents, testing_results = self.compute_testing(symp_agents_who_will_visit_hcf, day)
            negtest_agents = np.setdiff1d(symp_agents_who_will_visit_hcf, postest_agents) # get infected agents who tested negative

            self.total_symp_testing_results[day,:] += testing_results

            # positively tested agents will go into isolation
            if len(postest_agents) > 0:

                # add counts to confirmed cases
                self.reported_daily_case_arr[day, postest_agents] = 1
                # save symptomatic testing
                self.simpop_postest_setting[day, postest_agents] = -1

                # collect samples from agents who tested positively
                mask = np.isin(symp_agents_who_will_visit_hcf, postest_agents)
                self.hcf_sample_collection_day_arr[vtype_of_symp_agents_who_will_visit_hcf[mask], symp_agents_who_will_visit_hcf[mask]] = day

                # check eligibility and administer antiviral
                self.disburse_av(symp_agents_who_will_visit_hcf[mask], vtype_of_symp_agents_who_will_visit_hcf[mask], day)

                # quarantine or self-test close contacts of positively-tested agents
                if self.quarantine_hcf_bool > 0 or self.selftest_asymp_contacts_bool > 0:
                    self.identify_close_contacts_for_quar_selftest(postest_agents, day)

                # will positively-tested comply with isolation requirement?
                prob_arr = np.random.random(len(postest_agents))
                mask = prob_arr < self.isoquar_compliance[isolation]
                # compliant positively tested agents will visit HCF and be isolated
                compliant_agents = postest_agents[mask]
                if len(compliant_agents) > 0:
                    # visit HCF and isolate compliant positively tested agents
                    self.isoquar_agent_by_pos_test_at_hcf(compliant_agents, day)

            # agents tested negative will not be isolated and deemed that they had visit HCF already
            if len(negtest_agents) > 0:
                mask = np.isin(symp_agents_who_will_visit_hcf, negtest_agents)
                self.hcf_sample_collection_day_arr[vtype_of_symp_agents_who_will_visit_hcf[mask], symp_agents_who_will_visit_hcf[mask]] = -1

        if len(non_covid_test_agents) > 0:
            # test non-covid infected agents
            postest_agents, testing_results = self.compute_testing(non_covid_test_agents, day)
            self.total_symp_testing_results[day,:] += testing_results

            if len(postest_agents) > 0:
                # add (incorrect) counts to confirmed cases
                self.reported_daily_case_arr[day, postest_agents] = 1
                # save symptomatic testing for non-covid agents
                self.simpop_postest_setting[day, postest_agents] = -2

                # quarantine or self-test close contacts of positively-tested agents
                if self.quarantine_hcf_bool > 0 or self.selftest_asymp_contacts_bool > 0:
                    self.identify_close_contacts_for_quar_selftest(postest_agents, day)

                # will postest_agents comply with isolation requirement?
                prob_arr = np.random.random(len(postest_agents))
                mask = prob_arr < self.isoquar_compliance[isolation]
                # compliant positively tested agents will visit HCF and be isolated
                compliant_agents = postest_agents[mask]

                if len(compliant_agents) > 0:
                    # some agents may be in quarantine currently
                    mask = self.curr_isoquar_arr[quarantine, compliant_agents].toarray()[0] > 0
                    quarantined_agents = compliant_agents[mask]
                    if len(quarantined_agents) > 0:
                        # complete their quarantine status
                        self.complete_isoquar(quarantined_agents, quarantine, day, isoquar_reason='complete_quar_to_start_posiso')

                    # start isolation for positively tested agents
                    isoquar_period_arr = np.zeros(len(compliant_agents), dtype=np.int32) + self.isoquar_period[isolation]
                    self.start_isoquar(compliant_agents, day, isoquar_period_arr, isolation, isoquar_reason='noncovid_symp_testing')

    def review_disease_progression(self, int32 day):

        # review currently exposed individuals for each virus type
        cdef int32 virus_type, agent_id
        cdef np.ndarray agents_arr, today_status
        cdef Infection_Status status

        for virus_type, status in zip([0, 1], [exposed_wt, exposed_mt]):
            # update infection status of exposed agents
            agents_arr = self.curr_seird_arr[status,:].tocoo().col
            today_status = self.simpop_infection_status[day,agents_arr].toarray()[0]
            # get agents leaving exposed status
            agents_arr = agents_arr[today_status != status]
            if len(agents_arr) > 0:
                # update new status
                self.curr_seird_arr[status, agents_arr] = 0
                if virus_type > 0:
                    self.curr_seird_arr[infected_mt, agents_arr] = 1
                else:
                    self.curr_seird_arr[infected_wt, agents_arr] = 1

        # review currently infected individuals
        cdef int8 new_status
        cdef np.ndarray rec_dead_agents, infected_agents, mild_infected_agents, subset_agents, othset_agents, new_status_arr
        cdef np.ndarray disease_severity_arr, mask, prob_arr, days_since_symptom_onset_arr, quarantined_agents
        cdef object temp_coo_matrix
        cdef np.ndarray counts_arr, j_arr, dat_arr

        cdef object symp_agents_who_will_visit_hcf = []
        cdef object symp_agents_who_will_selftest = []
        cdef object vtype_of_symp_agents_who_will_visit_hcf = []
        cdef object vtype_of_symp_agents_who_will_selftest = []

        for virus_type, status in zip([0, 1], [infected_wt, infected_mt]):
            # update infection status of infected agents
            agents_arr = self.curr_seird_arr[status,:].tocoo().col
            today_status = self.simpop_infection_status[day,agents_arr].toarray()[0]

            #######

            # get agents who have either recovered or died
            rec_dead_agents = agents_arr[today_status != status]
            new_status_arr = today_status[today_status != status]
            if len(rec_dead_agents) > 0:
                # update new status post infection
                self.curr_seird_arr[status, rec_dead_agents] = 0

                for new_status in np.unique(new_status_arr):
                    if new_status == susceptible: # new_status should not be susceptible
                        raise Exception("FUCKITY!")

                    subset_agents = rec_dead_agents[new_status_arr == new_status]
                    self.curr_seird_arr[new_status, subset_agents] = 1

                    # reset curr_virus_type
                    self.curr_virus_type_arr[subset_agents] = -1

                    # discharge agents who were previously hospitalised
                    subset_agents = subset_agents[self.curr_isoquar_arr[hospitalised, subset_agents].tocoo().col]
                    if len(subset_agents) > 0:
                        self.complete_isoquar(subset_agents, hospitalised, day, isoquar_reason='discharged_hospital')
                        # add to confirmed deaths (i.e. those that had been hospitalised)
                        if new_status == death:
                            self.reported_daily_case_arr[day,subset_agents] = 3

            #######

            # get agents who are still infected and disease severity
            infected_agents = agents_arr[today_status == status]
            disease_severity_arr = self.simpop_disease_severity[day, infected_agents].toarray()[0]

            # subset severe agents who are not yet hospitalised
            subset_agents = infected_agents[disease_severity_arr==severe]

            # get disease severity up to today
            temp_coo_matrix = self.simpop_disease_severity[:day+1,subset_agents].tocoo()
            j_arr = temp_coo_matrix.col
            dat_arr = temp_coo_matrix.data
            # mask days when agents have severe disease
            mask = dat_arr == severe
            j_arr = j_arr[mask]
            # get unique agents and count number of occurrence for each of them
            j_arr, counts_arr = np.unique(j_arr, return_counts=True)
            # only subset agents who are experiencing their first severe symptoms today
            subset_agents = subset_agents[j_arr[counts_arr == 1]]

            mask = self.curr_isoquar_arr[hospitalised, subset_agents].toarray()[0] < 1
            subset_agents = subset_agents[mask]
            # mask agents who are also currently overseas
            mask = self.curr_agents_across_the_border[subset_agents] < 1
            subset_agents = subset_agents[mask]
            if len(subset_agents) > 0:
                # roll die to determine if agent will be hospitalised
                mask = np.random.random(len(subset_agents)) < self.isoquar_compliance[hospitalised]
                subset_agents = subset_agents[mask]
                if len(subset_agents) > 0:
                    # hospitalise
                    self.isoquar_agent_by_pos_test_at_hcf(subset_agents, day, hospitalised_bool=1)
                    # did hospitalised agents test positive before? if not, quarantine or self-test close contacts
                    subset_agents = np.setdiff1d(subset_agents, subset_agents[self.simpop_postest_setting[:,subset_agents].tocoo().col])
                    if self.quarantine_hcf_bool > 0 or self.selftest_asymp_contacts_bool > 0:
                        self.identify_close_contacts_for_quar_selftest(subset_agents, day)

            #######

            # get agents currently presenting mild symptoms
            mild_infected_agents = infected_agents[disease_severity_arr==mild]
            if len(mild_infected_agents) > 0:
                # mask agents who are currently overseas
                mask = self.curr_agents_across_the_border[mild_infected_agents] < 1
                mild_infected_agents = mild_infected_agents[mask]
            if len(mild_infected_agents) > 0:
                # remove agents who are currently in isolation
                mask = self.curr_isoquar_arr[isolation, mild_infected_agents].toarray()[0] < 1
                mild_infected_agents = mild_infected_agents[mask]

            # get agents who have onset of symptoms today
            mask = self.simpop_day_of_symptom_onset[virus_type,mild_infected_agents] == day
            subset_agents = mild_infected_agents[mask]
            # roll die to determine if agent will self-isolate upon symptom onset
            prob_arr = np.random.random(len(subset_agents))
            mask = prob_arr < self.isoquar_compliance[self_isolation]
            subset_agents = subset_agents[mask]
            if len(subset_agents) > 0:
                # some agents who have onset of symptoms today may be in quarantine currently
                mask = self.curr_isoquar_arr[quarantine, subset_agents].toarray()[0] > 0
                quarantined_agents = subset_agents[mask]
                if len(quarantined_agents) > 0:
                    # complete their quarantine status
                    self.complete_isoquar(quarantined_agents, quarantine, day, isoquar_reason='complete_quar_to_start_posiso')
                # before starting self isolation
                self.start_isoquar(subset_agents, day, np.array([self.isoquar_period[self_isolation]] * len(subset_agents)), self_isolation, isoquar_reason='onset_selfiso')

            ########

            # get number of days since onset of symptoms of mild infected agents
            days_since_symptom_onset_arr = -self.simpop_day_of_symptom_onset[virus_type,mild_infected_agents] + day
            # select only agents who are past hcf visit delay who would decide on doing self-testing
            mask = days_since_symptom_onset_arr >= get_rand_lognormal_dur(self.hcf_visit_delay, N=len(mild_infected_agents))[0]
            mild_infected_agents = mild_infected_agents[mask]

            # assumption: a symptomatic person who would test at HCF around symptom onset
            if len(mild_infected_agents) > 0:
                # mask agents who had visited HCF before for virus_type or have refused to visit HCF when they could
                mask = self.hcf_sample_collection_day_arr[virus_type, mild_infected_agents] < -1
                subset_agents = mild_infected_agents[mask]

                if len(subset_agents) > 0:
                    # willingness to travel to nearest HCF
                    prob_arr = np.random.random(len(subset_agents))
                    mask = prob_arr < self.voluntary_hcf_visit_prob[subset_agents]
                    # change hcf_sample_collection_day_arr status for agents who refuse to visit HCF
                    othset_agents = subset_agents[~mask]
                    subset_agents = subset_agents[mask]
                    self.hcf_sample_collection_day_arr[virus_type, othset_agents] = -1

                    # agents who doesn't want to go HCF immediately may want to get OTC testing
                    if self.selftest_otc_bool > 0:
                        prob_arr = np.random.random(len(othset_agents))
                        mask = prob_arr < self.selftest_otc_own_prob
                        othset_agents = othset_agents[mask]
                        if len(othset_agents) > 0:
                            symp_agents_who_will_selftest += list(othset_agents)
                            vtype_of_symp_agents_who_will_selftest += list(np.zeros(len(othset_agents), dtype=np.int8) + virus_type)

                    # remaining agents will visit HCF for symptomatic testing
                    if len(subset_agents) > 0:
                        symp_agents_who_will_visit_hcf += list(subset_agents)
                        vtype_of_symp_agents_who_will_visit_hcf += list(np.zeros(len(subset_agents), dtype=np.int8) + virus_type)

        # perform symptomatic testing and other related activities
        symp_agents_who_will_visit_hcf = np.array(symp_agents_who_will_visit_hcf)
        vtype_of_symp_agents_who_will_visit_hcf = np.array(vtype_of_symp_agents_who_will_visit_hcf)

        symp_agents_who_will_selftest = np.array(symp_agents_who_will_selftest)
        vtype_of_symp_agents_who_will_selftest = np.array(vtype_of_symp_agents_who_will_selftest)

        return symp_agents_who_will_visit_hcf, vtype_of_symp_agents_who_will_visit_hcf, symp_agents_who_will_selftest, vtype_of_symp_agents_who_will_selftest

    def review_isoquar(self, int32 day):

        cdef IsoQuar_Type isoquar_status
        cdef np.ndarray agents_arr, subset_agents, today_isoquar_status, mask, days_spent_in_isoquar, prob_still_in_isoquar, agents_still_in_isoquar, agents_who_stop_complying
        cdef np.ndarray postest_agents, testing_results, negtest_agents, isoquar_period_arr
        cdef int32 agent_id

        for agent_id in self.curr_isoquar_arr.tocoo().col:
            if self.curr_isoquar_arr[:,agent_id].sum() > 1:
                print (agent_id)
                print (self.curr_isoquar_arr[:,agent_id])
                print (self.individuals_df[agent_id], day)
                raise Exception("UGHGGGG")

        # get all agents in type of isoquar status (minus hospitalisation which will exit when they recover)
        for isoquar_status in [isolation, quarantine, self_isolation]:
            agents_arr = self.curr_isoquar_arr[isoquar_status,:].tocoo().col
            today_isoquar_status = self.curr_days_in_isoquar[day,agents_arr].toarray()[0]

            # check that there are different kind of isoquar status
            if set(today_isoquar_status) > set([isoquar_status+1, 0]):
                print (isoquar_status, set(today_isoquar_status), set([isoquar_status+1, 0]))
                print (set(today_isoquar_status) > set([isoquar_status+1, 0]))
                raise Exception("NO!")

            # release agents from each isoquar status once they have completed it
            mask = today_isoquar_status == 0
            subset_agents = agents_arr[mask]
            agents_still_in_isoquar = np.setdiff1d(agents_arr, subset_agents)
            if len(subset_agents) > 0:

                # exit testing
                if isoquar_status <= quarantine and self.exit_testing_boolean[isoquar_status] > 0:
                    postest_agents, testing_results = self.compute_testing(subset_agents, day,)
                    negtest_agents = np.setdiff1d(subset_agents, postest_agents)
                    self.total_exit_testing_results[day,isoquar_status,:] = testing_results

                    if len(negtest_agents) > 0:
                        # release agents with negative results
                        self.complete_isoquar(negtest_agents, isoquar_status, day, isoquar_reason='last_day_isoquar')

                    if len(postest_agents) > 0:
                        # continue isolating previously isolated agents who tested positive on exit testing
                        if isoquar_status == isolation and self.max_isolation_period - self.isoquar_period[isolation] > 0:
                            isoquar_period_arr = np.zeros(len(postest_agents), dtype=np.int32) + self.max_isolation_period - self.isoquar_period[isolation]
                            self.start_isoquar(postest_agents, day, isoquar_period_arr, isolation, isoquar_reason='exit_extend_iso')
                        # switch to isolation if agent was previously quarantined
                        elif isoquar_status == quarantine:
                            # first complete their quarantine
                            self.complete_isoquar(postest_agents, isoquar_status, day, isoquar_reason='exit_quar_postest')
                            # switch to isolation
                            isoquar_period_arr = np.zeros(len(postest_agents), dtype=np.int32) + self.isoquar_period[isolation]
                            self.start_isoquar(postest_agents, day, isoquar_period_arr, isolation, isoquar_reason='exit_quar_to_iso')
                else:
                    # complete previous isoquar status
                    self.complete_isoquar(subset_agents, isoquar_status, day, isoquar_reason='last_day_isoquar')

            # adjust adherence for agents still in isoquar
            if len(agents_still_in_isoquar) > 0:
                # get number of days spent in isoquar by counting number of idx
                # compute the number of days agents have spent in isoquar up till today
                days_spent_in_isoquar = (np.ravel(self.curr_days_in_isoquar[:day,agents_still_in_isoquar].tocsc().sum(axis=0))/(isoquar_status+1)).astype(np.int32)
                # get probability agent is still in isoquar
                prob_still_in_isoquar = 1 + days_spent_in_isoquar * (self.endpoint_isoquar_adherence[isoquar_status] - 1)/(self.isoquar_period[isoquar_status])
                # roll die to determine if agent will stay in isoquar
                mask = np.random.random(len(agents_still_in_isoquar)) < prob_still_in_isoquar
                # release agents who did not comply with full isoquar period
                subset_agents = agents_still_in_isoquar[~mask]
                if len(subset_agents) > 0:
                    self.complete_isoquar(subset_agents, isoquar_status, day, isoquar_reason='dropout')

        return

    def quarantine_agents(self, int32 day):

        cdef np.ndarray agents_arr = self.agents_to_quarantine[day,:].tocoo().col
        if len(agents_arr) == 0: # nobody to quarantine today
            return

        # mask agents who are currently in any form of isoquar
        # since they already in any form of isoquar (i.e. have shown symptoms, positively tested or already is in quarantine), makes no sense to ask them to quarantine again
        cdef IsoQuar_Type isoquar_status
        cdef np.ndarray mask
        for isoquar_status in [isolation, quarantine, hospitalised, self_isolation]:
            mask = self.curr_isoquar_arr[isoquar_status, agents_arr].toarray()[0] < 1
            agents_arr = agents_arr[mask]
            if len(agents_arr) == 0: # nobody to quarantine today
                return
        # ignore agents who are currently across the border
        mask = self.curr_agents_across_the_border[agents_arr] < 1
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0: # nobody to quarantine today
            return

        # no need to quarantine if agent has recently completed an isoquar
        cdef np.ndarray last_completed_day_arr, curr_agents_not_quarantined
        cdef int32 agent_id
        cdef object agents_not_quarantined = []
        for isoquar_status in [isolation, quarantine, hospitalised]:
            last_completed_day_arr = self.prev_completed_isoquar_day[isoquar_status,agents_arr]
            # agents excused from quarantine must have undergone isoquar before ...
            mask = last_completed_day_arr >= 0
            curr_agents_not_quarantined = agents_arr[mask]
            if len(curr_agents_not_quarantined) == 0:
                continue
            # ... and the time since their last isoquar must be <= required isoquar period
            last_completed_day_arr = last_completed_day_arr[mask]
            mask = -last_completed_day_arr + day <= self.isoquar_period[isoquar_status]
            curr_agents_not_quarantined = curr_agents_not_quarantined[mask]

            if len(curr_agents_not_quarantined) > 0:
                agents_not_quarantined += list(curr_agents_not_quarantined)

        # remove agents who need not be quarantined again
        agents_arr = np.setdiff1d(agents_arr, np.array(agents_not_quarantined))
        if len(agents_arr) == 0:
            return

        # will agents comply to quarantine requirements?
        mask = np.random.random(len(agents_arr)) < self.isoquar_compliance[quarantine]
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0:
            return

        # either perform daily testing or...
        cdef np.ndarray days_to_test
        cdef int32 d
        if self.daily_test_quarantine_bool > 0:
            # compute the days agents will need to perform self-testing
            days_to_test = np.arange(day, day+self.isoquar_period[quarantine])
            days_to_test = days_to_test[days_to_test<self.total_days]
            for d in days_to_test:
                self.daily_test_quarantine_agents[d,agents_arr] = 1
        # ... perform actual quarantine
        else:
            self.start_isoquar(agents_arr, day, np.array([self.isoquar_period[quarantine]]*len(agents_arr)), quarantine, isoquar_reason='quarantine')

        return

    def daily_test_quarantine(self, int32 day):

        # perform daily test quarantine
        # get agents to perform daily tests
        cdef np.ndarray agents_arr = self.daily_test_quarantine_agents[day,:].tocoo().col
        if len(agents_arr) == 0:
            return

        # if agent is already isolated or hospitalised, no need to test again
        mask = self.curr_isoquar_arr[isolation,agents_arr].toarray()[0] == 0
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0:
            return

        mask = self.curr_isoquar_arr[hospitalised,agents_arr].toarray()[0] == 0
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0:
            return

        # ignore agents who are currently across the border
        mask = self.curr_agents_across_the_border[agents_arr] < 1
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0:
            return

        # perform tests
        cdef np.ndarray postest_agents, testing_results, samp_collect_day, virus_type_arr
        postest_agents, testing_results = self.compute_testing(agents_arr, day)
        self.total_daily_quarantine_testing_results[day] = testing_results

        cdef np.ndarray subset_agents
        if len(postest_agents) > 0:
            # some postest agents may already be in self isolation
            mask = self.curr_isoquar_arr[self_isolation,postest_agents].toarray()[0] > 0
            subset_agents = postest_agents[mask]
            if len(subset_agents) > 0:
                self.complete_isoquar(subset_agents, self_isolation, day, isoquar_reason='complete_selfiso_to_start_posiso')
            # isolate positively tested agents
            self.start_isoquar(postest_agents, day, np.array([self.isoquar_period[isolation]] * len(postest_agents)), isolation, isoquar_reason='daily_test_quar_iso')
            # no need to daily test postest agents after this
            self.daily_test_quarantine_agents[day+1:,postest_agents] = 0

            # daily tested positive agents will most likely not visit a HCF after serving their isolation
            # get virus type of postest_agents
            virus_type_arr = self.curr_virus_type_arr[postest_agents]
            # ignore false positive agents
            subset_agents = postest_agents[virus_type_arr>-1]
            virus_type_arr = virus_type_arr[virus_type_arr>-1]
            # get the sample collection day value of these agents
            samp_collect_day = self.hcf_sample_collection_day_arr[virus_type_arr,subset_agents]
            # for agents where samples have yet been collected (i.e. they have not visited any HCF)
            subset_agents = subset_agents[samp_collect_day < 0]
            virus_type_arr = virus_type_arr[samp_collect_day < 0]
            # ensure that these agents will not visit a HCF later
            self.hcf_sample_collection_day_arr[virus_type_arr, subset_agents] = -1

        return

    def compute_transmissions(self, int32 day, Social_Entity setting, object infect_sparse_mat):

        # get age structured susceptibility indices of susceptible_arr and infected_arr
        cdef np.ndarray susceptible_arr = infect_sparse_mat.tocoo().col
        cdef np.ndarray susceptible_pidx_arr = np.searchsorted(global_age_bins, self.pmap_agebins[susceptible_arr])

        # get relative susceptiblity structured by age
        cdef np.ndarray age_sus = self.age_rel_sus[susceptible_pidx_arr]

        # get contact factor of infected individuals in setting
        cdef np.ndarray infected_arr = infect_sparse_mat.tocoo().row
        cdef np.ndarray unique_infected_arr = np.unique(infected_arr) # get all unique infected agents
        cdef np.ndarray infected_contact_f_arr = self.curr_contact_f_arr[setting][infected_arr]

        # calculate overdisperion factor for infected agents
        cdef np.ndarray unique_overdisperion_f = neg_binomial(mean=self.ind_trans_dist["mean"], shape=self.ind_trans_dist["shape"], n=len(unique_infected_arr)) # create unique overdispersion factor for each agent
        cdef np.ndarray overdisperion_f = unique_overdisperion_f[np.searchsorted(unique_infected_arr, infected_arr)]

        # get contact factor of susceptible individuals in setting
        cdef np.ndarray susceptible_contact_f_arr = self.curr_contact_f_arr[setting][susceptible_arr]

        # get multiplier factor of setting
        cdef float setting_f_multiplier = self.f_setting[setting]

        # get beta_arr
        cdef np.ndarray beta_arr = np.zeros(len(infected_arr), dtype=float) + self.beta
        # get virus type arr of infected
        cdef np.ndarray virus_type_arr = self.curr_virus_type_arr[infected_arr]
        # get mutant f arr
        cdef np.ndarray mutant_f_arr = np.ones(len(infected_arr), dtype=float)
        mutant_f_arr[virus_type_arr > 0] = self.f_mutant_beta
        # get asymp arr
        cdef np.ndarray asymp_agents_arr = self.asymp_infector_arr[virus_type_arr,infected_arr]
        # get asymp f arr
        cdef np.ndarray asymp_f_arr = np.ones(len(infected_arr), dtype=float)
        asymp_f_arr[asymp_agents_arr > 0] = self.f_asymp

        # get viral load f arr
        cdef np.ndarray viral_load_f_arr = self.simpop_vload_factor[day, infected_arr].toarray()[0]

        # get relative immunity to infector virus type
        cdef np.ndarray rel_immunity = self.var_cross_immunity_arr[virus_type_arr, susceptible_arr]

        # get NPIs reduction factor to transmission probability
        cdef float npi_red_f = 1 - self.npi_trans_prob_reduction_f[setting]

        # calculate transmission probability
        cdef np.ndarray trans_prob =  beta_arr * mutant_f_arr * asymp_f_arr * infected_contact_f_arr * viral_load_f_arr * susceptible_contact_f_arr * rel_immunity * age_sus * setting_f_multiplier * overdisperion_f * npi_red_f

        # compute transmissions
        cdef np.ndarray rand_prob_ass = np.random.random(len(trans_prob))
        cdef np.ndarray mask = rand_prob_ass < trans_prob
        cdef np.ndarray infector_arr = infected_arr[mask]
        cdef np.ndarray exposed_arr = susceptible_arr[mask]

        if len(exposed_arr) == 0:
            # return if no one is infected
            return

        # randomly shuffle indices
        cdef np.ndarray shuffled_idx = np.arange(len(exposed_arr))
        np.random.shuffle(shuffled_idx)
        infector_arr = infector_arr[shuffled_idx]
        exposed_arr = exposed_arr[shuffled_idx]

        # remove duplicate exposed agents
        cdef np.ndarray unique_exposed_idx = np.unique(exposed_arr, return_index=True)[1]
        exposed_arr = exposed_arr[unique_exposed_idx]
        infector_arr = infector_arr[unique_exposed_idx]

        self.assign_infection_vars_to_exposed_array(exposed_arr, infector_arr, day, setting)

        return

    def contact_and_transmission(self, int32 day, Social_Entity setting, object contact_layer_arr=None):

        cdef np.ndarray infected_arr, mask
        cdef object susceptible_arr, recovered_wt_arr, infect_sparse_mat
        cdef np.ndarray rand_contacts_arr

        cdef object infected_wt_arr, infected_mt_arr, temp_sparse_mat, all_infected_arr, all_susceptible_arr, curr_susceptible_arr
        cdef np.ndarray entity_arr, entity_susceptible_arr, entity_infected_arr, household_members, all_entity_agents_arr, non_household_church_goers
        cdef int32 i, entity_id, j, agent_id, household_id, sample_n

        if setting == community:
            # get all infected (WT and MT) in the population
            infected_arr = np.union1d(self.curr_seird_arr[infected_wt,:].tocoo().col, self.curr_seird_arr[infected_mt,:].tocoo().col)
            if len(infected_arr) == 0: # no infectious individuals
                return

            # randomly generate random community contact layer array for all infectious agents
            rand_contacts_arr = np.random.poisson(self.mean_community_contact_size, size=len(infected_arr)).astype(np.int32)
            contact_density = rand_contacts_arr.sum()/(len(infected_arr) * self.pop_size)
            contact_layer_arr = sparse.random(len(infected_arr), self.pop_size, density=contact_density, random_state=np.random.default_rng(), format='csr')
            contact_layer_arr.data[:] = 1

            # get all susceptibles and recovered WT agents in the contact layer
            susceptible_arr = contact_layer_arr.multiply(self.curr_seird_arr[susceptible,:].tocsr()).tocoo()
            recovered_wt_arr = contact_layer_arr.multiply(self.curr_seird_arr[recovered_wt,:].tocsr()).tocoo()

            # mask any infected agents who are not in contact with susceptibles/recovered_wt
            mask = np.concatenate((susceptible_arr.row, recovered_wt_arr.row), axis=None)
            infected_arr = infected_arr[mask]

            # generate infection sparse matrix linking infected to susceptibles/recovered_wt
            susceptible_arr = np.concatenate((susceptible_arr.col, recovered_wt_arr.col), axis=None)
            if len(susceptible_arr) == 0: # no susceptible individuals
                return
            infect_sparse_mat = sparse.coo_matrix((np.ones(len(infected_arr), dtype=np.int32), (infected_arr, susceptible_arr))).tocsr()

        else:
            if setting == household:
                contact_layer_arr = self.household_contact_layer_arr
            elif setting != bars:
                contact_layer_arr = self.social_contact_layer_arr
            else: # bars
                if contact_layer_arr == None:
                    raise Exception('contact_layer_arr must be provided for bars') # dynamic layer

            # get all entities of setting type
            if setting == household or setting == bars:
                entity_arr = np.unique(contact_layer_arr.tocoo().row)
            else:
                entity_arr = np.sort(self.entity_type_to_ids[setting])

            # create a temporary array of all entities with infected individuals
            temp_sparse_mat = contact_layer_arr[entity_arr,:]

            # get all infected agents in the contact layer
            infected_wt_arr = temp_sparse_mat.multiply(self.curr_seird_arr[infected_wt,:].tocsr())
            infected_mt_arr = temp_sparse_mat.multiply(self.curr_seird_arr[infected_mt,:].tocsr())

            # get all susceptibles and recovered WT individuals in the contact layer with infected agents
            susceptible_arr = temp_sparse_mat.multiply(self.curr_seird_arr[susceptible,:].tocsr())
            recovered_wt_arr = temp_sparse_mat.multiply(self.curr_seird_arr[recovered_wt,:].tocsr())

            all_infected_arr = []
            all_susceptible_arr = []

            # for each entity...
            for i, entity_id in enumerate(entity_arr):

                if setting == church:
                    # for churches
                    # get all individuals at current church
                    all_entity_agents_arr = contact_layer_arr[entity_id,:].tocoo().col

                    # link infected_wt to susceptibles only
                    entity_infected_arr = infected_wt_arr[i].tocoo().col # get all infected linked to entity
                    entity_susceptible_arr = susceptible_arr[i].tocoo().col # get all susceptibles linked to entity

                    if len(entity_infected_arr) > 0 and len(entity_susceptible_arr) > 0:

                        for j, agent_id in enumerate(entity_infected_arr): # for each infected agent
                            # link them to household members attending the same church who are also susceptibles
                            household_id = self.pmap_households[agent_id]
                            household_members = self.population_arr[self.pmap_households==household_id]
                            curr_susceptible_arr = list(household_members[np.isin(household_members, entity_susceptible_arr)])

                            # random contacts with members of the same church other than household contacts
                            sample_n = np.random.poisson(self.mean_rand_church_contact_size)
                            while sample_n <= 0:
                                sample_n = np.random.poisson(self.mean_rand_church_contact_size)
                            non_household_church_goers = np.setdiff1d(all_entity_agents_arr, household_members)
                            if len(non_household_church_goers) > sample_n:
                                rand_contacts_arr = np.random.choice(non_household_church_goers, sample_n, replace=False)
                            else:
                                rand_contacts_arr = non_household_church_goers
                            curr_susceptible_arr += list(rand_contacts_arr[np.isin(rand_contacts_arr, entity_susceptible_arr)])

                            all_infected_arr += [agent_id] * len(curr_susceptible_arr)
                            all_susceptible_arr += curr_susceptible_arr

                    # link infected_mt to both susceptibles and recovered_wt
                    entity_infected_arr = infected_mt_arr[i].tocoo().col
                    entity_susceptible_arr = np.concatenate((susceptible_arr[i].tocoo().col, recovered_wt_arr[i].tocoo().col), axis=None)

                    if len(entity_infected_arr) > 0 and len(entity_susceptible_arr) > 0:

                        for j, agent_id in enumerate(entity_infected_arr): # for each infected agent
                            # link them to household members attending the same church who are also susceptibles
                            household_id = self.pmap_households[agent_id]
                            household_members = self.population_arr[self.pmap_households==household_id]
                            curr_susceptible_arr = list(household_members[np.isin(household_members, entity_susceptible_arr)])

                            # random contacts with members of the same church other than household contacts
                            sample_n = np.random.poisson(self.mean_rand_church_contact_size)
                            while sample_n <= 0:
                                sample_n = np.random.poisson(self.mean_rand_church_contact_size)
                            non_household_church_goers = np.setdiff1d(all_entity_agents_arr, household_members)
                            if len(non_household_church_goers) > sample_n:
                                rand_contacts_arr = np.random.choice(non_household_church_goers, sample_n, replace=False)
                            else:
                                rand_contacts_arr = non_household_church_goers
                            curr_susceptible_arr += list(rand_contacts_arr[np.isin(rand_contacts_arr, entity_susceptible_arr)])

                            all_infected_arr += [agent_id] * len(curr_susceptible_arr)
                            all_susceptible_arr += curr_susceptible_arr

                else:
                    # link infected_wt to susceptibles only
                    entity_susceptible_arr = susceptible_arr[i].tocoo().col
                    entity_infected_arr = infected_wt_arr[i].tocoo().col

                    if len(entity_infected_arr) > 0 and len(entity_susceptible_arr) > 0:
                        all_infected_arr += list(np.repeat(entity_infected_arr, len(entity_susceptible_arr)))
                        all_susceptible_arr += list(np.tile(entity_susceptible_arr, len(entity_infected_arr)))

                    # link infected_mt to both susceptibles and recovered_wt
                    entity_susceptible_arr = np.concatenate((susceptible_arr[i].tocoo().col, recovered_wt_arr[i].tocoo().col), axis=None)
                    entity_infected_arr = infected_mt_arr[i].tocoo().col
                    if len(entity_infected_arr) > 0 and len(entity_susceptible_arr) > 0:
                        all_infected_arr += list(np.repeat(entity_infected_arr, len(entity_susceptible_arr)))
                        all_susceptible_arr += list(np.tile(entity_susceptible_arr, len(entity_infected_arr)))

            if len(all_infected_arr) == len(all_susceptible_arr) == 0: # no susceptible and infected individuals
                return

            infect_sparse_mat = sparse.coo_matrix((np.ones(len(all_infected_arr), dtype=np.int32), (all_infected_arr, all_susceptible_arr))).tocsr()

        # compute transmission within setting
        self.compute_transmissions(day, setting, infect_sparse_mat)

        return

    def create_daily_bar_contacts(self):

        # create daily random bar contacts for individuals >= min_age_visiting_bars
        # get all eligible agents (those not in any form of isoquar or is currently overseas)
        cdef np.ndarray target_agents = self.population_arr[self.pmap_age>=self.min_age_visiting_bars]
        cdef np.ndarray mask = self.curr_agents_across_the_border[target_agents] < 1
        target_agents = target_agents[mask]

        cdef IsoQuar_Type isoquar_status
        for isoquar_status in [isolation, quarantine, hospitalised, self_isolation]:
            mask = self.curr_isoquar_arr[isoquar_status,target_agents].toarray()[0] < 1
            target_agents = target_agents[mask]

        # determine if agents will go to bar based on daily probability of visiting bar
        cdef np.ndarray prob_arr = np.random.random(size=len(target_agents))
        target_agents = target_agents[prob_arr < self.bar_visit_per_week/7]

        # create bars size array
        cdef int32 N = len(target_agents)
        cdef np.ndarray bars_N_arr = np.random.poisson(self.mean_bars_contact_size, size=int(np.ceil(N/self.mean_bars_contact_size)))
        bars_N_arr = np.delete(bars_N_arr, np.where(bars_N_arr==0)[0]) # remove any size entries that are zero
        # compute difference between sum of bars_N_arr and required pop_size
        cdef int32 bars_N_diff = bars_N_arr.sum() - N

        cdef int32 bars_N_arr_sum
        # if difference is < 0, add more individuals until we have excess
        while bars_N_diff < 0:
            bars_N_arr = np.array(list(bars_N_arr) + list(np.random.poisson(self.mean_bars_contact_size, size=int(np.ceil(abs(bars_N_diff)/self.mean_bars_contact_size)))))
            bars_N_arr = np.delete(bars_N_arr, np.where(bars_N_arr==0)[0])
            bars_N_arr_sum = bars_N_arr.sum()
            bars_N_diff = bars_N_arr_sum - N
        # remove bars with excess individuals
        while bars_N_diff > 0:
            bars_N_arr = bars_N_arr[1:]
            bars_N_arr_sum = bars_N_arr.sum()
            bars_N_diff = bars_N_arr_sum - N

        if bars_N_diff < 0: # add remaining difference
            bars_N_arr = np.array(list(bars_N_arr)+[np.abs(bars_N_diff)])

        # shuffle target_agents
        np.random.shuffle(target_agents)

        # populate class
        cdef int32 prev_i = 0
        cdef int32 bar_id_counter = 0
        cdef object row_ind = []

        for N in bars_N_arr:
            row_ind += [bar_id_counter] * N
            bar_id_counter += 1 # create entity (class)
            prev_i += N

        cdef object contact_layer_arr = sparse.coo_matrix((np.ones(bars_N_arr.sum(), dtype=np.int32), (np.array(row_ind), target_agents)), shape=(bar_id_counter, self.pop_size)).tocsr()

        return contact_layer_arr

    def community_testing(self, int32 day, int32 weekday, int32 week_nr, Social_Entity setting, object contact_layer_arr=None):

        # get testing strategy
        cdef object testing_strategy
        try:
            testing_strategy = self.testing_strategies[setting]
        except:
            raise Exception ("Testing strategy not available for setting %s"%(setting))

        cdef object agents_arr = []
        cdef np.ndarray mask
        cdef IsoQuar_Type isoquar_status

        cdef np.ndarray postest_agents, testing_results, isoquar_period_arr
        cdef np.ndarray immd_close_contacts_arr, entity_id_arr, social_contacts_arr
        cdef int8 quarantine_bool, select_high_entities_bool

        cdef int32 teacher_id, school_id, sample_n, sampled_entities_pop_size, test_freq_week, avail_tests_n, i
        cdef int32 tests_4_teachers = 0
        cdef np.ndarray subset_agents, children_agents, school_arr, temp_arr, sampled_entities_arr , entity_to_size
        cdef np.ndarray idx_arr, samp_f_arr
        cdef object testing_density, k, v, subset_entities, entities_tested_last_week

        if setting == community:
            # random community testing (legacy code)
            agents_arr = self.population_arr
            # randomly choose agents for community testing
            agents_arr = np.random.choice(agents_arr, np.around(testing_strategy['unlimit_percent_tested'] * len(agents_arr)).astype(np.int32), replace=False)

        else:

            #### Get all entity IDs for each setting ####

            if setting == household:
                # random household testing
                # get all households
                entity_id_arr = np.unique(self.household_contact_layer_arr.tocoo().row)
                # get contact layer array
                contact_layer_arr = self.household_contact_layer_arr

            elif setting == bars:
                # for bars
                if contact_layer_arr == None:
                    raise Exception("contact layer array must be provided for bars")
                entity_id_arr = np.unique(contact_layer_arr.tocoo().row)

            else:
                contact_layer_arr = self.social_contact_layer_arr
                if setting == school_class: # for schools
                    # final set of entities to be tested depends on if we are testing primary, secondary or both types of schools
                    if testing_strategy['school_to_test'] < 1:
                        # primary school only
                        school_arr = self.school_type_to_schools[0]
                        entity_id_arr = np.array([k for v in [self.school_to_classes[school_id] for school_id in school_arr] for k in v])
                    elif testing_strategy['school_to_test'] > 1:
                        # all schools to be tested
                        school_arr = np.array(self.entity_type_to_ids[school])
                        entity_id_arr = np.array(self.entity_type_to_ids[setting])
                    else:
                        # secondary school only
                        school_arr = self.school_type_to_schools[1]
                        entity_id_arr = np.array([k for v in [self.school_to_classes[school_id] for school_id in school_arr] for k in v])
                else:
                    # for all other social settings
                    entity_id_arr = np.array(self.entity_type_to_ids[setting])

            ###############################################

            ### Unlimited number of tests available ###
            # Fraction of agents tested will depend on given unlimit_percent_tested
            if self.curr_number_of_comm_rdt[setting] < 0:

                if setting == school_class: # if testing in schools
                    # ALL teachers are assumed to be always tested
                    for school_id in self.school_to_teachers.keys():
                        agents_arr += self.school_to_teachers[school_id]

                if setting == household: # if testing households
                    # unlimit_percent_tested is based on fraction of households tested
                    entity_id_arr = np.random.choice(entity_id_arr, np.around(testing_strategy['unlimit_percent_tested'] * len(entity_id_arr)).astype(np.int32), replace=False)
                    samp_f_arr = np.ones(len(entity_id_arr), dtype=float)
                else:
                    # fraction of agents in each entity to be tested given by unlimit_percent_tested input for all other settings
                    samp_f_arr = np.zeros(len(entity_id_arr), dtype=float) + testing_strategy['unlimit_percent_tested']

            ###############################################

            ### limited number of tests available for setting ###
            else:
                # get number of times testing will be performed each week
                test_freq_week = len(testing_strategy['test_days'])

                # get testing density
                testing_density = testing_strategy['density']

                # compute number of tests available today
                avail_tests_n = np.around(self.curr_number_of_comm_rdt[setting]/(test_freq_week - testing_strategy['test_days'].index(weekday))).astype(np.int32)

                # distribute tests to teachers first if we are testing in schools
                if setting == school_class:
                    # check if there are enough tests to be distributed to all teachers regardless of school type
                    #print (avail_tests_n, self.teachers_n)
                    if avail_tests_n >= self.teachers_n:
                        # ALL teachers are assumed to be always tested
                        for school_id in self.school_to_teachers.keys():
                            agents_arr += self.school_to_teachers[school_id]
                        # minus available tests for students
                        avail_tests_n -= self.teachers_n
                        tests_4_teachers = self.teachers_n
                    else:
                        print ('WARNING: Not enough tests for teachers. Students are not tested.')
                        # only a subset of teachers in randomly selected schools will get tested
                        temp_arr = np.array([k for v in [self.school_to_teachers[school_id] for school_id in self.school_to_teachers.keys()] for k in v])
                        agents_arr += list(np.random.choice(temp_arr, avail_tests_n, replace=False))
                        tests_4_teachers = avail_tests_n
                        # no need to test students
                        avail_tests_n = 0

                ### --- low density testing = as equitable as possible across all entities in setting --- ###
                if testing_density == 'low':

                    if setting == school_class: # for schools

                        # if there are still some tests left for students
                        if avail_tests_n > 0:

                            # check to see if we could equitably distribute tests to all classes to be tested (i.e. at least one person will be tested in each class)
                            if avail_tests_n >= len(entity_id_arr):
                                # if so, compute fraction of agents tested for each entity
                                sample_n = np.around(avail_tests_n/len(entity_id_arr)).astype(np.int32)
                                while sample_n * len(entity_id_arr) > avail_tests_n:
                                    if sample_n <= 1:
                                        break
                                    sample_n -= 1
                                samp_f_arr = np.array([min([sample_n/(len(contact_layer_arr[entity_id,:].tocoo().col)-1), 1.]) for entity_id in entity_id_arr], dtype=float)
                                '''print ('***LowSchool', sample_n, (sample_n * len(entity_id_arr)) + len(agents_arr), avail_tests_n)
                                print (len(agents_arr), len(set(agents_arr)))'''
                            else:
                                # randomly select i number of schools to distribute available tests with aim to test as many students across all classes as possible in each school
                                for i in np.arange(1, len(school_arr)+1):
                                    temp_arr = np.random.choice(school_arr, i, replace=False)
                                    sampled_entities_pop_size = sum([self.school_to_student_size[school_id] for school_id in temp_arr])
                                    if i == 1 and sampled_entities_pop_size > avail_tests_n:
                                        # if we only sampled 1 school and the student population size is already bigger than available number of tests
                                        sampled_entities_arr  = temp_arr
                                        break
                                    else:
                                        # otherwise, keep sampling until
                                        if sampled_entities_pop_size <= avail_tests_n:
                                            sampled_entities_arr  = temp_arr
                                        else:
                                            break

                                # change entity_id_arr to classes of selected schools only
                                entity_id_arr = np.array([k for v in [self.school_to_classes[school_id] for school_id in sampled_entities_arr ] for k in v])
                                # check to see if we could equitably distribute tests to all classes in the selected schools
                                if avail_tests_n >= len(entity_id_arr):
                                    # if so, compute fraction of agents tested for each entity
                                    sample_n = np.around(avail_tests_n/len(entity_id_arr)).astype(np.int32)
                                    while sample_n * len(entity_id_arr) > avail_tests_n:

                                        if sample_n <= 1:
                                            break
                                        sample_n -= 1
                                    samp_f_arr = np.array([min([sample_n/(len(contact_layer_arr[entity_id,:].tocoo().col)-1), 1.]) for entity_id in entity_id_arr], dtype=float)

                                    '''print ('***LowSchool', sample_n, (sample_n * len(entity_id_arr)) + len(agents_arr), avail_tests_n)'''

                                else:
                                    # not enough tests for all classes, distribute one for each class until no more tests available
                                    # get size of classes
                                    samp_f_arr = np.array([len(contact_layer_arr[entity_id,:].tocoo().col)-1 for entity_id in entity_id_arr], dtype=float)
                                    samp_f_arr = 1/samp_f_arr
                                    # randomly select classes to be given one test
                                    idx_arr = np.random.choice(np.arange(len(entity_id_arr)), avail_tests_n, replace=False)
                                    entity_id_arr = entity_id_arr[idx_arr]
                                    samp_f_arr = samp_f_arr[idx_arr]

                        else:
                            # otherwise, no need to test students
                            entity_id_arr = np.array([])

                    # all other types of social settings
                    else:
                        # if there are enough tests to distribute to all entities (i.e. at least one person will be tested in each entity)
                        if avail_tests_n >= len(entity_id_arr):
                            # compute fraction of agents tested for each entity
                            sample_n = np.around(avail_tests_n/len(entity_id_arr)).astype(np.int32)
                            while sample_n * len(entity_id_arr) > avail_tests_n:
                                if sample_n <= 1:
                                    break
                                sample_n -= 1
                            samp_f_arr = np.array([min([sample_n/len(contact_layer_arr[entity_id,:].tocoo().col), 1.]) for entity_id in entity_id_arr], dtype=float)

                        else:
                            # not enough tests for all entities, distribute one to each entity until no more tests available
                            # get size of entities
                            samp_f_arr = np.array([len(contact_layer_arr[entity_id,:].tocoo().col) for entity_id in entity_id_arr], dtype=float)
                            samp_f_arr = 1/samp_f_arr
                            # randomly select entities to be given one test
                            idx_arr = np.random.choice(np.arange(len(entity_id_arr)), avail_tests_n, replace=False)
                            entity_id_arr = entity_id_arr[idx_arr]
                            samp_f_arr = samp_f_arr[idx_arr]

                ### --- high density testing = always concentrate all available tests to select set of entities --- ###
                elif re.search("high", testing_density):

                    if setting == school_class: # for schools, select entities to be tested by schools
                        # if there are still tests left for students
                        if avail_tests_n > 0:

                            if re.search('fixed', testing_density):
                                # fixed schools to test throughout epidemic
                                try:
                                    school_arr = self.prev_tested_entities[week_nr][school]
                                    select_high_entities_bool  = 0
                                except:
                                    try:
                                        school_arr = self.prev_tested_entities[week_nr-1][school]
                                        select_high_entities_bool  = 0
                                    except:
                                        select_high_entities_bool  = 1
                                print ('***', select_high_entities_bool, week_nr, self.prev_tested_entities.keys())

                            else:
                                # randomly select schools to test wholly
                                # randomly select i number of entities to distribute available tests with aim to test all agents in entities
                                try:
                                    school_arr = self.prev_tested_entities[week_nr][school] # same entities have already been tested earlier this week
                                    select_high_entities_bool  = 0
                                except:
                                    select_high_entities_bool  = 1
                                    # schools tested last week will not be tested again
                                    try:
                                        entities_tested_last_week = list(self.prev_tested_entities[week_nr-1][school])
                                    except:
                                        entities_tested_last_week = []
                                    # remove schools that were tested over the last week from consideration
                                    print ('***, tested last week', entities_tested_last_week)
                                    school_arr = np.setdiff1d(school_arr, entities_tested_last_week)

                            if select_high_entities_bool  > 0:
                                # randomly focusing on subset of schools this week OR that we have yet determined the fixed focused schools
                                # choose as many entities as possible testing every agent with the current available tests

                                # sort schools by size
                                school_arr = school_arr[np.argsort([self.school_to_student_size[school_id] for school_id in school_arr])]

                                for i in np.arange(1, len(school_arr)+1):

                                    if re.search('fixed', testing_density):
                                        temp_arr = school_arr[:i]
                                    else:
                                        temp_arr = np.random.choice(school_arr, i, replace=False)
                                    sampled_entities_pop_size = sum([self.school_to_student_size[school_id] for school_id in temp_arr])

                                    if i == 1 and sampled_entities_pop_size > avail_tests_n:
                                        # if we only sample 1 school and the student population size is already bigger than available number of tests
                                        sampled_entities_arr = temp_arr
                                        break
                                    else:
                                        # otherwise, keep sampling until
                                        if sampled_entities_pop_size <= avail_tests_n:
                                            sampled_entities_arr = temp_arr
                                        else:
                                            break

                                school_arr = sampled_entities_arr

                            # save focused schools tested this week
                            if len(school_arr) > 0:
                                try:
                                    self.prev_tested_entities[week_nr][school] = school_arr
                                except:
                                    self.prev_tested_entities[week_nr] = {school:school_arr}

                            # get classes to test
                            entity_id_arr = np.array([k for v in [self.school_to_classes[school_id] for school_id in school_arr] for k in v])
                            sample_n = sum([self.school_to_student_size[school_id] for school_id in school_arr])
                            # if there are not enough tests to test all classes in selected schools
                            if avail_tests_n < sample_n:
                                # choose as many classes as possible where we can test every student
                                for i in np.arange(1, len(entity_id_arr)+1):
                                    if len(np.unique(contact_layer_arr[entity_id_arr[:i],:].tocoo().col)) - i > avail_tests_n:
                                        break
                                entity_id_arr = entity_id_arr[:i-1]

                            '''if len(entity_id_arr) > 0:
                                try:
                                    self.prev_tested_entities[week_nr][setting] = entity_id_arr
                                except:
                                    self.prev_tested_entities[week_nr] = {setting:entity_id_arr}'''

                            print ("***, tested entities", school_arr, entity_id_arr)
                            # test everyone in sampled schools
                            samp_f_arr = np.ones(len(entity_id_arr), dtype=float)

                        else:
                            # otherwise, no need to test students
                            entity_id_arr = np.array([])

                    # all other types of social settings
                    else:
                        if re.search('fixed', testing_density):
                            # fixed entities to test throughout epidemic
                            try:
                                entity_id_arr = self.prev_tested_entities[week_nr][setting]
                                select_high_entities_bool  = 0
                            except:
                                try:
                                    entity_id_arr = self.prev_tested_entities[week_nr-1][setting]
                                    select_high_entities_bool  = 0
                                except:
                                    select_high_entities_bool  = 1
                            print ('***', select_high_entities_bool, week_nr, self.prev_tested_entities.keys())
                        else:
                            # randomly select i number of entities to distribute available tests with aim to test all agents in entities
                            try:
                                entity_id_arr = self.prev_tested_entities[week_nr][setting] # same entities have already been tested earlier this week
                                select_high_entities_bool  = 0
                            except:
                                select_high_entities_bool  = 1
                                # entities tested in the previous week will not be tested again
                                if setting != bars: # bars contacts are random each day, hence would not be meaningful to not test previously tested entities
                                    try:
                                        entities_tested_last_week = list(self.prev_tested_entities[week_nr-1][setting])
                                    except:
                                        entities_tested_last_week = []

                                    # remove entities that were tested over the last week from consideration
                                    print ('***, tested last week', entities_tested_last_week)
                                    entity_id_arr = np.setdiff1d(entity_id_arr, entities_tested_last_week)

                        if select_high_entities_bool  > 0:
                            # randomly focusing on subset of entities this week OR that we have yet determined the fixed focused entities
                            # choose as many entities as possible testing every agent with the current available tests

                            # sort entities by size
                            entity_id_arr = entity_id_arr[np.argsort([len(np.unique(contact_layer_arr[i,:].tocoo().col)) for i in entity_id_arr])]

                            for i in np.arange(1, len(entity_id_arr)+1):

                                if re.search('fixed', testing_density):
                                    temp_arr = entity_id_arr[:i]
                                else:
                                    temp_arr = np.random.choice(entity_id_arr, i, replace=False)
                                sampled_entities_pop_size = len(np.unique(contact_layer_arr[temp_arr,:].tocoo().col))

                                if i == 1 and sampled_entities_pop_size > avail_tests_n:
                                    # not enough tests for even one entity
                                    entity_id_arr = np.array([])
                                    break
                                else:
                                    # otherwise, keep sampling until
                                    if sampled_entities_pop_size <= avail_tests_n:
                                        sampled_entities_arr  = temp_arr
                                    else:
                                        entity_id_arr = sampled_entities_arr
                                        break

                        '''# save entities tested this week
                        if len(entity_id_arr) > 0:
                            try:
                                self.prev_tested_entities[week_nr][setting] = entity_id_arr
                            except:
                                self.prev_tested_entities[week_nr] = {setting:entity_id_arr}'''

                        print ("***, tested entities", entity_id_arr)
                        samp_f_arr = np.ones(len(entity_id_arr), dtype=float)

            ### having sorted entity_id_arr and samp_f_arr ###
            if len(entity_id_arr) > 0:

                try:
                    self.prev_tested_entities[week_nr][setting] = entity_id_arr
                except:
                    self.prev_tested_entities[week_nr] = {setting:entity_id_arr}

                for i, entity_id in enumerate(entity_id_arr):
                    # get all agents in entity
                    subset_agents = contact_layer_arr[entity_id].tocoo().col
                    # only sample from students if testing schools
                    if setting == school_class:
                        subset_agents = np.setdiff1d(subset_agents, agents_arr)
                    # sample agents for testing based on samp_f_arr
                    if samp_f_arr[i] > 0.:
                        if samp_f_arr[i] < 1.:
                            sample_n = np.around(samp_f_arr[i] * len(subset_agents)).astype(np.int32)
                            subset_agents = np.random.choice(subset_agents, sample_n, replace=False)
                        agents_arr += list(subset_agents)

            # get all agents to be tested
            agents_arr = np.unique(agents_arr)
            # skip if there is not enough tests to test target agents
            if len(agents_arr) > avail_tests_n + tests_4_teachers:
                agents_arr = []

            print ("***t", 'avail', avail_tests_n + tests_4_teachers, 'stock', self.curr_number_of_comm_rdt[setting], 'entity_n', len(entity_id_arr), 'agents_n', len(agents_arr))

        ###################################################

        cdef np.ndarray prob_arr

        # if there are agents to community screened
        if len(agents_arr) > 0:
            # leave out agents who are currently overseas
            mask = self.curr_agents_across_the_border[agents_arr] < 1
            agents_arr = agents_arr[mask]

            # leave out agents who are already in any form of isoquar
            for isoquar_status in [isolation, quarantine, hospitalised, self_isolation]:
                mask = self.curr_isoquar_arr[isoquar_status, agents_arr].toarray()[0] < 1
                agents_arr = agents_arr[mask]

            if len(agents_arr) > 0:
                # minus test used
                self.curr_number_of_comm_rdt[setting] -= len(agents_arr)
                # compute testing
                postest_agents, testing_results = self.compute_testing(agents_arr, day)
                # save testing results
                self.total_community_testing_results[day,setting,:] = testing_results

                if len(postest_agents) > 0:
                    # will postest_agents comply with isolation requirement?
                    prob_arr = np.random.random(len(postest_agents))
                    mask = prob_arr < self.isoquar_compliance[isolation]
                    # compliant positively tested agents will visit HCF and be isolated
                    postest_agents = postest_agents[mask]

                if len(postest_agents) > 0:
                    # start isolation for positively tested agents
                    isoquar_period_arr = np.zeros(len(postest_agents), dtype=np.int32) + self.isoquar_period[isolation]
                    self.start_isoquar(postest_agents, day, isoquar_period_arr, isolation, isoquar_reason='comm_testing', comm_test_setting=setting)
                    # save setting
                    self.simpop_postest_setting[day,postest_agents] = setting + 1

                    # quarantine immediate close contacts in contact with positively-tested agents
                    quarantine_bool = testing_strategy['quarantine_bool']
                    if quarantine_bool > 0:
                        # get all contacts living in the same households
                        immd_close_contacts_arr = self.household_contact_layer_arr[self.pmap_households[postest_agents]].tocoo().col
                        immd_close_contacts_arr = np.setdiff1d(immd_close_contacts_arr, postest_agents)

                        # household contacts will be informed on the same day
                        if len(immd_close_contacts_arr) > 0 and day + 1 < self.total_days:
                             self.agents_to_quarantine[day+1, immd_close_contacts_arr] = 1

                        if self.quarantine_social_bool > 0:
                            # get all social entities of positively tested agents
                            entity_id_arr = self.social_contact_layer_arr[:,postest_agents].tocoo().row
                            # get corresponding social contacts
                            social_contacts_arr = self.social_contact_layer_arr[entity_id_arr].tocoo().col
                            social_contacts_arr = np.setdiff1d(social_contacts_arr, immd_close_contacts_arr)
                            social_contacts_arr = np.setdiff1d(social_contacts_arr, postest_agents)

                            # social contacts may only be contacted after delay
                            if len(social_contacts_arr) > 0:
                                # quarantine will only be meted if contact tracing delay < quarantine period
                                if self.contact_tracing_delay < self.isoquar_period[quarantine] and (day + 1) < self.total_days and (day + self.contact_tracing_delay) < self.total_days:
                                    if self.contact_tracing_delay == 0:
                                        self.agents_to_quarantine[day + 1, social_contacts_arr] = 1
                                    else:
                                        self.agents_to_quarantine[day + self.contact_tracing_delay, social_contacts_arr] = 1

        return

    def compute_Reff(self, int32 day):

        # computation of daily Reff
        # get all infected individuals
        cdef np.ndarray infected_wt_arr = self.curr_seird_arr[infected_wt,:].tocoo().col
        cdef np.ndarray infected_mt_arr = self.curr_seird_arr[infected_mt,:].tocoo().col

        # minus agents who were infected overseas
        infected_wt_arr = np.setdiff1d(infected_wt_arr, self.agents_infected_across_the_border[0,:].tocoo().col)
        infected_mt_arr = np.setdiff1d(infected_mt_arr, self.agents_infected_across_the_border[1,:].tocoo().col)

        # compute number of people currently infectious
        cdef int32 infectious_wt_n = len(infected_wt_arr)
        cdef int32 infectious_mt_n = len(infected_mt_arr)
        cdef int32 infectious_n = infectious_wt_n + infectious_mt_n

        # compute infectious period of all currently infected individuals
        cdef np.ndarray infectious_period_wt_arr = self.length_of_infectious_period[0,infected_wt_arr]
        cdef np.ndarray infectious_period_mt_arr = self.length_of_infectious_period[1,infected_mt_arr]
        # check
        if (infectious_period_wt_arr < 0).any():
            raise Exception("FUCKALOO")

        cdef np.ndarray infectious_period_arr = np.concatenate((infectious_period_wt_arr, infectious_period_mt_arr), axis=None)

        # compute number of new infections for the day = sum of all incidence array for the day (minus overseas infections)
        cdef int32 new_infections_wt_n = self.setting_incidence_arr[day,1:,0].sum()
        cdef int32 new_infections_mt_n = self.setting_incidence_arr[day,1:,1].sum()
        cdef int32 new_infections_n = new_infections_wt_n + new_infections_mt_n

        # compute Reff
        self.Reff_arr[day,0] = infectious_period_wt_arr.mean()*(new_infections_wt_n/infectious_wt_n) if (infectious_wt_n > 0 and len(infectious_period_wt_arr) > 0) else 0
        self.Reff_arr[day,1] = infectious_period_mt_arr.mean()*(new_infections_mt_n/infectious_mt_n) if (infectious_mt_n > 0 and len(infectious_period_mt_arr) > 0) else 0
        self.Reff_arr[day,2] = infectious_period_arr.mean()*(new_infections_n/infectious_n) if (infectious_n > 0 and len(infectious_period_arr) > 0) else 0

        return

    def initialise_cross_border_trader_agents(self):
        # initialise agents that who would perform cross border travel
        cdef int32 i, agent_id
        # get all employed agents who are not teachers
        cdef np.ndarray pmap_employed = np.array([self.individuals_df[agent_id]["non_teacher_employed_bool"] for agent_id in self.population_arr], dtype=np.int32)
        cdef np.ndarray employed_arr = np.where(pmap_employed>0)[0]

        # randomly select employed agents who will perform periodic cross-border travels
        cdef np.ndarray ind_cols = np.random.choice(employed_arr, np.around(len(employed_arr) * self.cross_border_traders_percent_employment).astype(np.int32), replace=False)
        np.random.shuffle(ind_cols) # randomly shuffle inds

        # rows = frequency of cross-border travels (three types - 1/per day, 1/per week, 1/per month)
        cdef object ind_rows = []
        cdef np.ndarray temp_arr = np.around(self.cross_border_traders_travel_freq_prop * len(ind_cols)).astype(np.int32)
        for i in np.arange(len(temp_arr)):
            if i < len(temp_arr) - 1:
                ind_rows += [i] * temp_arr[i]
            else:
                ind_rows += [i] * (len(ind_cols) - len(ind_rows))
        ind_rows = np.array(ind_rows)
        # create sparse matrix of informal cross border traders (shape travel_freq x pop_size)
        self.cross_border_travelers = sparse.csc_matrix(([1] * len(ind_rows), (ind_rows, ind_cols)), shape=(3, self.pop_size), dtype=np.int8)

        return

    def cross_border_travel(self, int32 day):

        # facilitate cross border travels
        # agents who are currently in any kind of isoquar/across the border/dead will not cross the border today
        cdef np.ndarray mask = self.curr_agents_across_the_border > 0
        cdef np.ndarray agents_not_travelling_today = np.union1d(self.curr_isoquar_arr.tocoo().col, self.population_arr[mask])
        agents_not_travelling_today = np.union1d(agents_not_travelling_today, self.curr_seird_arr[death,:].tocoo().col)

        cdef int32 i, cross_border_period
        cdef np.ndarray agents_to_cross_borders, prob_arr, length_of_stay_idxarr, length_of_stay, day_range
        # daily probability of cross-border travel for each type of travellers
        cdef object i_to_daily_border_crossing_prob = {0:1., 1:1/7, 2:1/28}
        # boolean to require prior 72h test
        cdef int8 prior_72h_test_bool = self.testing_strategies[overseas]['prior_72h_test']

        for i in np.arange(3):
            # compute agents who would travel today
            border_crossing_prob = i_to_daily_border_crossing_prob[i]
            agents_to_cross_borders = np.setdiff1d(self.cross_border_travelers[i].tocoo().col, agents_not_travelling_today)

            # create prob_arr
            prob_arr = np.random.random(size=len(agents_to_cross_borders))
            mask = prob_arr < border_crossing_prob
            agents_to_cross_borders = agents_to_cross_borders[mask]

            # add outbound cross border stats
            self.border_crossing_stats[day, agents_to_cross_borders] += 2

            # compute the amount of time agents would be overseas
            length_of_stay_idxarr = np.where(np.random.multinomial(1, pvals=self.cross_border_traders_length_of_stay_prop, size=len(agents_to_cross_borders)) > 0)[-1]
            length_of_stay = np.zeros(len(length_of_stay_idxarr), dtype=np.int32)

            # for agents returning on the same day, their of length of stay will be zero
            # agents staying between 2 days and 1 week
            length_of_stay[length_of_stay_idxarr==1] = np.random.choice(np.arange(2, 8, dtype=np.int32), len(length_of_stay[length_of_stay_idxarr==1]))
            # agents staying between >1 week and <4 weeks
            length_of_stay[length_of_stay_idxarr==2] = np.random.choice(np.arange(8, 29, dtype=np.int32), len(length_of_stay[length_of_stay_idxarr==2]))

            # save all agents currently across the border
            self.curr_agents_across_the_border[agents_to_cross_borders] = 1

            # save days in which agent is travelling (-1 = day of return)
            mask = length_of_stay == 0
            self.simpop_travel_days[day, agents_to_cross_borders[mask]] = -1

            # save agents travelling >1 day
            mask = length_of_stay > 0
            for agent_id, cross_border_period in zip(agents_to_cross_borders[mask], length_of_stay[mask]):
                day_range = np.arange(day, day+cross_border_period-1)
                day_range = day_range[day_range>=0]
                day_range = day_range[day_range<self.total_days]
                self.simpop_travel_days[day_range, agent_id] = 1
                # prior 72h test required = 2
                if prior_72h_test_bool > 0:
                    day_range = np.arange(day+cross_border_period-3, day+cross_border_period-1)
                    day_range = day_range[day_range>=0]
                    day_range = day_range[day_range<self.total_days]
                    self.simpop_travel_days[day_range, agent_id] = 2
                # day of return
                if day+cross_border_period-1 >= 0 and day+cross_border_period-1 < self.total_days:
                    self.simpop_travel_days[day+cross_border_period-1, agent_id] = -1

        return

    def review_cross_border_travel(self, int32 day):

        ### infection overseas
        # all agents across the border will risk the chance of getting infected
        cdef np.ndarray mask = self.curr_agents_across_the_border > 0
        cdef np.ndarray agents_across_the_border = self.population_arr[mask]

        # get all susceptibles and recovered wt among agents across the border
        mask = self.curr_seird_arr[susceptible, agents_across_the_border].toarray()[0] > 0
        cdef np.ndarray susceptible_arr = agents_across_the_border[mask]

        mask = self.curr_seird_arr[recovered_wt, agents_across_the_border].toarray()[0] > 0
        cdef np.ndarray recovered_wt_arr = agents_across_the_border[mask]

        cdef int8 virus_type
        cdef np.ndarray infected_wt_arr, infected_mt_arr, prob_arr, curr_susceptible_arr
        for virus_type in range(2):
            # consolidate all susceptibles for each virus type
            if virus_type > 0:
                curr_susceptible_arr = np.union1d(susceptible_arr, recovered_wt_arr)
            else:
                curr_susceptible_arr = susceptible_arr
            # create probability array and roll die to determine if agent becomes infected
            prob_arr = np.random.random(len(curr_susceptible_arr))
            mask = prob_arr < self.cross_border_exposure_prob[virus_type]
            if virus_type > 0:
                infected_mt_arr = curr_susceptible_arr[mask]
            else:
                infected_wt_arr = curr_susceptible_arr[mask]

        # consolidate all infected agents
        cdef np.ndarray exposed_arr = np.hstack((infected_wt_arr, infected_mt_arr))
        cdef np.ndarray virus_type_arr = np.hstack(([0]*len(infected_wt_arr), [1]*len(infected_mt_arr)))

        # randomly shuffle indices
        cdef np.ndarray shuffled_idx = np.arange(len(exposed_arr))
        np.random.shuffle(shuffled_idx)
        exposed_arr = exposed_arr[shuffled_idx]
        virus_type_arr = virus_type_arr[shuffled_idx]

        # remove duplicate exposed agents
        cdef np.ndarray unique_exposed_idx = np.unique(exposed_arr, return_index=True)[1]
        exposed_arr = exposed_arr[unique_exposed_idx]
        virus_type_arr = virus_type_arr[unique_exposed_idx]

        for virus_type in range(2):
            mask = virus_type_arr == virus_type
            if len(exposed_arr[mask]) > 0:
                self.assign_infection_vars_to_exposed_array(exposed_arr[mask], np.repeat(-1, len(exposed_arr[mask])), day, overseas, root_virus_type=virus_type)

        #### agents who are within 72 hours of return will need to get tested (if required)
        # boolean to require prior 72h test
        cdef int8 prior_72h_test_bool = self.testing_strategies[overseas]['prior_72h_test']
        cdef np.ndarray subset_agents, postest_agents, negtest_agents, postest_agents_returning_tomorrow
        if prior_72h_test_bool > 0:
            mask = self.simpop_travel_days[day,agents_across_the_border].toarray()[0] > 1
            subset_agents = agents_across_the_border[mask]

            if len(subset_agents) > 0:
                postest_agents = self.compute_testing(subset_agents, day)[0]
                negtest_agents = np.setdiff1d(subset_agents, postest_agents)

                # if agents are tested positive and they are returning tomorrow
                if len(postest_agents) > 0:
                    if day+1 < self.total_days:
                        mask = self.simpop_travel_days[day+1,postest_agents].toarray()[0] < 0
                        postest_agents_returning_tomorrow = postest_agents[mask]
                        # move back their return day by a day
                        self.simpop_travel_days[day+1, postest_agents_returning_tomorrow] = 2
                        if day+2 < self.total_days:
                            self.simpop_travel_days[day+2, postest_agents_returning_tomorrow] = -1

                # if agents have a negative test, they don't need to be tested again until their return
                if len(negtest_agents) > 0:
                    for agent_id in negtest_agents:
                        for d in np.arange(day+1, self.total_days, dtype=np.int32):
                            if self.simpop_travel_days[d,agent_id] < 0:
                                break
                            elif self.simpop_travel_days[d,agent_id] > 1:
                                self.simpop_travel_days[d,agent_id] = 1

        #### agents who are returning today ####
        mask = self.simpop_travel_days[day,agents_across_the_border].toarray()[0] < 0
        subset_agents = agents_across_the_border[mask]
        # remove them from across the border
        self.curr_agents_across_the_border[subset_agents] = 0
        # add inbound cross border stats
        self.border_crossing_stats[day, subset_agents] += 1
        # testing prior to entry
        cdef np.ndarray isoquar_period_arr, severe_agents
        if self.testing_strategies[overseas]['boolean'] > 0:
            # hospitalise agents who are presenting severe symptoms upon return
            mask = self.simpop_disease_severity[day,subset_agents].toarray()[0] == severe
            severe_agents = subset_agents[mask]
            if len(severe_agents) > 0:
                mask = np.random.random(len(severe_agents)) < self.isoquar_compliance[hospitalised]
                if len(severe_agents[mask]) > 0:
                    # hospitalise and isolate
                    self.isoquar_agent_by_pos_test_at_hcf(severe_agents[mask], day, hospitalised_bool=1)

            # remove severe agents from subset_agents
            subset_agents = np.setdiff1d(subset_agents, severe_agents)
            # compute testing
            postest_agents, testing_results = self.compute_testing(subset_agents, day)
            # save testing results
            self.total_community_testing_results[day,overseas,:] = testing_results
            if len(postest_agents) > 0:
                # will postest_agents comply with isolation requirement?
                prob_arr = np.random.random(len(postest_agents))
                mask = prob_arr < self.isoquar_compliance[isolation]
                postest_agents = postest_agents[mask]

                if len(postest_agents) > 0:
                    # start isolation for compliant positively tested agents
                    isoquar_period_arr = np.zeros(len(postest_agents), dtype=np.int32) + self.isoquar_period[isolation]
                    self.start_isoquar(postest_agents, day, isoquar_period_arr, isolation, isoquar_reason='return_overseas')
                    # save setting
                    self.simpop_postest_setting[day,postest_agents] = overseas + 1

        return

    def assign_hcf(self,):
        """
        Assign healthcare facilities and HCF visit probabilities due to distance to HCF
        """
        # calculate total number of healthcare facilities
        cdef int32 hcf_n = max([2, np.around(self.pop_size/self.pop_to_hcf_ratio).astype(np.int32)])

        print ("%i healthcare facilities in total."%(hcf_n))

        # assign healthcare facilities to households

        if self.fixed_voluntary_hcf_visit_prob >= 0:
            # fix voluntary HCF visit prob
            print ('Fixed voluntary HCF visit probability...')

        # get all household ids
        cdef np.ndarray household_arr = np.sort(np.unique(self.household_contact_layer_arr.tocoo().row))

        cdef int32 hcf_id, household_id, i, prev_i, curr_i
        cdef float prop
        cdef np.ndarray row_ind, col_ind

        self.voluntary_hcf_visit_prob = np.zeros(self.pop_size, dtype=float)

        cdef np.ndarray hcf_arr
        if self.hcf_spatial_dist['type'] == 'uniform':
            # equitably distribute HCFs across households
            hcf_arr = np.linspace(0, len(household_arr), hcf_n+1).astype(np.int32)
        elif self.hcf_spatial_dist['type'] == 'gamma':
            # get gamma distribution density based on given shape and scale parameters
            hcf_arr = np.array([gamma.cdf(i, a=self.hcf_spatial_dist['params'][0], scale=self.hcf_spatial_dist['params'][-1]) for i in np.arange(hcf_n+1)])
            hcf_arr = hcf_arr/hcf_arr[-1]
            hcf_arr = np.around(hcf_arr * len(household_arr)).astype(np.int32)
        else:
            raise Exception('Spatial distribution of type %s not written yet.'%(self.hcf_spatial_dist['type']))

        cdef np.ndarray curr_household_set, ind_arr, temp_arr

        cdef int32 total_N

        for hcf_id, household_id in enumerate(hcf_arr[1:]):
            curr_household_set = np.arange(hcf_arr[hcf_id], household_id)

            # sort households by distance (smaller number = nearer to HCF )
            prev_i = 0
            for i, prop in enumerate(self.dist_of_distance_to_hcf):
                if i < len(self.dist_of_distance_to_hcf) - 1:
                    curr_i = np.around(prop * len(curr_household_set)).astype(np.int32)
                    temp_arr = curr_household_set[prev_i:prev_i+curr_i]
                    prev_i += curr_i
                else:
                    temp_arr = curr_household_set[prev_i:]
                # get agents living in this households of this distance away from nearest HCF
                ind_arr = self.household_contact_layer_arr[temp_arr].tocoo().col

                if self.fixed_voluntary_hcf_visit_prob < 0:
                    # assign probability of visiting HCF
                    self.voluntary_hcf_visit_prob[ind_arr] = self.hcf_visit_probability_dist[i]
                else:
                    # fix voluntary HCF visit prob
                    self.voluntary_hcf_visit_prob[ind_arr] = self.fixed_voluntary_hcf_visit_prob

            # get all individuals in households
            ind_arr = self.household_contact_layer_arr[curr_household_set].tocoo().col
            # assign hcf to individuals
            try:
                row_ind = np.concatenate((row_ind, np.zeros(len(ind_arr), dtype=np.int32) + hcf_id))
            except:
                row_ind = np.zeros(len(ind_arr), dtype=np.int32) + hcf_id

            try:
                col_ind = np.concatenate((col_ind, ind_arr))
            except:
                col_ind = ind_arr

        cdef np.ndarray data = np.ones(len(col_ind), dtype=np.int32)
        # create hcf_contact_layer_arr as a sparse matrix (hcf x agent_id)
        self.hcf_contact_layer_arr = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(hcf_n, self.pop_size)).tocsc()

        return hcf_n

    def stock_up_hcfs(self, int32 total_stocks_to_add, int8 rdt_bool):
        ### stock up hcfs with tests or antivirals ###

        cdef int32 hcf_n
        if rdt_bool > 0:
            hcf_n = len(self.curr_hcf_test_stocks)
        else:
            hcf_n = len(self.curr_hcf_av_stocks)

        cdef int32 i, add_stocks_diff
        cdef np.ndarray hcf_pop_density, stocks_to_add_hcf

        if (rdt_bool > 0 and self.symp_rdt_dist_type == 0) or (rdt_bool < 1 and self.symp_av_dist_type == 0): # proportional to pop size linked to each clinic
            hcf_pop_density = np.array([len(self.hcf_contact_layer_arr[i,:].tocoo().col) for i in np.arange(hcf_n)])
            hcf_pop_density = hcf_pop_density/hcf_pop_density.sum()

            stocks_to_add_hcf = np.around(hcf_pop_density * total_stocks_to_add).astype(np.int32)

        elif (rdt_bool > 0 and self.symp_rdt_dist_type == 1) or (rdt_bool < 1 and self.symp_av_dist_type == 1): # equal number of tests for all clinics
            stocks_to_add_hcf = np.zeros(hcf_n, dtype=np.int32) + np.around(total_stocks_to_add/hcf_n).astype(np.int32)

        elif (rdt_bool > 0 and self.symp_rdt_dist_type == 2) or (rdt_bool < 1 and self.symp_av_dist_type == 2): # all test stocks go to clinic linked to most densely populated region
            hcf_pop_density = np.array([len(self.hcf_contact_layer_arr[i,:].tocoo().col) for i in np.arange(hcf_n)])
            stocks_to_add_hcf = np.zeros(hcf_n, dtype=np.int32)
            stocks_to_add_hcf[np.argmax(hcf_pop_density)] = total_stocks_to_add

        # take care of rounding issues
        cdef int32 hcf_id
        add_stocks_diff = stocks_to_add_hcf.sum() - total_stocks_to_add
        if add_stocks_diff < 0:
            hcf_id = np.random.choice(np.arange(hcf_n))
            stocks_to_add_hcf[hcf_id] += abs(add_stocks_diff)
        elif add_stocks_diff > 0:
            hcf_id = np.random.choice(np.arange(hcf_n)[stocks_to_add_hcf>=abs(add_stocks_diff)])
            stocks_to_add_hcf[hcf_id] -= abs(add_stocks_diff)

        add_stocks_diff = stocks_to_add_hcf.sum() - total_stocks_to_add
        if add_stocks_diff != 0:
            raise Exception('NOPE.')

        if rdt_bool > 0:
            self.curr_hcf_test_stocks += stocks_to_add_hcf
        else:
            self.curr_hcf_av_stocks += stocks_to_add_hcf

        return

    def self_testing(self, int32 day,):

        cdef np.ndarray agents_arr = self.curr_selftest_arr[day,:].tocoo().col
        # no close contacts to self-test or no more test available at healthcare clinics
        if len(agents_arr) == 0 or (self.curr_number_of_hcf_rdt > -1 and self.curr_number_of_hcf_rdt == 0):
            return

        # ignore agents who are currently across the border
        cdef np.ndarray mask = self.curr_agents_across_the_border[agents_arr] < 1
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0:
            return

        # mask agents who are currently in any form of isoquar
        # assumes that since they already in any form of isoquar (i.e. have shown symptoms, positively tested or already is in quarantine), they would not come to self-test (not necessarily true)
        # check that none of the agents is currently in any time of isoquar
        cdef IsoQuar_Type isoquar_status
        for isoquar_status in [isolation, quarantine, hospitalised, self_isolation]:
            mask = self.curr_isoquar_arr[isoquar_status, agents_arr].toarray()[0] < 1
            if len(agents_arr[~mask]) > 0: # already in isoquar, won't self-test again
                self.curr_selftest_arr[day:,agents_arr[~mask]] = 0
            agents_arr = agents_arr[mask]
            if len(agents_arr) == 0:
                return

        cdef np.ndarray selftest_agents_first_day, agents_hcf_arr, agents_at_hcf
        cdef object agents_not_getting_test
        cdef int32 hcf_id, temp_int

        # agents do not go to clinics day by day for self-testing
        if self.selftest_at_clinic_bool < 1 and self.curr_number_of_hcf_rdt > -1: # finite number of tests available
            agents_not_getting_test = []

            selftest_agents_first_day = agents_arr[self.curr_selftest_arr[day,agents_arr].tocoo().data == 1]
            # get clinics linked to agents where they will get tests
            agents_hcf_arr = self.hcf_contact_layer_arr.tocsc()[:,selftest_agents_first_day].tocoo().row
            # withdraw all required number of tests from clinic
            for hcf_id in np.arange(len(self.curr_hcf_test_stocks)):
                # agents at each clinic
                agents_at_hcf = selftest_agents_first_day[agents_hcf_arr == hcf_id]

                if self.curr_hcf_test_stocks[hcf_id] == 0:
                    # no more test stock at clinic
                    agents_not_getting_test += list(agents_at_hcf)
                    continue

                temp_int = np.floor(self.curr_hcf_test_stocks[hcf_id]/self.selftest_period).astype(np.int32)
                if len(agents_at_hcf) > temp_int:
                    # more agents than test stock at clinic
                    # randomly select agents NOT getting test
                    agents_not_getting_test += list(np.random.choice(agents_at_hcf, len(agents_at_hcf) - temp_int, replace=False))
                    print (self.curr_hcf_test_stocks[hcf_id], temp_int * self.selftest_period, '**')
                    self.curr_hcf_test_stocks[hcf_id] -= temp_int * self.selftest_period
                    self.selftest_given_out[day] += temp_int * self.selftest_period
                else:
                    # more tests than agents - all get test
                    self.curr_hcf_test_stocks[hcf_id] -= len(agents_at_hcf) * self.selftest_period
                    self.selftest_given_out[day] += len(agents_at_hcf) * self.selftest_period

            self.curr_number_of_hcf_rdt = self.curr_hcf_test_stocks.sum() # update current hcf test stock total

            # remove agents not getting test/tested
            agents_arr = np.setdiff1d(agents_arr, agents_not_getting_test)
            # if agents never recieve any tests, they will not self-test again
            self.curr_selftest_arr[day:,np.array(agents_not_getting_test)] = 0
            if len(agents_arr) == 0:
                return

        # adherence to self-testing
        cdef np.ndarray prob_arr = np.random.random(len(agents_arr))
        mask = prob_arr < self.curr_selftest_arr[day,agents_arr].tocoo().data
        # agents who won't adhere today won't again in the future
        if len(agents_arr[~mask]) > 0:
            self.curr_selftest_arr[day:,agents_arr[~mask]] = 0
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0:
            return

        if self.selftest_at_clinic_bool > 0 and self.curr_number_of_hcf_rdt > -1: # finite number of tests available
            agents_not_getting_test = []
            # get clinics linked to agents where they will get tests
            agents_hcf_arr = self.hcf_contact_layer_arr.tocsc()[:,agents_arr].tocoo().row
            for hcf_id in np.arange(len(self.curr_hcf_test_stocks)):
                # agents at each clinic
                agents_at_hcf = agents_arr[agents_hcf_arr == hcf_id]

                if self.curr_hcf_test_stocks[hcf_id] == 0:
                    # no more test stock at clinic
                    agents_not_getting_test += list(agents_at_hcf)
                    continue

                if len(agents_at_hcf) > self.curr_hcf_test_stocks[hcf_id]:
                    # more agents than test stock at clinic
                    # randomly select agents NOT getting test
                    agents_not_getting_test += list(np.random.choice(agents_at_hcf, len(agents_at_hcf) - self.curr_hcf_test_stocks[hcf_id], replace=False))

                    self.selftest_given_out[day] += self.curr_hcf_test_stocks[hcf_id]
                    self.curr_hcf_test_stocks[hcf_id] = 0
                else:

                    # more tests than agents - all get test
                    self.selftest_given_out[day] += len(agents_at_hcf)
                    self.curr_hcf_test_stocks[hcf_id] -= len(agents_at_hcf)

            self.curr_number_of_hcf_rdt = self.curr_hcf_test_stocks.sum() # update current hcf test stock total

            # remove agents not getting test
            agents_arr = np.setdiff1d(agents_arr, agents_not_getting_test)
            if len(agents_arr) == 0: # no agents are getting tested
                return

        # perform self-testing
        cdef np.ndarray postest_agents, testing_results, isoquar_period_arr, compliant_agents
        cdef np.ndarray posagents_inf_status, posagents_vtype_arr
        cdef np.ndarray untested_agents = np.array([])
        # self-test remaining agents
        postest_agents, testing_results = self.compute_testing(agents_arr, day)
        # add testing results to self testing counts
        self.total_selftest_results[day,:] += testing_results
        if self.selftest_at_clinic_bool > 0:
            self.total_symp_testing_results[day,:] += testing_results

        if len(postest_agents) > 0:
            # add counts to confirmed cases
            self.reported_daily_case_arr[day, postest_agents] = 1
            # save symptomatic testing for self-tested agents
            self.simpop_postest_setting[day, postest_agents] = -3
            # will no longer go for self-testing
            self.curr_selftest_arr[day:,postest_agents] = 0

            # disburse antiviral
            # identify those currently infected with wt and mt virus
            posagents_inf_status = self.simpop_infection_status[day, postest_agents].toarray()[0]
            posagents_vtype_arr = np.zeros(len(postest_agents), dtype=np.int8) - 1 # uninfected gets vtype = -1
            posagents_vtype_arr[(posagents_inf_status==infected_wt)|(posagents_inf_status==exposed_wt)] = 0
            posagents_vtype_arr[(posagents_inf_status==infected_mt)|(posagents_inf_status==exposed_mt)] = 1
            #print (postest_agents, posagents_vtype_arr)
            self.disburse_av(postest_agents, posagents_vtype_arr, day)

            # will postest_agents comply with isolation requirement?
            prob_arr = np.random.random(len(postest_agents))
            mask = prob_arr < self.isoquar_compliance[isolation]
            # compliant positively tested agents will visit HCF and be isolated
            compliant_agents = postest_agents[mask]

            if len(compliant_agents) > 0:
                # start isolation for positively tested agents
                isoquar_period_arr = np.zeros(len(compliant_agents), dtype=np.int32) + self.isoquar_period[isolation]
                self.start_isoquar(compliant_agents, day, isoquar_period_arr, isolation, isoquar_reason='self_testing')

        return

    def disburse_av(self, np.ndarray agents_arr, np.ndarray virus_type_arr, int32 day):
        # check eligibility and disburse antiviral

        # return if we are out of antivirals
        if (self.curr_hcf_av_stocks.sum() > -1 and self.curr_hcf_av_stocks.sum() == 0):
            return

        # mask agents that are currently already hospitalised (already experiencing severe symptoms)
        cdef np.ndarray mask = self.curr_isoquar_arr[hospitalised, agents_arr].toarray()[0] < 1
        agents_arr = agents_arr[mask]
        if len(agents_arr) == 0:
            return

        # filter for agents who recently recieved treatment
        cdef int32 min_day = max([0, day-self.av_period_before_next_course])
        cdef np.ndarray agents_av_before = agents_arr[self.simpop_agents_w_av[min_day:day,agents_arr].tocoo().col]
        if len(agents_av_before) > 0:
            mask = np.isin(agents_arr, agents_av_before)
            agents_arr = agents_arr[~mask]
            if len(agents_arr) == 0:
                return
            virus_type_arr = virus_type_arr[~mask]

        # check eligibility
        cdef np.ndarray age_mask
        if self.antiviral_req['exclude_children'] > 0:
            # exclude children
            age_mask = self.pmap_age[agents_arr]>=18
            agents_arr = agents_arr[age_mask]
            if len(agents_arr) == 0:
                return
            virus_type_arr = virus_type_arr[age_mask]

        # filter by age and vaccination status
        age_mask = (self.pmap_age[agents_arr] >= self.antiviral_req['age_range'][0])&(self.pmap_age[agents_arr] <= self.antiviral_req['age_range'][-1])
        cdef np.ndarray vacc_mask = self.pmap_vacc_status[agents_arr] <= self.antiviral_req['max_vacc_level']
        # risk mask
        cdef np.ndarray risk_mask = self.pmap_adults_at_risk[agents_arr] > 1 - self.antiviral_req['risk']

        # Logical mode (0 = age|risk|vacc, 1 = (age&vacc)|(risk&vacc), 2 = (age&vacc)|risk, 3 = age|(risk&vacc), 4 = (age&risk)|vacc, 5 = age&risk&vacc, 6 = age|risk)
        if self.antiviral_req['logic_mode'] ==  0:
            agents_arr = agents_arr[age_mask|risk_mask|vacc_mask]
        elif self.antiviral_req['logic_mode'] ==  1:
            agents_arr = agents_arr[(age_mask&vacc_mask)|(risk_mask&vacc_mask)]
        elif self.antiviral_req['logic_mode'] ==  2:
            agents_arr = agents_arr[(age_mask&vacc_mask)|risk_mask]
        elif self.antiviral_req['logic_mode'] ==  3:
            agents_arr = agents_arr[age_mask|(risk_mask&vacc_mask)]
        elif self.antiviral_req['logic_mode'] ==  4:
            agents_arr = agents_arr[(age_mask&risk_mask)|vacc_mask]
        elif self.antiviral_req['logic_mode'] ==  5:
            agents_arr = agents_arr[(age_mask&risk_mask&vacc_mask)]
        else:
            agents_arr = agents_arr[age_mask|risk_mask]

        if len(agents_arr) == 0:
            return

        # filter by day since symptom onset (if actually infected, otherwise we assume onset on the same day)
        if self.antiviral_req['logic_mode'] ==  0:
            virus_type_arr = virus_type_arr[age_mask|risk_mask|vacc_mask]
        elif self.antiviral_req['logic_mode'] ==  1:
            virus_type_arr = virus_type_arr[(age_mask&vacc_mask)|(risk_mask&vacc_mask)]
        elif self.antiviral_req['logic_mode'] ==  2:
            virus_type_arr = virus_type_arr[(age_mask&vacc_mask)|risk_mask]
        elif self.antiviral_req['logic_mode'] ==  3:
            virus_type_arr = virus_type_arr[age_mask|(risk_mask&vacc_mask)]
        elif self.antiviral_req['logic_mode'] ==  4:
            virus_type_arr = virus_type_arr[(age_mask&risk_mask)|vacc_mask]
        elif self.antiviral_req['logic_mode'] ==  5:
            virus_type_arr = virus_type_arr[(age_mask&risk_mask&vacc_mask)]
        else:
            virus_type_arr = virus_type_arr[age_mask|risk_mask]

        # for uninfected and asymptomatic agents, we assumed that their "onset" is today and will get past this filter
        cdef np.ndarray agents_onset_arr = np.zeros(len(agents_arr), dtype=np.int8) + day
        agents_onset_arr[virus_type_arr>=0] = self.simpop_day_of_symptom_onset[virus_type_arr[virus_type_arr>=0], agents_arr[virus_type_arr>=0]]

        # check asymptomatics
        if len(agents_onset_arr[agents_onset_arr<0]) > 0:
            if sum(self.asymp_infector_arr[virus_type_arr[agents_onset_arr<0], agents_arr[agents_onset_arr<0]]) != len(agents_arr[agents_onset_arr<0]):

                print (agents_arr[agents_onset_arr<0][0], virus_type_arr[agents_onset_arr<0][0],)

                print (self.simpop_disease_severity[:,agents_arr[agents_onset_arr<0][0]])
                print (self.simpop_day_of_symptom_onset[:,agents_arr[agents_onset_arr<0][0]])

                print (self.asymp_infector_arr[:,agents_arr[agents_onset_arr<0][0]])

                raise Exception("FUUUUUUK")
        agents_onset_arr[agents_onset_arr<0] = day

        cdef np.ndarray onset_mask = day - agents_onset_arr <= self.antiviral_req['days_since_onset']
        agents_arr = agents_arr[onset_mask]
        if len(agents_arr) == 0:
            return
        virus_type_arr = virus_type_arr[onset_mask]

        # get clinics linked to agents where they will get tests
        cdef np.ndarray agents_hcf_arr, agents_at_hcf
        cdef object agents_not_getting_av
        cdef int32 hcf_id
        if self.curr_hcf_av_stocks.sum() > -1:
            agents_hcf_arr = self.hcf_contact_layer_arr.tocsc()[:,agents_arr].tocoo().row
            agents_not_getting_av = []

            for hcf_id in np.arange(len(self.curr_hcf_av_stocks)):
                # agents at each clinic
                agents_at_hcf = agents_arr[agents_hcf_arr == hcf_id]

                if self.curr_hcf_av_stocks[hcf_id] == 0:
                    # no more test stock at clinic
                    agents_not_getting_av += list(agents_at_hcf)
                    continue

                if len(agents_at_hcf) > self.curr_hcf_av_stocks[hcf_id]:
                    # more agents than test stock at clinic
                    # randomly select agents NOT getting test
                    agents_not_getting_av += list(np.random.choice(agents_at_hcf, len(agents_at_hcf) - self.curr_hcf_av_stocks[hcf_id], replace=False))
                    self.curr_hcf_av_stocks[hcf_id] = 0
                else:
                    # more tests than agents - all get test
                    self.curr_hcf_av_stocks[hcf_id] -= len(agents_at_hcf)

            # filter out agents not getting antivirals
            agents_arr = agents_arr[~np.isin(agents_arr, agents_not_getting_av)]

        # agents get antiviral
        self.simpop_agents_w_av[day, agents_arr] = 1

        # find all agents who are actually infected
        cdef np.ndarray agents_infstatus = self.simpop_infection_status[day,agents_arr].toarray()[0]
        cdef np.ndarray infected_mask = (agents_infstatus>=exposed_wt)&(agents_infstatus<recovered_wt)
        agents_arr = agents_arr[infected_mask]
        if len(agents_arr) == 0:
            return
        virus_type_arr = virus_type_arr[infected_mask]

        # reassign disease severity and symptom period
        self.reassign_dissev_sympperiod(agents_arr, virus_type_arr, day)

        return

    def rds_worker(self, np.ndarray agents_arr, np.ndarray virus_type_arr, int32 day):

        cdef np.ndarray curr_agents_arr, curr_agents_idx, curr_agents_days, curr_agents_dissev, temp_arr, adjusted_symp_period, day_range
        cdef int32 agent, i, symp_onset_day, onset_diff, start_day, cess_day, vload_start_day, last_inf_day
        cdef int8 virus_type
        cdef float last_inf_f
        cdef Infection_Status curr_inf_status, end_inf_status

        for virus_type in np.arange(2):

            curr_agents_arr = agents_arr[virus_type_arr == virus_type]

            if len(curr_agents_arr) == 0:
                continue

            # re-sample period of mild to recovery
            temp_arr = np.array([self.tau_recovery_mild[virus_type][0] * self.av_rom_symp_period, self.tau_recovery_mild[virus_type][-1]])
            adjusted_symp_period = get_rand_lognormal_dur(temp_arr, N=len(curr_agents_arr), min_val=1)[0]
            # get all symptomatic period of agents
            curr_agents_days = self.simpop_disease_severity[:,curr_agents_arr].tocoo().row
            curr_agents_idx = self.simpop_disease_severity[:,curr_agents_arr].tocoo().col
            curr_agents_dissev = self.simpop_disease_severity[:,curr_agents_arr].tocoo().data

            for i in np.unique(curr_agents_idx):
                # for each agent
                agent = curr_agents_arr[i]
                # get day of symptom onset
                symp_onset_day = self.simpop_day_of_symptom_onset[virus_type, agent]
                # get difference between today and symp_onset_day
                onset_diff = day - symp_onset_day

                # if current day is after onset:
                if onset_diff > 0:
                    # day of viral load cessation
                    cess_day = day + adjusted_symp_period[i]
                # curret day = onset
                elif onset_diff == 0:
                    cess_day = symp_onset_day + adjusted_symp_period[i]
                # current day is before onset
                else:
                    cess_day = symp_onset_day + adjusted_symp_period[i]

                if symp_onset_day >= 0:
                    start_day = symp_onset_day
                    if start_day >= self.total_days:
                        continue
                else:
                    start_day = 0
                # change self.simpop_disease_severity
                self.simpop_disease_severity[start_day:,agent] = 0 # reset
                if cess_day < self.total_days:
                    self.simpop_disease_severity[start_day:cess_day, agent] = mild
                else:
                    self.simpop_disease_severity[start_day:, agent] = mild

                # change self.simpop_infection_status
                self.simpop_infection_status[start_day:,agent] = 0 # reset
                if virus_type > 0:
                    curr_inf_status = infected_mt
                    end_inf_status = recovered_mt
                else:
                    curr_inf_status = infected_wt
                    end_inf_status = recovered_wt

                if cess_day < self.total_days:
                    self.simpop_infection_status[start_day:cess_day, agent] = curr_inf_status
                    self.simpop_infection_status[cess_day, agent] = end_inf_status
                else:
                    self.simpop_infection_status[start_day:, agent] = curr_inf_status

                # if current day is after onset:
                if onset_diff > 0:
                    # day of viral load cessation
                    vload_start_day = day
                # curret day = onset
                elif onset_diff == 0:
                    vload_start_day = symp_onset_day
                # current day is before onset
                else:
                    vload_start_day = symp_onset_day
                if vload_start_day >= self.total_days:
                    continue

                # change viral load from day of treatment
                temp_arr = self.simpop_ct_arr[vload_start_day:,agent].tocoo().data
                # update period of infectiousness
                self.length_of_infectious_period[virus_type,agent] -= len(temp_arr[temp_arr<=self.infectious_ct_thres])

                self.simpop_ct_arr[vload_start_day+1:,agent] = 0 # reset
                # linearly interpolate to cessation day from vload_start_day
                temp_arr = np.zeros(cess_day-vload_start_day, dtype=np.int32) + self.simpop_ct_arr[vload_start_day,agent]
                try:
                    temp_arr += np.around(np.arange(cess_day-vload_start_day) * (40-self.simpop_ct_arr[vload_start_day,agent])/((cess_day-1)-vload_start_day)).astype(np.int32)
                except:
                    pass
                temp_arr[temp_arr>40] = 40
                temp_arr[-1] = 40
                if cess_day < self.total_days:
                    self.simpop_ct_arr[vload_start_day:cess_day,agent] = temp_arr
                    # update period of infectiousness
                    self.length_of_infectious_period[virus_type,agent] += len(temp_arr[temp_arr<=self.infectious_ct_thres])
                else:
                    day_range = np.arange(vload_start_day, cess_day, dtype=np.int32)
                    self.simpop_ct_arr[vload_start_day:,agent] = temp_arr[day_range < self.total_days]
                    # update period of infectiousness
                    self.length_of_infectious_period[virus_type,agent] += len(temp_arr[day_range < self.total_days][temp_arr[day_range < self.total_days]<=self.infectious_ct_thres])

                # get last infectious day
                try:
                    last_inf_day = np.arange(vload_start_day, cess_day, dtype=np.int32)[temp_arr<=self.infectious_ct_thres][-1] + 1
                except:
                    # no longer infectious from vload_start_day
                    self.simpop_vload_factor[vload_start_day:,agent] = 0 # reset
                    continue

                """print ('vload_start_day', vload_start_day, cess_day)
                print (np.arange(vload_start_day, cess_day, dtype=np.int32), temp_arr)
                print (np.arange(vload_start_day, cess_day, dtype=np.int32)[temp_arr<=self.infectious_ct_thres], last_inf_day)"""

                # compute infectiousness factor based on Ct values on last infectious day
                last_inf_f = self.max_vload_f + ((temp_arr[np.arange(vload_start_day, cess_day, dtype=np.int32) == last_inf_day-1][0] - self.simpop_ct_arr[symp_onset_day,agent]) * ((1 - self.max_vload_f)/(self.infectious_ct_thres - self.simpop_ct_arr[symp_onset_day,agent])))
                # change self.simpop_vload_factor
                self.simpop_vload_factor[vload_start_day+1:,agent] = 0 # reset
                # linearly interpolate to cessation day from vload_start_day
                temp_arr = np.zeros(last_inf_day-vload_start_day, dtype=float) + self.simpop_vload_factor[vload_start_day,agent]
                try:
                    temp_arr += np.arange(last_inf_day-vload_start_day, dtype=float) * (last_inf_f-self.simpop_vload_factor[vload_start_day,agent])/((last_inf_day-1)-vload_start_day)
                except:
                    pass
                temp_arr[temp_arr<1] = 1.
                temp_arr[-1] = last_inf_f
                if last_inf_day < self.total_days:
                    self.simpop_vload_factor[vload_start_day:last_inf_day,agent] = temp_arr
                    """print ('ok', temp_arr)
                    print (self.simpop_vload_factor[vload_start_day:last_inf_day,agent])"""
                else:
                    day_range = np.arange(vload_start_day, last_inf_day, dtype=np.int32)
                    self.simpop_vload_factor[vload_start_day:,agent] = temp_arr[day_range < self.total_days]
                    """print ('***', temp_arr, temp_arr[:last_inf_day], last_inf_day)
                    print (self.simpop_vload_factor[vload_start_day:,agent])"""

        return

    def reassign_dissev_sympperiod(self, np.ndarray agents_arr, np.ndarray virus_type_arr, int32 day):

        # find agents who will develop disease
        cdef np.ndarray dissev_arr = self.simpop_disease_severity[day:,agents_arr].tocoo().data
        cdef np.ndarray agents_idx = self.simpop_disease_severity[day:,agents_arr].tocoo().col

        cdef np.ndarray severe_agents = np.unique(agents_arr[agents_idx[dissev_arr == severe]])
        cdef np.ndarray mild_agents = np.unique(agents_arr[agents_idx[dissev_arr == mild]])
        mild_agents = np.setdiff1d(mild_agents, severe_agents)

        cdef np.ndarray infstatus_agents_arr, infstatus_virus_type_arr, prob_idx, severe_prob, agents_vacc_status, cond_severe_prob, severe_bool, sev_prob_adjustment, agents_risk
        cdef np.ndarray sev2mild_agents, vtype_sev2mild_agents

        if len(mild_agents) > 0:
            infstatus_agents_arr = agents_arr[np.isin(agents_arr, mild_agents)]
            infstatus_virus_type_arr = virus_type_arr[np.isin(agents_arr, mild_agents)]

            self.rds_worker(infstatus_agents_arr, infstatus_virus_type_arr, day)
            self.simpop_agents_av_benefit[infstatus_virus_type_arr, infstatus_agents_arr] = 1

        if len(severe_agents) > 0:
            infstatus_agents_arr = agents_arr[np.isin(agents_arr, severe_agents)]
            infstatus_virus_type_arr = virus_type_arr[np.isin(agents_arr, severe_agents)]

            # for severe agents, reassess if they will still be
            # get the age bin index of agent's age
            prob_idx = np.floor(self.pmap_agebins[infstatus_agents_arr]/5).astype(np.int32)
            # get probability agent will have severe disease outcomes
            severe_prob = self.p_severe[prob_idx]
            # multiply by severity prob factor if mutant virus
            severe_prob = severe_prob * self.f_mutant_severe_prob
            # multiply by risk factor
            agents_risk = self.pmap_adults_at_risk[infstatus_agents_arr]
            if len(severe_prob[agents_risk>0]) > 0:
                severe_prob[agents_risk>0] *= self.f_risk_severe_prob
            # if vaccinated, reduce severe prob by given protection
            agents_vacc_status = self.pmap_vacc_status[infstatus_agents_arr]
            if len(severe_prob[agents_vacc_status > 0]) > 0:
                severe_prob[agents_vacc_status > 0] *= 1 - self.vacc_severe_f_arr[infstatus_virus_type_arr[agents_vacc_status > 0], agents_vacc_status[agents_vacc_status > 0]-1]
            # adjust severe probability with effects from antiviral
            sev_prob_adjustment = self.av_or_red_sev / (1 - severe_prob + (severe_prob * self.av_or_red_sev))
            severe_prob *= sev_prob_adjustment
            # conditional proabability of severe disease given symptomatic
            cond_severe_prob = severe_prob/self.p_symptomatic[prob_idx]
            # get severe boolean
            severe_bool = np.random.random(len(infstatus_agents_arr)) < cond_severe_prob
            # identify those that would no longer have severe disease
            sev2mild_agents = infstatus_agents_arr[~severe_bool]
            vtype_sev2mild_agents = infstatus_virus_type_arr[~severe_bool]

            if len(sev2mild_agents) > 0:
                self.rds_worker(sev2mild_agents, vtype_sev2mild_agents, day)
                self.simpop_agents_av_benefit[infstatus_virus_type_arr, infstatus_agents_arr] = 2

        return

    def execute(self, int32 total_days, object start_date='2021-01-01', int8 verbose=1):

        # declare local variables
        cdef int8 sec_school_bool
        cdef int32 day, weekday, agent_id, entity_id
        cdef object day_obj, day_bars_contact_layer_arr
        cdef np.ndarray mask, entity_arr, infected_arr, susceptible_arr
        cdef Social_Entity setting
        # odd week boolean
        cdef int8 odd_week_bool = 0

        # pmap
        print ('Preparing simulation run...')
        self.pmap_households = np.array([self.individuals_df[agent_id]['household_id'] for agent_id in self.population_arr], dtype=np.int32)
        # get age map of individuals
        self.pmap_agebins = np.array([self.individuals_df[agent_id]['agent_age_bin'] for agent_id in self.population_arr], dtype=np.int32)
        # agent age array
        self.pmap_age = np.array([self.individuals_df[agent_id]["agent_age"] for agent_id in self.population_arr], dtype=np.int32)
        # compute school to teacher
        self.school_to_teachers = {}
        self.teachers_n = 0
        # get school arr
        entity_arr = np.array(self.entity_type_to_ids[school])
        for agent_id in self.population_arr:
            if self.individuals_df[agent_id]["formal_employed_bool"] > 0 and self.individuals_df[agent_id]['school_bool'] > 0:
                self.teachers_n += 1 # count total number of teachers
                entity_id = list(set(self.individuals_df[agent_id]['social_id_arr'])&set(entity_arr))[0]
                try:
                    self.school_to_teachers[entity_id].append(agent_id)
                except:
                    self.school_to_teachers[entity_id] = [agent_id]

        # compute student size of each school
        self.school_to_student_size = {}
        cdef int32 teach_n
        for sec_school_bool in range(2):
            for entity_id in self.school_type_to_schools[sec_school_bool]:
                # get all classes of each school
                entity_arr = np.array(self.school_to_classes[entity_id])
                self.school_to_student_size[entity_id] = len(self.social_contact_layer_arr[entity_arr,:].tocoo().col)
                try:
                    teach_n = len(self.school_to_teachers[entity_id])
                except:
                    teach_n = 0
                self.school_to_student_size[entity_id] -= teach_n

        # set total number of days to simulate as a global variable
        self.total_days = total_days
        # set up date range to run simulation
        cdef object date_range = pd.date_range(datetime.fromisoformat(start_date), periods=self.total_days).tolist()

        # initialise arrays
        # boolean array denoting current disease progression of agents
        self.curr_seird_arr = sparse.dok_matrix((8, self.pop_size), dtype=np.int8)
        # variant cross immunity factor of agents
        self.var_cross_immunity_arr = np.ones((2, self.pop_size), dtype=float)
        # current virus type infecting agents (-1 = not infected by any virus)
        self.curr_virus_type_arr = np.zeros(self.pop_size, dtype=np.int8) - 1
        # boolean array denoting asymptomatic infectors
        self.asymp_infector_arr = np.zeros((2, self.pop_size), dtype=np.int8)
        # disease periods array (virus_type x population x disease period types)
        self.simpop_disease_periods_arr = np.zeros((2, self.pop_size, 7), dtype=np.int32)
        # array to save fated disease severity of agents
        self.fated_symp_severity_arr = sparse.dok_matrix((2, self.pop_size), dtype=np.int8)
        # array to save agents that are fated to die
        self.fated_to_die_arr = sparse.dok_matrix((2, self.pop_size), dtype=np.int8)
        # array to save agents vaccination status
        self.pmap_vacc_status = np.zeros(self.pop_size, dtype=np.int8)

        # incidence array (total_days x settings x variant)
        self.setting_incidence_arr = np.zeros((self.total_days, 9, 2), dtype=np.int32)
        # array for Reff computed for each day (WT, MT, overall)
        self.Reff_arr = np.zeros((self.total_days, 3), dtype=float)

        # infection status of agents over time
        self.simpop_infection_status = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)
        # disease severity of agents over time
        self.simpop_disease_severity = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)
        # simulated Ct values of infected population
        self.simpop_ct_arr = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int32)
        # viral load factor of infected population
        self.simpop_vload_factor = sparse.dok_matrix((self.total_days, self.pop_size), dtype=float)
        # day of symptom onset of agents
        self.simpop_day_of_symptom_onset = np.zeros((2, self.pop_size), dtype=np.int32) - 100
        # save length of infectious period of agents
        self.length_of_infectious_period = np.zeros((2, self.pop_size), dtype=np.int32) - 1
        # save day agents got an antiviral
        self.simpop_agents_w_av = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)
        # save agents who benefited from antiviral (virus_type x agent; 1=mild, 2=severe2mild)
        self.simpop_agents_av_benefit = sparse.dok_matrix((2, self.pop_size), dtype=np.int8)

        # boolean array denoting who is in isoquar
        self.curr_isoquar_arr = sparse.dok_matrix((4, self.pop_size), dtype=np.int8)
        # array saving the number of days agent spent in current isoquar
        self.curr_days_in_isoquar = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)
        # output array saving no. of days agent spent in different isoquar
        self.simpop_isoquar_arr = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)
        # ouput array saving when agent was tested positive and where (setting + 1, -1 = symptomatic testing, -2 = noncovid symptomatic testing, -3 = self-test due to contact with pos agents, -4 = self-test OTC)
        self.simpop_postest_setting = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int32)
        # agent current contact rate multiplier array
        self.curr_contact_f_arr = np.ones((9, self.pop_size), dtype=float) # shape = domestic Social_Entity x pop_size
        # array saving the days to start quarantining agents
        self.agents_to_quarantine = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int32)
        # array tracking the days to whichs agents will self-test instead of quarantine
        self.daily_test_quarantine_agents = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int32)
        # array saving the day agents completed previous isoquar
        self.prev_completed_isoquar_day = np.zeros((4, self.pop_size), dtype=np.int32) - 1

        # array saving the day agents have to go for self-testing + their adherence probabilities to doing so
        self.curr_selftest_arr = sparse.dok_matrix((self.total_days, self.pop_size), dtype=float)
        # counter for selftest given out
        self.selftest_given_out = np.zeros(self.total_days, dtype=np.int32)

        # border crossing stats (inbound, outbound)
        self.border_crossing_stats = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)
        # boolean array denoting who is currently overseas
        self.curr_agents_across_the_border = np.zeros(self.pop_size, dtype=np.int8)
        # boolean matrix of agents infected across the border (virus_type x pop_size)
        self.agents_infected_across_the_border = sparse.dok_matrix((2, self.pop_size), dtype=np.int8)
        # boolean matrix saving travel days of agents
        self.simpop_travel_days = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)

        # array to save day when samples were collected from agents by HCF visits
        self.hcf_sample_collection_day_arr = np.zeros((2, self.pop_size), dtype=np.int32) - 2 # base = -2 (-1 = will NOT visit) HCF even if symptomatic; >-1 = actual day of visit to HCF) - shape = virus_type x pop_size
        # array saving COVID symptomatic agents who were not tested due to a lack of test (total days x pop_size)
        self.untested_covid_symp_lack_of_test_arr = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)
        # array saving number of non-COVID agents seeking tests due to lack of test (total days)
        self.untested_non_covid_symp_lack_of_test = np.zeros(self.total_days, dtype=np.int32)
        # array to save number of daily confirmed cases (1=confirmed, 2=hospitalised, 3=death x days)
        self.reported_daily_case_arr = sparse.dok_matrix((self.total_days, self.pop_size), dtype=np.int8)

        # symptomatic testing results (shape = total_days X TP, FP, TN, FN )
        self.total_symp_testing_results = np.zeros((self.total_days, 4), dtype=np.int32)
        # self-test results
        self.total_selftest_results = np.zeros((self.total_days, 4), dtype=np.int32)
        # community testing results (shape = total_days x Social_Entity x TP, FP, TN, FN)
        self.total_community_testing_results = np.zeros((self.total_days, 9, 4), dtype=np.int32)
        # exit testing results (shape = total_days x (iso, quar) x TP, FP, TN, FN)
        self.total_exit_testing_results = np.zeros((self.total_days, 2, 4), dtype=np.int32)
        # daily test quarantine results (total_days x TP, FP, TN, FN)
        self.total_daily_quarantine_testing_results = np.zeros((self.total_days, 4), dtype=np.int32)

        # arrays saving agent info
        # infector (row + 2) to infectee (col) array which saves the type of virus infected (1 = WT, 2 = MT)
        self.vtype_infector_to_infectee = sparse.dok_matrix((self.pop_size, self.pop_size), dtype=np.int8)
        self.setting_infectee = np.zeros((2, self.pop_size), dtype=np.int32) - 100
        self.exposed_day_infectee = np.zeros((2, self.pop_size), dtype=np.int32) - 100

        # array saving total number of agents in each compartment
        cdef np.ndarray epi_seird_arr = np.zeros((self.total_days, 8), dtype=np.int32)
        # array saving total number of agents in each type of isoquar
        cdef np.ndarray epi_isoquar_arr = np.zeros((self.total_days, 4), dtype=np.int32)
        # array saving EOD tests left
        cdef object eod_test_arr = sparse.dok_matrix((self.total_days, 10)) # total days x Social entity + symptomatic

        # assign healthcare facilities to households
        print ("Initialising healthcare facilities...")
        cdef int32 hcf_n # number of healthcare facilities
        hcf_n = self.assign_hcf()

        # array saving total number of tests and antivirals in each facility at the end of each day
        cdef np.ndarray eod_hcf_tav_stocks = np.zeros((self.total_days, hcf_n, 2), dtype=np.int32)

        # initialise infections
        print ("Initialising infections...")
        self.initialise_infections()

        # initialise agents who will perform cross border travels
        if self.cross_border_travel_bool > 0:
            print ("Initialising cross-border travellers...")
            self.initialise_cross_border_trader_agents()

        # compute number of available tests to replenish per period
        cdef np.ndarray replenish_day # list to save day of restock
        cdef np.ndarray rdt_replenish_arr = np.zeros(self.total_days, dtype=np.int32) # array to save day and amount to restock

        # dist_comm_rdt_allocation total must equal 1 or 0
        if sum(self.dist_comm_rdt_allocation.values()) > 0 and sum(self.dist_comm_rdt_allocation.values()) < 1:
            raise Exception('Allocation proportion of community tests must be equals to 0 or 1.')

        # initialise array saving number of test stocks at different healthcare clinics
        self.curr_hcf_test_stocks = np.zeros(hcf_n, dtype=np.int32)
        if self.number_of_rdt_per_100k_per_day >= 0:
            print ('%i tests/100k/day available...'%(self.number_of_rdt_per_100k_per_day))
            self.curr_number_of_hcf_rdt = 0
            self.curr_number_of_comm_rdt = {setting:0 for setting in [overseas, household, school_class, workplace_formal, community, church, bars] if setting in self.dist_comm_rdt_allocation}

            if self.rdt_restock_period == 0: # weekly restock
                replenish_day = np.array([day for day, day_obj in enumerate(date_range) if day_obj.weekday() == 0])

            elif self.rdt_restock_period == 1: # monthly restock
                replenish_day = np.array([day for day, day_obj in enumerate(date_range) if day_obj.day == 1])
                if self.prop_rdt_hcf_allocation < 0:
                    raise Exception("Symptomatic-testing first community testing would not make sense if RDT restock period is >1 week.")

            elif self.rdt_restock_period == 2: # quarterly restock
                replenish_day = np.array([day for day, day_obj in enumerate(date_range) if (day_obj.month in [1, 4, 7, 10]) and (day_obj.day == 1)])
                if self.prop_rdt_hcf_allocation < 0:
                    raise Exception("Symptomatic-testing first community testing would not make sense if RDT restock period is >1 week.")

            rdt_replenish_arr[0] = self.number_of_rdt_per_100k_per_day * replenish_day[0] * self.pop_size/1e5
            rdt_replenish_arr[replenish_day[:-1]] = (replenish_day[1:] - replenish_day[:-1]) * self.number_of_rdt_per_100k_per_day * self.pop_size/1e5
            rdt_replenish_arr[replenish_day[-1]] = (self.total_days - replenish_day[-1]) * self.number_of_rdt_per_100k_per_day * self.pop_size/1e5

        else:
            print ('Unlimited number of tests available...')
            self.curr_number_of_hcf_rdt = -1
            self.curr_number_of_comm_rdt = {overseas:-1, household:-1, school_class:-1, workplace_formal:-1, community:-1, church:-1, bars:-1}
            self.curr_hcf_test_stocks[:] = -1

        # initialise array saving number of antiviral stocks at different healthcare clinics
        self.curr_hcf_av_stocks = np.zeros(hcf_n, dtype=np.int32)

        # compute number of available antiviral therapy to replenish per period
        cdef np.ndarray av_replenish_arr = np.zeros(self.total_days, dtype=np.int32) # array to save day and amount to restock

        if self.number_of_av_per_100k_per_day >= 0:
            print ('%i courses of antiviral therapy/100k/day available...'%(self.number_of_av_per_100k_per_day))

            if self.av_restock_period == 0: # weekly restock
                replenish_day = np.array([day for day, day_obj in enumerate(date_range) if day_obj.weekday() == 0])

            elif self.av_restock_period == 1: # monthly restock
                replenish_day = np.array([day for day, day_obj in enumerate(date_range) if day_obj.day == 1])

            elif self.av_restock_period == 2: # quarterly restock
                replenish_day = np.array([day for day, day_obj in enumerate(date_range) if (day_obj.month in [1, 4, 7, 10]) and (day_obj.day == 1)])

            av_replenish_arr[0] = self.number_of_av_per_100k_per_day * replenish_day[0] * self.pop_size/1e5
            av_replenish_arr[replenish_day[:-1]] = (replenish_day[1:] - replenish_day[:-1]) * self.number_of_av_per_100k_per_day * self.pop_size/1e5
            av_replenish_arr[replenish_day[-1]] = (self.total_days - replenish_day[-1]) * self.number_of_av_per_100k_per_day * self.pop_size/1e5

        else:
            print ('Unlimited number of antiviral stocks available...')
            self.curr_hcf_av_stocks[:] = -1

        # initialise prev_tested_entities as a memory dictionary
        self.prev_tested_entities = {}

        # run simulation
        print ("\nStarting simulation...")
        cdef np.ndarray symp_agents_who_will_visit_hcf, vtype_of_symp_agents_who_will_visit_hcf, symp_agents_who_will_selftest, vtype_of_symp_agents_who_will_selftest
        cdef np.ndarray weekdays_arr = np.zeros(len(date_range), dtype=np.int32)
        cdef int32 week_nr = 0
        cdef int32 number_of_rdt_fixed_for_hcf, number_of_rdt_stock_left_for_comm

        for day, day_obj in enumerate(date_range):

            # get day of the week
            weekday = day_obj.weekday()
            weekdays_arr[day] = weekday

            # start of the week on Monday
            if weekday == 0:
                week_nr += 1

                if odd_week_bool > 0:
                    odd_week_bool = 0
                else:
                    odd_week_bool = 1

            # day of test stock replenishment
            if rdt_replenish_arr[day] > 0:
                print ('Replenish %i tests stocks (day %i)...'%(rdt_replenish_arr[day], day))

                # Two ways to do HCF allocation
                # 1) fixed allotment strategy; HCF allocation is fixed a certain proportion
                if self.prop_rdt_hcf_allocation > -1:
                    # initialise number of available tests at the start of the simulation
                    number_of_rdt_fixed_for_hcf = np.around(self.prop_rdt_hcf_allocation * rdt_replenish_arr[day]).astype(np.int32)
                    self.curr_number_of_hcf_rdt += number_of_rdt_fixed_for_hcf
                    self.stock_up_hcfs(number_of_rdt_fixed_for_hcf, 1)

                    # compute number of tests available for each community setting under fixed allotment strategy
                    number_of_rdt_stock_left_for_comm = rdt_replenish_arr[day] - number_of_rdt_fixed_for_hcf
                    for setting in [overseas, household, school_class, workplace_formal, community, church, bars]:
                        if setting in self.dist_comm_rdt_allocation:
                            self.curr_number_of_comm_rdt[setting] += np.around(self.dist_comm_rdt_allocation[setting] * number_of_rdt_stock_left_for_comm).astype(np.int32)

                else:
                    # 2) All tests go to symptomatic testing first
                    self.curr_number_of_hcf_rdt += rdt_replenish_arr[day]
                    self.stock_up_hcfs(rdt_replenish_arr[day], 1)

                print ("Community test balance:", self.curr_number_of_comm_rdt)
                print ("HCF test balance:", self.curr_hcf_test_stocks, self.curr_hcf_test_stocks.sum(), self.curr_number_of_hcf_rdt)

            # day of antiviral stock replenishment
            if av_replenish_arr[day] > 0:
                print ('Replenish %i antiviral stocks (day %i)...'%(av_replenish_arr[day], day))
                self.stock_up_hcfs(av_replenish_arr[day], 0)
                print ("antiviral balance:", self.curr_hcf_av_stocks, self.curr_hcf_av_stocks.sum())

            # introduce mutant varaint in population
            if self.init_mt_prop > 0 and day == self.mt_intro_delay:
                self.introduce_mt_infections(day)

            # update infection and disease progression
            # get array of symptomatic agents to be tested today
            symp_agents_who_will_visit_hcf, vtype_of_symp_agents_who_will_visit_hcf, symp_agents_who_will_selftest, vtype_of_symp_agents_who_will_selftest = self.review_disease_progression(day)
            # perform symptomatic testing at HCF
            self.symptomatic_testing(symp_agents_who_will_visit_hcf, vtype_of_symp_agents_who_will_visit_hcf, symp_agents_who_will_selftest, vtype_of_symp_agents_who_will_selftest, day)
            # perform self-test using clinic test stocks (if any)
            self.self_testing(day)
            # quarantine agents (if any)
            self.quarantine_agents(day)
            if self.daily_test_quarantine_bool > 0:
                # agents perform daily test quarantine
                self.daily_test_quarantine(day)
            # update isoquar status
            self.review_isoquar(day)

            if self.cross_border_travel_bool > 0:
                # simulate cross border travels
                self.cross_border_travel(day)
                # review and test cross border travelers
                self.review_cross_border_travel(day)

            # testing in households
            if (self.testing_strategies[household]["boolean"]>0.) and (weekday in self.testing_strategies[household]["test_days"]): # testing is only done on after every test_days
                self.community_testing(day, weekday, week_nr, household)

            # testing in community
            if (self.testing_strategies[community]["boolean"]>0.) and (odd_week_bool in self.testing_strategies[community]["odd_week_bool"]) and (weekday in self.testing_strategies[community]["test_days"]): # testing is only done on after every test_days
                self.community_testing(day, weekday, week_nr, community)

            # contact and transmissions
            # transmissions arising households
            if self.transmission_bool[household] > 0:
                self.contact_and_transmission(day, household)

            if weekday < 5: # weekdays
                # transmissions arising from schools
                if self.transmission_bool[school] > 0:
                    # testing in school
                    if (self.testing_strategies[school_class]["boolean"]>0.) and (odd_week_bool in self.testing_strategies[school_class]["odd_week_bool"]) and (weekday in self.testing_strategies[school_class]["test_days"]): # testing is only done on after every test_days
                        self.community_testing(day, weekday, week_nr, school_class)
                    # transmission within class
                    self.contact_and_transmission(day, school_class)
                    # among teachers
                    self.contact_and_transmission(day, school)

                # transmissions arising from formal workplaces
                if self.transmission_bool[workplace_formal] > 0:
                    # testing in formal workplaces
                    if (self.testing_strategies[workplace_formal]["boolean"]>0.) and (weekday in self.testing_strategies[workplace_formal]["test_days"]): # testing is only done on after every test_days
                        self.community_testing(day, weekday, week_nr, workplace_formal)
                    # transmission
                    self.contact_and_transmission(day, workplace_formal)

            # weekends specific
            # transmission arising from church on sunday (mass gathering)
            if self.transmission_bool[church] > 0:
                if (self.testing_strategies[church]["boolean"]>0.) and (weekday in self.testing_strategies[church]['test_days']):
                    self.community_testing(day, weekday, week_nr, church)
                if weekday == 6:
                    self.contact_and_transmission(day, church)

            # transmissions arising from informal workplaces
            if self.transmission_bool[workplace_informal] > 0:
                self.contact_and_transmission(day, workplace_informal)

            """
            # bars
            if self.transmission_bool[bars] > 0:
                # create today's bar contact array
                day_bars_contact_layer_arr = self.create_daily_bar_contacts()
                # testing before visit
                if (self.testing_strategies[bars]["boolean"]>0.):
                    self.community_testing(day, weekday, week_nr, bars, contact_layer_arr=day_bars_contact_layer_arr)
                # transmission
                self.contact_and_transmission(day, bars, contact_layer_arr=day_bars_contact_layer_arr)
            """

            # transmissions arising random community contacts
            if self.transmission_bool[community] > 0:
                self.contact_and_transmission(day, community)

            # special variable with a fixed contribution of WT infections into certain setting of interest (schools and mass gatherings only)
            if self.fixed_community_prev > 0.:
                for setting in [school, church]:

                    if self.transmission_bool[setting] < 1:
                        continue

                    if setting == school:
                        setting = school_class # change setting to school_class for school

                    print ('Fixed %.1f prevalence introduction to agents associated with setting %i...'%(self.fixed_community_prev*100, setting))

                    # get all entities of setting
                    entity_arr =  np.array(self.entity_type_to_ids[setting])
                    susceptible_arr = self.social_contact_layer_arr[entity_arr,:].multiply(self.curr_seird_arr[susceptible,:].tocsr()).tocoo().col
                    susceptible_arr = np.unique(susceptible_arr) # some teachers may be linked to multiple classes

                    # get probability of infection from community and mask agents not infected
                    mask = np.random.random(len(susceptible_arr)) < self.fixed_community_prev
                    infected_arr = susceptible_arr[mask]
                    if len(infected_arr) > 0:
                        self.assign_infection_vars_to_exposed_array(infected_arr, np.repeat(-1, len(infected_arr)), day, community, root_virus_type=0)

            # save total number of SERID comparments to epi_seird_arr
            epi_seird_arr[day] = self.curr_seird_arr.sum(axis=1).T[0]
            # save total number of agents in isoquar to epi_isoquar_arr
            epi_isoquar_arr[day] = self.curr_isoquar_arr.sum(axis=1).T[0]

            # compute Reff
            self.compute_Reff(day)

            if verbose == 2: # 2nd level verbosity
                if self.agent_to_track < 0:
                    print (day, weekday, 'SEIRD:', epi_seird_arr[day], 'ISOQUAR:', epi_isoquar_arr[day])
                    print ('incidence:', self.setting_incidence_arr[day].sum(axis=1), self.setting_incidence_arr[day].sum(),)
                    print ('isoquar:', epi_isoquar_arr[day])
                    print ('exit test:', self.total_exit_testing_results[day,:,:].sum(axis=0))
                    print ('community test (results):', self.total_community_testing_results[day,:,:].sum(axis=0),)
                    print ('community test (setting):', self.total_community_testing_results[day,:,:].sum(axis=1),)
                    print ('daily quar test:', self.total_daily_quarantine_testing_results[day,:])
                    print ('Reff:', self.Reff_arr[day,:], '\n')
                else:
                    print ('*', day, self.agent_to_track, self.curr_seird_arr[:,self.agent_to_track].toarray().T, 'sev', self.simpop_disease_severity[day,self.agent_to_track], 'ct', self.simpop_ct_arr[day,self.agent_to_track], 'onset:', self.simpop_day_of_symptom_onset[:,self.agent_to_track], 'curr_isoquar', self.curr_days_in_isoquar[day,self.agent_to_track], 'curr_days_in_isoquar', (np.ravel(self.curr_days_in_isoquar[:day,self.agent_to_track].tocsc().sum(axis=0))/(self.curr_days_in_isoquar[day,self.agent_to_track])).astype(np.int32), 'hcf', self.hcf_sample_collection_day_arr[:,self.agent_to_track])
                    print ('travel_state', self.curr_agents_across_the_border[self.agent_to_track], 'travel_days', self.simpop_travel_days[day,self.agent_to_track])
                    print (self.individuals_df[self.agent_to_track], '\n')

            elif verbose == 1:
                print (day, weekday, 'SEIRD:', epi_seird_arr[day], 'ISOQUAR:', epi_isoquar_arr[day], 'Reff:', self.Reff_arr[day,:])
                if self.total_symp_testing_results[day,:].sum() > 0:
                    print ("reported test_pos_rate: %.2f%%"%(100 * self.total_symp_testing_results[day,:2].sum()/self.total_symp_testing_results[day,:].sum()))
                else:
                    print ("test_pos_rate: N.A.")
                print ("Community test balance:", self.curr_number_of_comm_rdt)
                if self.selftest_at_clinic_bool < 1:
                    print ("HCF test balance:", self.curr_hcf_test_stocks, self.curr_hcf_test_stocks.sum(), self.curr_number_of_hcf_rdt, self.total_symp_testing_results[day,:].sum() + self.selftest_given_out[day])
                else:
                    print ("HCF test balance:", self.curr_hcf_test_stocks, self.curr_hcf_test_stocks.sum(), self.curr_number_of_hcf_rdt, self.total_symp_testing_results[day,:].sum())
                print ("close contact self-test results:", self.total_selftest_results[day,:])
                print ("self-test otc n = %i"%(self.selftest_otc_n))
                print ("antiviral balance:", self.curr_hcf_av_stocks, self.curr_hcf_av_stocks.sum(), '\n')

            # save EOD test
            if self.curr_number_of_hcf_rdt > -1:
                eod_test_arr[day,-1] = self.curr_number_of_hcf_rdt
                for setting in [overseas, household, school_class, workplace_formal, community, church, bars]:
                    if setting in self.dist_comm_rdt_allocation:
                        eod_test_arr[day,setting] = self.curr_number_of_comm_rdt[setting]

                # last day of the week
                # symptomatic-first community testing strategy
                if self.prop_rdt_hcf_allocation < 0 and weekday == 6 and self.curr_number_of_hcf_rdt > 0:
                    # all unused symp tests last week go to next week's community screening
                    self.curr_hcf_test_stocks[:] = 0
                    for setting in [overseas, household, school_class, workplace_formal, community, church, bars]:
                        if setting in self.dist_comm_rdt_allocation:
                            self.curr_number_of_comm_rdt[setting] += np.around(self.dist_comm_rdt_allocation[setting] * self.curr_number_of_hcf_rdt ).astype(np.int32)
                    self.curr_number_of_hcf_rdt = self.curr_hcf_test_stocks.sum()

            if self.curr_hcf_test_stocks.sum() > 0:
                eod_hcf_tav_stocks[day,:,0] = self.curr_hcf_test_stocks
            if self.curr_hcf_av_stocks.sum() > 0:
                eod_hcf_tav_stocks[day,:,1] = self.curr_hcf_av_stocks

        return weekdays_arr, self.pmap_age, self.pmap_adults_at_risk, self.pmap_vacc_status, epi_seird_arr, epi_isoquar_arr, self.Reff_arr, self.asymp_infector_arr, self.setting_incidence_arr, self.exposed_day_infectee, self.simpop_day_of_symptom_onset, self.length_of_infectious_period, self.setting_infectee, self.hcf_sample_collection_day_arr, self.untested_non_covid_symp_lack_of_test, self.total_symp_testing_results, self.total_selftest_results, self.total_community_testing_results, self.total_exit_testing_results, self.total_daily_quarantine_testing_results, eod_hcf_tav_stocks, eod_test_arr, self.reported_daily_case_arr, self.untested_covid_symp_lack_of_test_arr, self.vtype_infector_to_infectee, self.simpop_infection_status, self.simpop_disease_severity, self.simpop_ct_arr, self.simpop_postest_setting, self.simpop_isoquar_arr, self.border_crossing_stats, self.simpop_travel_days, self.prev_tested_entities, self.hcf_contact_layer_arr.tocsr(), self.simpop_agents_w_av, self.simpop_agents_av_benefit
