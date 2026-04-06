from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BDsScore
from pgmpy.factors.discrete import State
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
import random

import pysmile
import pysmile_license
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import pdb

from utilities_test import *
from patients import patient
from dist_prob_cit import plot_histograms_count_distrib


def run_iteration(i, p_crc, scr, upper_K, n_K_points, total_sim, age):

    p_no_crc = 1 - p_crc
    scr_decision_patient = ["No_screening", scr]

    v_x_K_arr = np.zeros((n_K_points,))

    for ind_k, k in enumerate(np.linspace(0, upper_K, n_K_points)):
        count_arr = np.zeros(len(scr_decision_patient))
        for _ in range(total_sim):

            arr = np.array( [
                    sensitivity(scr) * prob_crc_cit(age) * utility_cit(age, crc=1, r_scr=1, scr = scr, K=k) +
                    (1 - specificity(scr)) * (1-prob_crc_cit(age)) * utility_cit(age, crc=0, r_scr=1, scr = scr, K=k)
                +
                    (1 - sensitivity(scr)) * prob_crc_cit(age) * utility_cit(age, crc=1, r_scr=0, scr = scr, K=k) +
                    specificity(scr) * (1-prob_crc_cit(age)) * utility_cit(age, crc=0, r_scr=0, scr = scr, K=k)
                for scr in scr_decision_patient] )

            argmax = np.argmax(arr)
            count_arr[argmax] += 1
        

        # Approximate the probability of each decision
        p_scr_K = count_arr / total_sim


        # Calculate the expected utility of the government for each incentive amount K
        v_x_K = [ p_scr_K[i] * (
            sensitivity(scr) * p_crc * utility_gov(age, crc=1, r_scr=1, scr = scr, K=k) +
            (1 - specificity(scr)) * p_no_crc * utility_gov(age, crc=0, r_scr=1, scr = scr, K=k)
        +  
            (1 - sensitivity(scr)) * p_crc * utility_gov(age, crc=1, r_scr=0, scr = scr, K=k) +
            specificity(scr) * p_no_crc * utility_gov(age, crc=0, r_scr=0, scr = scr, K=k)
        ) for i, scr in enumerate(scr_decision_patient)] 


        v_x_K_arr[ind_k] = sum(v_x_K)

    opt_val_loc = np.argmax(v_x_K_arr)
    opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]
    
    return v_x_K_arr








if __name__ == "__main__": 
    import os

    # read patient num from terminal as python main_test.py 1 if given, else 1
    patient_num = int(os.sys.argv[1]) if len(os.sys.argv) > 1 else 1


    # ---------------------- Load the Model (Model 2 containing Model 1) ----------------------
    net2 = pysmile.Network()
    net2.read_file(f"../models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()


    # ---------------------- Set the Patient Characteristics ----------------------
    df_test_w_util_lim = pd.read_csv("../models/df_test.csv", index_col=0)

    # Take a sample for illustrative purposes
    df_test_w_util_lim_sampled = df_test_w_util_lim
    # df_test_w_util_lim_sampled = df_test_w_util_lim.sample(10000)

    '''count = 1
    while df_test_w_util_lim_sampled["CRC"].sum() < 5:
        count += 1
        df_test_w_util_lim_sampled = df_test_w_util_lim.sample(10000)
    print(f"Number of iterations to find at least a CRC positive patient: {count}")'''

    try:
        df_test_w_util_lim_sampled.sort_values("max_value_w_lim", ascending=False, inplace=True)
    except:
        df_test_w_util_lim_sampled.sort_values("max_value", ascending=False, inplace=True)
    
    df_test_w_util_lim_sampled["opt_K"] = np.nan


    incentive_limit = 4e4

    incentive_accumulated = 0
    num_detected_crc = 0

    for ind, row in df_test_w_util_lim_sampled.iterrows():
        net2.clear_all_evidence()
        patient_chars = row[:13].to_dict()
        for key, value in patient_chars.items():
            net2.set_evidence(key, value)
        
        net2.update_beliefs()

        # Calculate the probability of CRC for the patient. (Model 1)
        p_crc = net2.get_node_value("CRC")[1]
        p_no_crc = net2.get_node_value("CRC")[0]

        vars1 = net2.get_outcome_ids("Screening")
        arr = np.array(net2.get_node_value("Screening"))
        df_scr = pd.DataFrame(arr.reshape(1,-1), index=["Screening"], columns=vars1)

        # Take the asigned screening decision for the patient. (Model 2)
        scr = df_scr.idxmax(axis=1).values[0]
        scr_decision_patient = ["No_screening", scr]

        # ----
        # plot_histograms_count_distrib(net2, patient_chars)

        # ---------------------- Run the Simulation ----------------------
        total_sim = 50
        age = patient_chars["Age"]

        n_K_points = 80
        upper_K = 160
        n_random_trials = 50

        v_scr_K_iter = []

        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_iteration, i, p_crc, scr, upper_K, n_K_points, total_sim, age) for i in range(n_random_trials)]
            
            for future in tqdm(as_completed(futures), total=n_random_trials, desc="Processing iterations"):
                v_scr_K_iter.append(future.result())


        opt_val_loc = np.argmax(np.mean(np.stack(v_scr_K_iter), axis = 0))
        opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]*EQ5D(age)

        df_test_w_util_lim_sampled.loc[ind, "opt_K"] = opt_K

        incentive_accumulated += opt_K

        print(f"Patient {ind} has an optimal incentive of {opt_K} and the accumulated incentive is {incentive_accumulated}")
        
        if df_test_w_util_lim_sampled.loc[ind,"CRC"] == True:
            num_detected_crc += 1
            print(f"We have incentivised CRC positive patient with id {ind}. Total detected number: {num_detected_crc}")


        if incentive_accumulated > incentive_limit:
            print(f"We have reached the incentive limit of {incentive_limit}")
            break



        df_test_w_util_lim_sampled.to_csv("../models/df_test_new_w_lim_and_incentives.csv")