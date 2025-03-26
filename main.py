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

from utilities import *
from patients import patient
from dist_prob_cit import plot_histograms_count_distrib


def run_iteration(i, p_crc, scr, upper_K, n_K_points, total_sim, age):

    p_no_crc = 1 - p_crc

    scr_decision_patient = list(set(["No_screening", scr]))

    v_x_K_arr = np.zeros((n_K_points,))

    for ind_k, k in enumerate(np.linspace(0, upper_K, n_K_points)):
        count_arr = np.zeros(len(scr_decision_patient))
        if scr != "No_screening":
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
                
                '''# Calculate the expected utility of the citizen for each screening decision
                arr = np.array( [
                        sensitivity(scr) * p_crc * utility_cit(age, crc=1, r_scr=1, scr = scr, K=k) +
                        (1 - specificity(scr)) * p_no_crc * utility_cit(age, crc=0, r_scr=1, scr = scr, K=k)
                    +
                        (1 - sensitivity(scr)) * p_crc * utility_cit(age, crc=1, r_scr=0, scr = scr, K=k) +
                        specificity(scr) * p_no_crc * utility_cit(age, crc=0, r_scr=0, scr = scr, K=k)
                    for scr in scr_decision_patient] )

                # Save the decision with highest expected utility.
                argmax = np.argmax(arr)
                count_arr[argmax] += 1'''
            

            # Approximate the probability of each decision
            p_scr_K = count_arr / total_sim
        
        else:
            p_scr_K = [1]


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
    patient_chars = patient(patient_num=patient_num)

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
    scr_decision_patient = list(set(["No_screening", scr]))

    # ----
    plot_histograms_count_distrib(net2, patient_chars)

    # ---------------------- Run the Simulation ----------------------
    total_sim = 100
    age = patient_chars["Age"]

    n_K_points = 250
    upper_K = 300
    n_random_trials = 100

    v_scr_K_iter = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_iteration, i, p_crc, scr, upper_K, n_K_points, total_sim, age) for i in range(n_random_trials)]
        
        for future in tqdm(as_completed(futures), total=n_random_trials, desc="Processing iterations"):
            v_scr_K_iter.append(future.result())


    # ---------------------- Plot the Results ----------------------
    # plot each scr  in a differnte color
    plt.plot(np.linspace(0, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0), label = scr)
    # Add best K in plot as a line and in legend
    opt_val_loc = np.argmax(np.mean(np.stack(v_scr_K_iter), axis = 0))
    if scr == "No_screening":
        opt_K = 0
    else:
        opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]
    plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best Incentive = {opt_K:.2f}")
    plt.legend(loc='upper right')
    plt.xlabel("Incentive")
    plt.ylabel(r"$\psi(I| x)$")
    plt.title(f"{age} patient with p(CRC|X) = {p_crc:.2f}")
    plt.savefig(f"outputs/patient_{patient_num}_best_K_ARA.png")
    plt.close()


    # plot mean and std curves for each K
    plt.plot(np.linspace(1, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0), label = scr)
    plt.fill_between(np.linspace(1, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0) - np.std(np.stack(v_scr_K_iter), axis = 0), np.mean(np.stack(v_scr_K_iter), axis = 0) + np.std(np.stack(v_scr_K_iter), axis = 0), alpha=0.5)
    plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best Incentive = {opt_K:.2f}")
    plt.legend(loc='upper right')
    plt.xlabel("Incentive")
    plt.ylabel(r"$\psi(I| x)$")
    plt.title(f"{age} patient with p(CRC|X) = {p_crc:.5f}")
    plt.savefig(f"outputs/patient_{patient_num}_best_K_ARA_with_std.png")
    plt.close()
    
    # pd.DataFrame(np.mean(np.stack(v_scr_K_iter), axis = 0)).to_csv("df_v_scr_K_iter.csv")
    print("Done")
    # ----------------