from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BDsScore
from pgmpy.factors.discrete import State
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
import random


import pdb

import itertools
import pysmile
import pysmile_license

from pgmpy.models import BayesianNetwork
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc

from pgmpy.readwrite import BIFReader

import traceback 

u_EQ5D = {
    "age_1_young_adult": 0.966,
    "age_2_young": 0.963,
    "age_3_young_adult": 0.939,
    "age_4_adult": 0.911,
    "age_5_old_adult": 0.884,
}

def EQ5D(age):
    return u_EQ5D[age]


scr_costs_dict = {
    "No_screening": 0,
    "FIT": 14.34,
    "Blood_based": 123.13,
    "Stool_DNA": 236.88,
    "CTC": 95.41,
    "CC": 510.24,
    "Colonoscopy": 1000
}

def scr_costs(scr):
    return scr_costs_dict[scr]


sensitivity_dict = {
    "No_screening": 0,
    "gFOBT": 0.45,
    "FIT": 0.75,
    "Blood_based": 0.66,
    "Stool_DNA": 0.923,
    "CTC": 0.8,
    "CC": 0.87,
    "Colonoscopy": 0.97
}

def sensitivity(scr):
    return sensitivity_dict[scr]

specificity_dict = {
    "No_screening": 1,
    "gFOBT": 0.978,
    "FIT": 0.966,
    "Blood_based": 0.91,
    "Stool_DNA": 0.866,
    "CTC": 0.89,
    "CC": 0.92,
    "Colonoscopy": 0.99
}

def specificity(scr):
    return specificity_dict[scr]


comfort_dict = {
    "No_screening": 0,
    "gFOBT": 1,
    "FIT": 1,
    "Blood_based": 1,
    "Stool_DNA": 1,
    "CTC": 2,
    "CC": 2,
    "Colonoscopy": 3
}

def comfort(scr):
    return comfort_dict[scr]


def diff_QALY_cit(crc, r_scr):
    if crc == 1 and r_scr == 1:
        return np.random.uniform(1.3, 1.4 )
    elif crc == 1 and r_scr == 0:
        return np.random.uniform(0.8, 0.95)
    else:
        return np.random.uniform(0.993, 1.005)

def diff_QALY_gov(crc, r_scr):
    if crc == 1 and r_scr == 1:
        return np.random.uniform(10, 15 )
    elif crc == 1 and r_scr == 0:
        return np.random.uniform(0.1, 0.2)
    else:
        return np.random.uniform(0.999, 1)
    

def utility_gov(age, crc, scr, r_scr, K):
    if scr == "No_screening":
        return 30000*EQ5D(age)*diff_QALY_gov(crc, r_scr)
    else:
        return 30000*EQ5D(age)*diff_QALY_gov(crc, r_scr) - K*EQ5D(age) - scr_costs(scr) - 30000*crc*r_scr

def utility_cit(age, crc, scr, r_scr, K):
    if scr == "No_screening":
        return 30000*EQ5D(age)*diff_QALY_cit(crc, r_scr)
    else:
        return 30000*EQ5D(age)*diff_QALY_cit(crc, r_scr) + K*EQ5D(age) - scr_costs(scr)*np.random.uniform(1 - 0.9/comfort(scr), 1 - 0.6/comfort(scr))


import itertools

def run_iteration(i, p_crc, scr, upper_K, n_K_points, total_sim, age):

    p_no_crc = 1 - p_crc
    scr_decision_patient = ["No_screening", scr]

    v_x_K_arr = np.zeros((n_K_points,))

    for ind_k, k in enumerate(np.linspace(1, upper_K, n_K_points)):
        count_arr = np.zeros(len(scr_decision_patient))
        for _ in range(total_sim):

            arr = np.array( [
                    sensitivity(scr) * p_crc * utility_cit(age, crc=1, r_scr=1, scr = scr, K=k) +
                    (1 - specificity(scr)) * p_no_crc * utility_cit(age, crc=0, r_scr=1, scr = scr, K=k)
                +
                    (1 - sensitivity(scr)) * p_crc * utility_cit(age, crc=1, r_scr=0, scr = scr, K=k) +
                    specificity(scr) * p_no_crc * utility_cit(age, crc=0, r_scr=0, scr = scr, K=k)
                for scr in scr_decision_patient] )

            argmax = np.argmax(arr)
            count_arr[argmax] += 1
        
        p_scr_K = count_arr / total_sim


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

    '''reader = XMLBIFReader('model_bn.xml')
    model = reader.get_model()  '''

    # ---------------------- Load the Model ----------------------
    net2 = pysmile.Network()
    net2.read_file(f"DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    patient_chars = {
    "Age": "age_5_old_adult", 
    "Sex": "M",
    "SD": "SD_2_normal",
    "PA": "PA_2",
    "Smoking": "sm_1_not_smoker",
    "BMI": "bmi_2_normal",
    "Alcohol": "low",
    "Diabetes": "False",
    "Hypertension": "False",
    }

    for key, value in patient_chars.items():
        net2.set_evidence(key, value)

    net2.update_beliefs()

    p_crc = net2.get_node_value("CRC")[1]
    p_no_crc = net2.get_node_value("CRC")[0]

    vars1 = net2.get_outcome_ids("Screening")
    arr = np.array(net2.get_node_value("Screening"))
    df_scr = pd.DataFrame(arr.reshape(1,-1), index=["Screening"], columns=vars1)

    # take column with the highest value
    scr = df_scr.idxmax(axis=1).values[0]
    scr_decision_patient = ["No_screening", scr]
    


    # ---------------------- Run the Simulation ----------------------
    total_sim = 100
    age = patient_chars["Age"]

    n_K_points = 500
    upper_K = 500
    n_random_trials = 100

    v_scr_K_iter = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_iteration, i, p_crc, scr, upper_K, n_K_points, total_sim, age) for i in range(n_random_trials)]
        
        for future in tqdm(as_completed(futures), total=n_random_trials, desc="Processing iterations"):
            v_scr_K_iter.append(future.result())

    # summ_v_scr_K_iter = 


    # ---------------------- Plot the Results ----------------------
    # plot each scr  in a differnte color
    plt.plot(np.linspace(0, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0), label = scr)
    # Add best K in plot as a line and in legend
    opt_val_loc = np.argmax(np.mean(np.stack(v_scr_K_iter), axis = 0))
    opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]
    plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K}")
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("Mean V_scr_K")
    plt.title(f"Average Screening Value Over {n_random_trials} Random Trials")
    plt.savefig("best_K_ARA.png")
    plt.close()


    # plot mean and std curves for each K
    plt.plot(np.linspace(1, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0), label = scr)
    plt.fill_between(np.linspace(1, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0) - np.std(np.stack(v_scr_K_iter), axis = 0), np.mean(np.stack(v_scr_K_iter), axis = 0) + np.std(np.stack(v_scr_K_iter), axis = 0), alpha=0.5)
    opt_val_loc = np.argmax(np.mean(np.stack(v_scr_K_iter), axis = 0))
    opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]
    plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K}")
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("Mean V_scr_K")
    plt.title(f"Average Screening Value Over {n_random_trials} Random Trials")
    plt.savefig("best_K_ARA_with_std.png")
    plt.close()
    
    # pd.DataFrame(np.mean(np.stack(v_scr_K_iter), axis = 0)).to_csv("df_v_scr_K_iter.csv")
    print("Done")
    # ----------------