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

from pgmpy.models import BayesianNetwork
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc

from pgmpy.readwrite import BIFReader

import traceback 

u_EQ5D = {
    "1_young_adult": 0.966,
    "2_young": 0.963,
    "3_young_adult": 0.939,
    "4_adult": 0.911,
    "5_old_adult": 0.884,
}

def QALY(crc, scr):
    if crc == 1 and scr == 1:
        return np.random.uniform(0.6, 0.7)
    elif crc == 1 and scr == 0:
        return np.random.uniform(0, 0.05)
    else:
        return np.random.uniform(0.95, 1)
    
def EQ5D(age):
    return u_EQ5D[age]

cost_scr = 150
def utility_gov(age, crc, scr, K):
    return 30000*EQ5D(age)*QALY(crc, scr) - K*scr - scr*cost_scr

def utility_cit(age, crc, scr, K):
    return 30000*EQ5D(age)*QALY(crc, scr) + K*scr - scr*cost_scr*np.random.uniform(0.6, 0.7)


def plot_histograms_count_distrib(patient_chars, model, total_sim, n_random_trials):
    model_inference = VariableElimination(model)

    # This assumes that the patient has access to the model and can calculate p(CRC)
    p_no_crc = model_inference.query(variables=['CRC'], evidence=patient_chars).values[0]
    p_crc = model_inference.query(variables=['CRC'], evidence=patient_chars).values[1]

    count = 0
    age = patient_chars["Age"]

    for K in [1, 10, 30, 100]:
        plt_arr = []
        for _ in range(n_random_trials):
            for _ in range(total_sim):
                U_C0_I0 = utility_cit(age, crc = 0, scr = 0, K = K)
                U_C0_I1 = utility_cit(age, crc = 0, scr = 1, K = K)
                U_C1_I0 = utility_cit(age, crc = 1, scr = 0, K = K)
                U_C1_I1 = utility_cit(age, crc = 1, scr = 1, K = K)

                if U_C1_I1 * p_crc + U_C0_I1 * p_no_crc > U_C1_I0 * p_crc + U_C0_I0 * p_no_crc:
                    count += 1
                
            p_scr = count / total_sim
            plt_arr.append(p_scr)
            count = 0
        
        # for each K, plot the distribution of p_scr with a different color
        plt.hist(plt_arr, bins = 16 , alpha=0.5, label=str(K))
        plt.legend(loc='upper right')
        plt.title("p_A(scr | Age = " + age + ")")
        print(f"K = {K}, mean_p_scr = {np.mean(plt_arr)}, median_p_scr = {np.median(plt_arr)}")

    plt.savefig(f"K_{K}_age_{age}.png")
    plt.close()
        


def run_iteration(i, model, patient_chars, upper_K, n_K_points, total_sim, age):
    model_inference = VariableElimination(model)

    p_no_crc = model_inference.query(variables=['CRC'], evidence=patient_chars).values[0]
    p_crc = model_inference.query(variables=['CRC'], evidence=patient_chars).values[1]

    p_scr_K_arr = []
    v_x_K_arr = []

    for K in np.linspace(1, upper_K, n_K_points):
        count = sum(
            utility_cit(age, crc=1, scr=1, K=K) * p_crc +
            utility_cit(age, crc=0, scr=1, K=K) * p_no_crc >
            utility_cit(age, crc=1, scr=0, K=K) * p_crc +
            utility_cit(age, crc=0, scr=0, K=K) * p_no_crc
            for _ in range(total_sim)
        )

        p_scr_K = count / total_sim
        # p_scr_K_arr.append(p_scr_K)

        v_x_K = p_scr_K * (
            p_crc * utility_gov(age, crc=1, scr=1, K=K) +
            p_no_crc * utility_gov(age, crc=0, scr=1, K=K)
        ) + (1 - p_scr_K) * (
            p_crc * utility_gov(age, crc=1, scr=0, K=K) +
            p_no_crc * utility_gov(age, crc=0, scr=0, K=K)
        )

        v_x_K_arr.append(float(v_x_K))

    return v_x_K_arr, p_scr_K


# ---------------------- Main Execution ----------------------

if __name__ == "__main__":  # ✅ Add this line
    import os

    reader = XMLBIFReader('model_bn.xml')
    model = reader.get_model()  


    patient_chars = {
    "Age": "4_adult", 
    "Sex": "M",
    "SD": "normal",
    "PA": "2_0",
    "Smoking": "1_not_smoker",
    "BMI": "2_normal",
    "Alcohol": "low",
    "Diabetes": "False",
    "Hypertension": "False",
    }
    

    total_sim = 500
    age = patient_chars["Age"]

    n_K_points = 101
    upper_K = 30
    n_random_trials = 500

    v_scr_K_iter = []

    plot_histograms_count_distrib(patient_chars, model, total_sim, n_random_trials)


    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_iteration, i, model, patient_chars, upper_K, n_K_points, total_sim, age) for i in range(n_random_trials)]
        
        for future in tqdm(as_completed(futures), total=n_random_trials, desc="Processing iterations"):
            v_scr_K_iter.append(future.result())

    df_v_scr_K_iter = pd.DataFrame(v_scr_K_iter)

    # ---------------------- Plot the Results ----------------------
    plt.plot(np.linspace(0, upper_K, n_K_points), df_v_scr_K_iter.median(axis=0))
    plt.xlabel("K")
    plt.ylabel("Mean V_scr_K")
    plt.title(f"Average Screening Value Over {n_random_trials} Random Trials")
    plt.savefig("best_K_ARA.png")
    plt.close()


    # plot mean and std curves for each K
    plt.plot(np.linspace(1, upper_K, n_K_points), df_v_scr_K_iter.mean(axis = 0))
    plt.fill_between(np.linspace(1, upper_K, n_K_points), df_v_scr_K_iter.mean(axis = 0) - df_v_scr_K_iter.std(axis = 0), df_v_scr_K_iter.mean(axis = 0) + df_v_scr_K_iter.std(axis = 0), alpha=0.5)
    plt.xlabel("K")
    plt.ylabel("Mean V_scr_K")
    plt.title(f"Average Screening Value Over {n_random_trials} Random Trials")
    plt.savefig("best_K_ARA_with_std.png")
    
    df_v_scr_K_iter.to_csv("df_v_scr_K_iter.csv")
    print("Done")
    # ----------------