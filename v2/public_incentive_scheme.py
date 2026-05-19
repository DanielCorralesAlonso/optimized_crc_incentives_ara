import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
import random

from pgmpy.readwrite import XMLBIFReader
import itertools

import pysmile
import pysmile_license
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import pdb
import os
import joblib

from costs_and_utilities import cost_SP, sensitivity, specificity, cara_health
from patients import patient
# from v2.dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *
from simulator_cit_p_scr import run_simulation, train_emulator

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



import numpy as np
import pandas as pd


from utils import generate_grid



def run_iteration(i, J_SP, full_grid, model, assigned_screening_individuals, simulated_df = None, emulate = False, ):

    k = full_grid.loc[i, "K_Incentive"]
    if emulate and simulated_df is None:
        p_scr_K_emulator = joblib.load("models/xgb_cit_model.pkl")
        feature_metadata = joblib.load("models/xgb_cit_model_meta.pkl")

        # Keep feature order identical to training to avoid schema drift at predict time.
        categorical_levels = feature_metadata.get("categorical_levels", {})

        for col, levels in categorical_levels.items():
            if col in X.columns:  # Good practice: ensure the column exists after reindexing
                X[col] = pd.Categorical(X[col], categories=levels)

    z = {
        "z1_Threshold_100_BP": full_grid.loc[i, "z1_Threshold_100_BP"],
        "z2_Threshold_50_BP": full_grid.loc[i, "z2_Threshold_50_BP"],
        "z3_Bonus_Euros": full_grid.loc[i, "z3_Bonus_Euros"]
    }

    X = assigned_screening_individuals.iloc[:, :7].copy()

    X["K"] = float(k)
    if emulate and simulated_df is not None:
        X = X[feature_metadata["feature_columns"]]
    else: 
        X = X

    u_sampled_sp = np.zeros((J_SP,))
    for ind_j in range(J_SP):

        p_crc = np.zeros((len(assigned_screening_individuals),))
        p_scr_K = np.zeros((len(assigned_screening_individuals),))
        total_cost_SP = np.zeros((len(assigned_screening_individuals),))

        infer = VariableElimination(model)
        result = infer.query(variables=list(model.get_parents("CRC")), joint=True)

        count = 0
        for i, patient_chars in enumerate(assigned_screening_individuals.iloc[:,:7].to_dict(orient="records")):
            
            age = patient_chars["Age"]

            patient_chars["Age"] = patient_chars["Age"].replace("age_", "")
            patient_chars["Smoking"] = patient_chars["Smoking"].replace("sm_", "")
            

            evidence = patient_chars
            evidence["Hyperchol."] = patient_chars.pop("Hyperchol_")

            # transform bool values in text
            for key, value in evidence.items():
                if value == 1:
                    evidence[key] = "True"
                elif value == 0:
                    evidence[key] = "False"


            # ---- p_{PM}(c | x) ---- Calculate the probabiltiy of having CRC 
            p_crc[i] = float(infer.query(variables=["CRC"], evidence=evidence).values[1])
            # -------------------------

            # ---- Check which is the screening decision given the decision model (Model 2) for the patient profile x
            try:
                scr = assigned_screening_individuals.loc[i, "best_option"]
            except:
                scr = assigned_screening_individuals.loc[i, "best_option_w_lim"]
            # -------------------------

            # Simulate from the available probabilities
            n_total_patients_with_chars = int(assigned_screening_individuals.loc[i, "total_count"])  # Number of patients in the dataset with these characteristics
            
            c_sim = np.array([np.random.binomial(1, p_crc[i]) for _ in range(n_total_patients_with_chars)])
            
             # ----- p_{SP}(s | I, x) ---- Calculate the probability for the citizen to accept screening given incentive K and covariates x
            if emulate:
                p_scr_K[i] = p_scr_K_emulator.predict(X.iloc[[i]]).squeeze()
                s_sim = np.array([np.random.binomial(1, p_scr_K[i]) for _ in range(n_total_patients_with_chars)])
            else:
                s_sim = np.array([np.random.binomial(1, simulated_df.loc[count:count+n_total_patients_with_chars-1, f"p_scr_K_{k:.2f}"].values[i]) for i in range(n_total_patients_with_chars)])

            r_sim = np.array([np.random.binomial(1, sensitivity(scr) * c_sim[j] + (1 - specificity(scr)) * (1 - c_sim[j])) if s_sim[j] == 1 else 0 for j in range(n_total_patients_with_chars)])
            
            #pdb.set_trace()
            # Simulate from costs (right now costs have a random component, not utilities)
            cost_sim = np.array([cost_SP(age=age, crc=c_sim[j], scr=scr, scr_decision=s_sim[j], r_scr=r_sim[j], K=k) for j in range(n_total_patients_with_chars)])
            total_cost_SP[i] = cost_sim.sum()  # Average cost across the simulated patients with these characteristics

            count += n_total_patients_with_chars

        #  pdb.set_trace()
        u_sampled_sp[ind_j] = cara_health(total_cost_SP.sum() / (25000 * count), 0.01)  ### This is the unnormalized utility for the SP for this iteration of K and Z, based on the simulated patient responses and outcomes.

    # pdb.set_trace()
    return u_sampled_sp  ### This is the unnormalized distribution over the Z grid for each K


if __name__ == "__main__":
    limit = False
    J_SP = 5

    # Define grid of incentives K to evaluate
    n_K_points = 8
    upper_K = 200
    N_ara = 10

    # Define possible Z's (parameterized OBP schemes)
    full_grid = generate_grid(
        upper_K=upper_K, n_K_points=n_K_points
    )

    v_scr_K_iter = []

    net2 = pysmile.Network()
    net2.read_file(f"models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    df_test_w_util_lim = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)

    reader = XMLBIFReader("models/model_bn.xml")
    model = reader.get_model()

    best_options = get_all_combinations_id_w_optimal_scr(net2, df_test_w_util_lim, limit = limit)
    try:
        assigned_screening_individuals = best_options[ best_options["best_option_w_lim"] != "No_screening" ].reset_index(drop=True).copy()
    except:
        assigned_screening_individuals = best_options[ best_options["best_option"] != "No_screening" ].reset_index(drop=True).copy()


    simulation_required = True
    if simulation_required:
        print("Simulation required: Running simulator_cit_p_scr to generate probabilities for the SP's decision model under different K values. ")
        
        # Call the refactored function instead of running a subprocess!
        simulated_df = run_simulation(
            limit=limit, 
            N_ara=N_ara,             # Adjust if you want more inner citizens
            n_K_points=n_K_points, 
            upper_K=upper_K
        )
    else: 
        simulated_df = None


    # Built for parallelization.
    expected_util = np.zeros((len(full_grid),))
    credible_interval = np.zeros((len(full_grid), 2))  # To store the 2.5 and 97.5 percentiles for the credible interval
    for i in tqdm(range(len(full_grid))):
        u_sp = run_iteration(i, J_SP, full_grid, model, assigned_screening_individuals, simulated_df=simulated_df, emulate= not simulation_required)
    

        expected_util[i] = u_sp.mean()  ### This is the expected utility for the SP for this K, marginalized over the Z grid (since we are simulating from the distribution over Z given K and then averaging the utility across those simulations).
        credible_interval[i] = np.percentile(u_sp, [2.5, 97.5])

    
    # Parallelized version
    '''expected_util = np.zeros((len(full_grid),))
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_iteration, i, J_SP, full_grid, model, assigned_screening_individuals, len(df_test_w_util_lim)): i for i in range(len(full_grid))}
        for future in tqdm(as_completed(futures), total=len(full_grid)):
            i = futures[future]
            expected_util[i] = future.result()'''


    # pdb.set_trace()
    expected_util_reshaped = expected_util.reshape((-1, n_K_points)) 
    pd.DataFrame(expected_util_reshaped).to_csv("expected_util.csv")

    plt.plot(np.linspace(0, upper_K, n_K_points) , expected_util_reshaped[0])
    plt.fill_between(np.linspace(0, upper_K, n_K_points), credible_interval[:, 0], credible_interval[:, 1], color='b', alpha=0.2, label="95% Credible Interval")
    plt.xlabel(f"Incentive K given Z_grid[0]")
    plt.ylabel("Expected Utility for the SP")
    plt.title("Expected Utility vs Incentive K")

    try:
        plt.savefig("outputs/public_incentive_scheme/expected_utility_vs_K.png")
    except:
        plt.savefig("expected_utility_vs_K.png")
    plt.close()