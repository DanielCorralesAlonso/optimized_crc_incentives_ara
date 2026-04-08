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

from costs_and_utilities import *
from patients import patient
from dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

plt.rcParams.update({'font.size': 15})  # You can adjust the font size as needed

def generate_obp_full_grid(z1_start = None, z1_stop = None, z1_step = None, 
                           z2_start = None, z2_stop = None, z2_step = None, 
                           z3_start = None, z3_stop = None, z3_step = None,
                           upper_K = None, n_K_points = None):
    """
    Generates a full grid combining possible Z = (z1, z2, z3) schemes 
    with the incentive values (K). Enforces the constraint: z1 > z2.
    """
    
    # Generate ranges for Z. If an axis is entirely None, keep a single None value.
    # This supports runs where OBP thresholds/bonus are intentionally disabled.
    def _build_z_values(start, stop, step, axis_name):
        if start is None and stop is None and step is None:
            return np.array([None], dtype=object)
        if start is None or stop is None or step is None:
            raise ValueError(
                f"Incomplete range for {axis_name}: provide start/stop/step or all as None."
            )
        return np.arange(start, stop + step, step)

    z1_values = _build_z_values(z1_start, z1_stop, z1_step, "z1")
    z2_values = _build_z_values(z2_start, z2_stop, z2_step, "z2")
    z3_values = _build_z_values(z3_start, z3_stop, z3_step, "z3")

    # Generate ranges for K
    if upper_K is None or n_K_points is None:
        raise ValueError("upper_K and n_K_points must be provided.")
    k_values = np.linspace(0, upper_K, n_K_points)

    valid_combinations = []
    
    # Iterate through all possible combinations
    for z1 in z1_values:
        for z2 in z2_values:
            # Enforce z1 > z2 only when both thresholds are numeric.
            if z1 is None or z2 is None or z1 > z2:
                for z3 in z3_values:
                    for k in k_values:  # <-- New loop for K
                        valid_combinations.append({
                            "z1_Threshold_100_BP": None if z1 is None else round(z1, 2),
                            "z2_Threshold_50_BP": None if z2 is None else round(z2, 2),
                            "z3_Bonus_Euros": None if z3 is None else round(z3, 2),
                            "K_Incentive": round(k, 2)  # <-- Added K to the dictionary
                        })
                    
    # Convert the list of dictionaries into a Pandas DataFrame
    df_grid = pd.DataFrame(valid_combinations)
    
    return df_grid


def run_iteration(i, J_SP, J_cit, full_grid, model, assigned_screening_individuals, n_different_patients):

    k = full_grid.loc[i, "K_Incentive"]
    

    u_sampled_sp = np.zeros((J_SP,))
    for ind_j in range(J_SP):

        v_K = np.zeros((1,))
        total_screened = 0
        total_detected = 0
        total_cost = 0
    
        for i, patient_chars in enumerate(assigned_screening_individuals.iloc[:,:7].to_dict(orient="records")):
            
            age = patient_chars["Age"]

            infer = VariableElimination(model)
            result = infer.query(variables=list(model.get_parents("CRC")), joint=True)

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

            p_evidence = result.get_value(**evidence)


            # ---- p_{PM}(c | x) ---- Calculate the probabiltiy of having CRC 
            p_crc = float(infer.query(variables=["CRC"], evidence=evidence).values[1])
            p_no_crc = float(infer.query(variables=["CRC"], evidence=evidence).values[0])
            # -------------------------

            # ---- Check which is the screening decision given the decision model (Model 2) for the patient profile x
            try:
                scr = assigned_screening_individuals.loc[i, "best_option"]
            except:
                scr = assigned_screening_individuals.loc[i, "best_option_w_lim"]
            scr_decision_patient = np.unique(["No_screening", scr])
            # -------------------------
            
            # ----- p_{SP}(s | I, x) ---- Calculate the probability for the citizen to accept screening given incentive K and covariates x
            # ----- This is done via simulation based on adversarial risk analysis.
            
            count_arr = np.zeros(len(scr_decision_patient))

            # Simulate utility function for the citizen
            s_opt = np.zeros((J_cit,))
            for j_cit in range(J_cit):
                    '''c_sim = np.random.binomial(n=1, p=prob_crc_cit(age))
                    if c_sim == 1 and scr != "No_screening":
                        r_sim = np.random.binomial(n=1, p=sensitivity(scr))
                    elif c_sim == 0 and scr != "No_screening":
                        r_sim = np.random.binomial(n=1, p=1-specificity(scr))
                    else:
                        r_sim = 0

                    total_cost_cit_array = cost_cit(age = age, crc=c_sim, scr=scr, scr_decision = scr_decision_patient, r_scr=r_sim, K=k)
                    u_sampled_cit_array = random_utilities_cit(total_cost_cit_array)'''

                    arr = np.array( [
                            sensitivity(scr) * prob_crc_cit(age) * cost_cit(age, crc=1, r_scr=1, scr = scr, K=k) +
                            (1 - specificity(scr)) * (1-prob_crc_cit(age)) * cost_cit(age, crc=0, r_scr=1, scr = scr, K=k)
                        +
                            (1 - sensitivity(scr)) * prob_crc_cit(age) * cost_cit(age, crc=1, r_scr=0, scr = scr, K=k) +
                            specificity(scr) * (1-prob_crc_cit(age)) * cost_cit(age, crc=0, r_scr=0, scr = scr, K=k)
                        for scr in scr_decision_patient] )

                    s_opt[j_cit] = np.argmax(arr)

            # Approximate the probability of each decision
            p_scr_K = s_opt.sum() / J_cit
            
            # ------------------------

            
            # ----- Simulate whether the patient is ill, goes to screening and whether cancer is detected.
            c_sim = np.random.binomial(n=1, p=p_crc)
            s_sim = scr_decision_patient[ np.random.binomial(n=1, p=p_scr_K) ]
            if c_sim == 1 and s_sim != "No_screening":
                r_sim = np.random.choice([0,1], p=[1-sensitivity(s_sim), sensitivity(s_sim)])
            elif c_sim == 0 and s_sim != "No_screening":
                r_sim = np.random.choice([0,1], p=[specificity(s_sim), 1-specificity(s_sim)])
            else:
                r_sim = 0
            # ------------------------

            # ----- Calculate cost C(x, c, s, r, k)
            total_screened += (s_sim != "No_screening")
            total_detected += (c_sim == 1 and r_sim == 1)
            total_cost += cost_PM(age, crc=c_sim, scr=s_sim, r_scr=r_sim, K=k)
            # ------------------------

            v_K += p_evidence * np.array(total_cost) #  * n_different_patients 


        u_sampled_sp[ind_j] = random_utilities_SP(v_K.item())

        # expected_util[ind_k, ind_z] = np.mean(u_sampled_sp)  ### This is the unnormalized distribution over the Z grid for each K
       
    return np.mean(u_sampled_sp)  



if __name__ == "__main__": 

    for limit in [False]: # [True, False]:

        # ---------------------- Run the Simulation ----------------------
        J_cit = 1000
        J_SP = 1000

        # Define grid of incentives K to evaluate
        n_K_points = 2
        upper_K = 200

        full_grid = generate_obp_full_grid(
                        upper_K=upper_K, n_K_points=n_K_points
                    )
        
        n_grid_points = len(full_grid)
        print(f"Total number of OBP scheme combinations (Z) and K incentives to evaluate: {n_grid_points}")
        
        expected_util = np.zeros((n_grid_points,))  ### This will store the expected utility for each point in the Z grid and each K

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


        with ProcessPoolExecutor(max_workers=4) as executor:
            # 1. Create a dictionary mapping each submitted Future to its index 'i'
                future_to_index = {
                    executor.submit(
                        run_iteration, i, J_SP, J_cit, full_grid, 
                        model, assigned_screening_individuals, len(df_test_w_util_lim)
                    ): i 
                    for i in range(n_grid_points)
                }
                
                # 2. Iterate over the keys (the futures) as they complete
                for future in tqdm(as_completed(future_to_index), total=n_grid_points, desc="Processing iterations"):
                    
                    # 3. Retrieve the original index for this specific completed future
                    i = future_to_index[future] 
                    
                    try:
                        # 4. Assign the result to the correct index in your array
                        expected_util[i] = future.result()
                        
                    except Exception as exc:
                        print(f"Iteration {i} generated an exception: {exc}")


        expected_util_reshaped = expected_util.reshape((n_K_points, -1))  
        pd.DataFrame(expected_util_reshaped).to_csv("expected_util_public_scheme.csv")


        # ---------------------- Plot the Results ----------------------
        plt.figure(figsize=(6, 5))
        # plot each scr  in a differnte color
        plt.plot(np.linspace(0, upper_K, n_K_points), expected_util_reshaped)
        # Add best K in plot as a line and in legend
        opt_val_loc = np.argmax(expected_util_reshaped)
        opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]
        plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K:.2f}")
        plt.tight_layout()

        plt.legend(loc='upper right')
        plt.xlabel("K")
        plt.ylabel(r"$\phi(I)$")
        plt.title(f"Marginalized Incentive Over The Entire Population")
        plt.savefig(f"outputs/overall_limit_{limit}_best_K_ARA.png", bbox_inches='tight')
        plt.close()

        ''' pd.DataFrame(v_scr_K_iter).to_csv(f"df_v_scr_K_iter_limit_{limit}.csv")

        plt.figure(figsize=(6, 5))
        # plot mean and std curves for each K
        plt.plot(np.linspace(1, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0))
        plt.fill_between(np.linspace(1, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0) - np.std(np.stack(v_scr_K_iter), axis = 0), np.mean(np.stack(v_scr_K_iter), axis = 0) + np.std(np.stack(v_scr_K_iter), axis = 0), alpha=0.5)
        opt_val_loc = np.argmax(np.mean(np.stack(v_scr_K_iter), axis = 0))
        opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]
        plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K:.2f}")
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.xlabel("K")
        plt.ylabel("Mean V_scr_K")
        plt.title(f"Value Over {n_random_trials} Random Trials")
        plt.savefig(f"outputs/overall_limit_{limit}_best_K_ARA_with_std.png", bbox_inches='tight')
        plt.close()

        # pd.DataFrame(np.mean(np.stack(v_scr_K_iter), axis = 0)).to_csv("df_v_scr_K_iter.csv")


        # ------ Plot smoothed curve with Gaussian Process Regression ------
        np.linspace(0, upper_K, n_K_points)
        X = np.linspace(1, upper_K, n_K_points).reshape(-1, 1)
        y = np.mean(np.stack(v_scr_K_iter), axis = 0)

        # Normalize X and y
        X_input = (X - X.min()) / (X.max() - X.min())
        y_input = (y - y.min()) / (y.max() - y.min())

        kernel = RBF(length_scale=2.0) + WhiteKernel(noise_level=1)
        gp = GaussianProcessRegressor(kernel=kernel).fit(X_input, y_input)
        X_pred = np.linspace(0, upper_K, 200).reshape(-1, 1)
        X_pred_input = (X_pred - X.min()) / (X.max() - X.min())
        y_pred, sigma = gp.predict(X_pred_input, return_std=True)
        y_pred = y_pred * (y.max() - y.min()) + y.min()  # Denormalize
        opt_val_smooth = np.argmax(y_pred)

        plt.figure(figsize=(6, 5))
        # plot each scr  in a differnte color
        plt.plot(X_pred, y_pred, label="GP Fit")
        # Add best K in plot as a line and in legend
        opt_val_loc = np.argmax(y_pred)
        opt_K = np.linspace(1, upper_K, 200)[opt_val_loc]
        plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K:.2f}")
        plt.tight_layout()

        plt.legend(loc='upper right')
        plt.xlabel("K")
        plt.ylabel(r"$\phi(I)$")
        plt.title(f"Marginalized Incentive Over The Entire Population")
        plt.savefig(f"outputs/smoothed_overall_limit_{limit}_best_K_ARA.png", bbox_inches='tight')
        plt.close()


        # ---- Also smoothen mean +/- 2*std ----
        y_mean = np.mean(np.stack(v_scr_K_iter), axis = 0)
        y_std = np.std(np.stack(v_scr_K_iter), axis = 0)
        y_upper = y_mean + 2*y_std
        y_lower = y_mean - 2*y_std

        y_upper_input = (y_upper - y.min()) / (y.max() - y.min())
        y_lower_input = (y_lower - y.min()) / (y.max() - y.min())

        gp_upper = GaussianProcessRegressor(kernel=kernel).fit(X_input, y_upper_input)
        gp_lower = GaussianProcessRegressor(kernel=kernel).fit(X_input, y_lower_input)

        y_upper_pred, sigma_upper = gp_upper.predict(X_pred_input, return_std=True)
        y_lower_pred, sigma_lower = gp_lower.predict(X_pred_input, return_std=True)
        y_upper_pred = y_upper_pred * (y.max() - y.min()) + y.min()  # Denormalize
        y_lower_pred = y_lower_pred * (y.max() - y.min()) + y.min()  # Denormalize
 
        plt.figure(figsize=(6, 5))
        # plot each scr  in a differnte color
        plt.plot(X_pred, y_pred, label="GP Fit")
        plt.fill_between(X_pred.flatten(), y_lower_pred, y_upper_pred, alpha=0.5, label="GP Uncertainty")
        # Add best K in plot as a line and in legend
        opt_val_loc = np.argmax(y_pred)
        opt_K = np.linspace(1, upper_K, 200)[opt_val_loc]
        plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K:.2f}")
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.xlabel("K")
        plt.ylabel(r"$\phi(I)$")
        plt.title(f"Marginalized Incentive Over The Entire Population")
        plt.savefig(f"outputs/smoothed_overall_limit_{limit}_best_K_ARA_with_std.png", bbox_inches='tight')


        print("Done")
        # ----------------'''