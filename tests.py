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

from v2.costs_and_utilities import *
from v2.patients import patient
# from v2.dist_prob_cit import plot_histograms_count_distrib
from v2.get_combinations import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



import numpy as np
import pandas as pd

# Define grid parameters
n_K_points = 20
upper_K = 200

def generate_obp_full_grid(z1_start, z1_stop, z1_step, 
                           z2_start, z2_stop, z2_step, 
                           z3_start, z3_stop, z3_step,
                           upper_K, n_K_points):
    """
    Generates a full grid combining possible Z = (z1, z2, z3) schemes 
    with the incentive values (K). Enforces the constraint: z1 > z2.
    """
    
    # Generate ranges for Z
    z1_values = np.arange(z1_start, z1_stop + z1_step, z1_step)
    z2_values = np.arange(z2_start, z2_stop + z2_step, z2_step)
    z3_values = np.arange(z3_start, z3_stop + z3_step, z3_step)
    
    # Generate ranges for K
    k_values = np.linspace(0, upper_K, n_K_points)

    valid_combinations = []
    
    # Iterate through all possible combinations
    for z1 in z1_values:
        for z2 in z2_values:
            # Check the primary constraint: 100% threshold > 50% threshold
            if z1 > z2:
                for z3 in z3_values:
                    for k in k_values:  # <-- New loop for K
                        valid_combinations.append({
                            "z1_Threshold_100_BP": round(z1, 2),
                            "z2_Threshold_50_BP": round(z2, 2),
                            "z3_Bonus_Euros": round(z3, 2),
                            "K_Incentive": round(k, 2)  # <-- Added K to the dictionary
                        })
                    
    # Convert the list of dictionaries into a Pandas DataFrame
    df_grid = pd.DataFrame(valid_combinations)
    
    return df_grid


def run_iteration(i, J_SP, J_cit, upper_K, n_K_points, full_grid, model, assigned_screening_individuals, n_different_patients):

    k = full_grid.loc[i, "K_Incentive"]
    z = {
        "z1_Threshold_100_BP": full_grid.loc[i, "z1_Threshold_100_BP"],
        "z2_Threshold_50_BP": full_grid.loc[i, "z2_Threshold_50_BP"],
        "z3_Bonus_Euros": full_grid.loc[i, "z3_Bonus_Euros"]
    } 

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
                    
                    c_sim = np.random.binomial(n=1, p=prob_crc_cit(age))
                    if c_sim == 1 and scr != "No_screening":
                        r_sim = np.random.binomial(n=1, p=sensitivity(scr))
                    elif c_sim == 0 and scr != "No_screening":
                        r_sim = np.random.binomial(n=1, p=1-specificity(scr))
                    else:
                        r_sim = 0

                    total_cost_cit_array = cost_cit(age = age, crc=c_sim, scr=scr, scr_decision = scr_decision_patient, r_scr=r_sim, K=k)
                    u_sampled_cit_array = random_utilities_cit(total_cost_cit_array)
                    s_opt[j_cit] = np.argmax(u_sampled_cit_array)

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

            v_K += p_evidence * n_different_patients * np.array(total_cost)


        if total_screened / n_different_patients > list(z.values())[0] :
            v_K +=  - total_cost
        elif total_screened / n_different_patients > list(z.values())[1]:
            v_k += - 0.5 * total_cost

        v_K += list(z.values())[2] * total_detected

        u_sampled_sp[ind_j] = random_utilities_SP(v_K.item())

        # expected_util[ind_k, ind_z] = np.mean(u_sampled_sp)  ### This is the unnormalized distribution over the Z grid for each K
       
    return np.mean(u_sampled_sp)  ### This is the unnormalized distribution over the Z grid for each K


if __name__ == "__main__":
    i = 0
    limit = False
    J_cit = 5
    J_SP = 5

    # Define grid of incentives K to evaluate
    n_K_points = 5
    upper_K = 200

    # Define possible Z's (parameterized OBP schemes)
    full_grid = generate_obp_full_grid(
        z1_start=0.5, z1_stop=0.7, z1_step=0.2,
        z2_start=0.2, z2_stop=0.4, z2_step=0.2,
        z3_start=0, z3_stop=50, z3_step=50,
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

    expected_util = np.zeros((len(full_grid),))
    for i in tqdm(range(len(full_grid))):
        expected_util[i] = run_iteration(i, J_cit, J_SP, upper_K, n_K_points, full_grid, model, assigned_screening_individuals, n_different_patients = len(df_test_w_util_lim))


    expected_util_reshaped = expected_util.reshape((n_K_points, -1))  
    pd.DataFrame(expected_util_reshaped).to_csv("expected_util.csv")

    plt.plot(np.linspace(0, upper_K, n_K_points) , expected_util_reshaped[:,0])
    plt.xlabel(f"Incentive K given Z_grid[0]")
    plt.ylabel("Expected Utility for the SP")
    plt.title("Expected Utility vs Incentive K")

    plt.savefig("expected_utility_vs_K.png")
    plt.close()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the array as an image, with 'coolwarm' indicating magnitude
    im = plt.imshow(expected_util_reshaped, cmap='coolwarm', aspect='auto')

    # Add the legend/scale on the side
    plt.colorbar(im, label='Expected Utility')

    K_vals = full_grid["K_Incentive"].unique()
    Z_records = full_grid[["z1_Threshold_100_BP", "z2_Threshold_50_BP", "z3_Bonus_Euros"]].drop_duplicates().to_dict(orient="records")   

    # --- DYNAMIC Y-AXIS (Incentives K) ---
    # Map the row indices to the actual K values (formatted to 1 decimal place)
    plt.yticks(
        ticks=np.arange(len(K_vals)), 
        labels=[f"{k:.1f}" for k in K_vals]
    )
    # Update the label to be informative and dynamic
    plt.ylabel(f"Incentive K (Range: 0 to {upper_K})", fontsize=12, fontweight='bold')

    # --- DYNAMIC X-AXIS (OBP Schemes Z) ---
    # Create a readable string for each Z combination (e.g., "0.5 | 0.2 | 0")
    Z_labels = [" | ".join([f"{v}" for v in z.values()]) for z in Z_records]

    # Rotate the labels 45 degrees so they don't overlap each other
    plt.xticks(
        ticks=np.arange(len(Z_labels)), 
        labels=Z_labels, 
        rotation=45, 
        ha='right'
    )
    # Update the label dynamically
    plt.xlabel(f"OBP Schemes Z ({len(Z_records)} total combinations [z1 | z2 | z3])", fontsize=12, fontweight='bold')

    # Formatting
    plt.title('Expected Utility Heatmap')
    plt.tight_layout() # Crucial: prevents the rotated X-axis labels from being cut off

    plt.savefig("expected_utility_heatmap.png")