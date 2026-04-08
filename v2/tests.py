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

from costs_and_utilities import *
from patients import patient
# from v2.dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



import numpy as np
import pandas as pd

import emulator_cit_test

# Define grid parameters
n_K_points = 20
upper_K = 200

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
    p_scr_K_emulator = joblib.load("models/xgb_cit_model.pkl")
    feature_metadata = joblib.load("models/xgb_cit_model_meta.pkl")

    # Keep feature order identical to training to avoid schema drift at predict time.
    emulator_feature_cols = feature_metadata.get("feature_columns", [])
    categorical_levels = feature_metadata.get("categorical_levels", {})

    z = {
        "z1_Threshold_100_BP": full_grid.loc[i, "z1_Threshold_100_BP"],
        "z2_Threshold_50_BP": full_grid.loc[i, "z2_Threshold_50_BP"],
        "z3_Bonus_Euros": full_grid.loc[i, "z3_Bonus_Euros"]
    }

    u_sampled_sp = np.zeros((J_SP,))
    c_counter_sp, r_counter_sp = 0, 0
    for ind_j in range(J_SP):

        p_crc = np.zeros((len(assigned_screening_individuals),))
        p_scr_K = np.zeros((len(assigned_screening_individuals),))
        p_evidence_arr = np.zeros((len(assigned_screening_individuals),))
    
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

            p_evidence_arr[i] = result.get_value(**evidence)


            # ---- p_{PM}(c | x) ---- Calculate the probabiltiy of having CRC 
            p_crc[i] = float(infer.query(variables=["CRC"], evidence=evidence).values[1])
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
            # ----- This is done via emulation based on adversarial risk analysis.
            simulation_required = False
            
            
            if simulation_required:
                ### MISSING 
                emulator_cit_test()
            else:
                row_dict = assigned_screening_individuals.loc[i, assigned_screening_individuals.columns[:7]].to_dict()
                row_dict["K"] = float(k)
                X = pd.DataFrame([row_dict])
                X = X.reindex(columns=emulator_feature_cols)

                for col, levels in categorical_levels.items():
                    X[col] = pd.Categorical(X[col], categories=pd.Index(levels))
                
                p_scr_K[i] = p_scr_K_emulator.predict(X).squeeze()


            # ------------------------
        # SOMETHING IS OFF, best solution is always the quantity after zero incentive.
        # This works if we only have cost, not sure if it works if we have also the utility part.
        total_screened = int ( np.matmul(p_scr_K, p_evidence_arr) * n_different_patients )
        total_detected = int( (p_scr_K * p_crc * sensitivity(scr) * p_evidence_arr * n_different_patients).sum() )
        total_cost = - 20 * total_screened - 2000 * total_detected - k * total_screened
        if z["z1_Threshold_100_BP"] is not None and z["z2_Threshold_50_BP"] is not None and z["z3_Bonus_Euros"] is not None:
            total_cost += z["z3_Bonus_Euros"] * total_detected
            if total_screened / (n_different_patients * p_evidence_arr.sum()) >= z["z1_Threshold_100_BP"]:
                total_cost += 20 * total_screened
            elif  z["z1_Threshold_100_BP"] > total_detected / total_screened >= z["z2_Threshold_50_BP"]:
                total_cost += 20 * total_screened / 2
        u_sampled_sp[ind_j] = total_cost 

        # What if we do have utilities??? Do we instantly need another emulator?



    return np.mean(u_sampled_sp)  ### This is the unnormalized distribution over the Z grid for each K


if __name__ == "__main__":
    i = 0
    limit = False
    J_cit = 5
    J_SP = 1

    # Define grid of incentives K to evaluate
    n_K_points = 10
    upper_K = 5

    # Define possible Z's (parameterized OBP schemes)
    full_grid = generate_obp_full_grid(
        z1_start=0.5, z1_stop=0.6, z1_step=0.1,
        z2_start=0.3, z2_stop=0.5, z2_step=0.1,
        z3_start=3000, z3_stop=6000, z3_step=1000,
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
        expected_util[i] = run_iteration(i, J_SP, J_cit, full_grid, model, assigned_screening_individuals, n_different_patients = len(df_test_w_util_lim))


    # pdb.set_trace()
    expected_util_reshaped = expected_util.reshape((-1, n_K_points)) 
    pd.DataFrame(expected_util_reshaped).to_csv("expected_util.csv")

    plt.plot(np.linspace(0, upper_K, n_K_points) , expected_util_reshaped[0])
    plt.xlabel(f"Incentive K given Z_grid[0]")
    plt.ylabel("Expected Utility for the SP")
    plt.title("Expected Utility vs Incentive K")

    plt.savefig("expected_utility_vs_K.png")
    plt.close()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the array as an image, with 'coolwarm' indicating magnitude
    heatmap_data = expected_util_reshaped.T
    max_abs = np.nanmax(np.abs(heatmap_data))
    im = plt.imshow(heatmap_data, cmap='coolwarm', aspect='auto', vmin=-max_abs, vmax=max_abs)

    # Mark the highest-value cell with a star and centered "1st" label.
    max_row, max_col = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
    plt.scatter(
        max_col,
        max_row,
        marker='*',
        s=500,
        c='gold',
        edgecolors='black',
        linewidths=1.2,
        zorder=5
    )

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