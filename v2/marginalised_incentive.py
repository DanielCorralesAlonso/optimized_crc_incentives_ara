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

from utilities_test import *
from patients import patient
from dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

plt.rcParams.update({'font.size': 15})  # You can adjust the font size as needed


def run_iteration(i, upper_K, n_K_points, total_sim, model, combinations_bn, best_options):

    v_x_K_arr = np.zeros((n_K_points,))

    for ind_k, k in enumerate(np.linspace(0, upper_K, n_K_points)):

        v_K = np.zeros((2,))
    
        for i, patient_chars in enumerate(best_options.iloc[:,:7].to_dict(orient="records")):
            
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

            p_crc = float(infer.query(variables=["CRC"], evidence=evidence).values[1])
            p_no_crc = float(infer.query(variables=["CRC"], evidence=evidence).values[0])

            try:
                scr = best_options.loc[i, "best_option"]
            except:
                scr = best_options.loc[i, "best_option_w_lim"]
            
            scr_decision_patient = np.unique(["No_screening", scr]).tolist()
            
            if scr != "No_screening":
                count_arr = np.zeros(len(scr_decision_patient))
                for _ in range(total_sim):

                    # Calculate the expected utility of the citizen for each screening decision
                    arr = np.array( [
                            sensitivity(scr) * prob_crc_cit(age) * utility_cit(age, crc=1, r_scr=1, scr = scr, K=k) +
                            (1 - specificity(scr)) * (1-prob_crc_cit(age)) * utility_cit(age, crc=0, r_scr=1, scr = scr, K=k)
                        +
                            (1 - sensitivity(scr)) * prob_crc_cit(age) * utility_cit(age, crc=1, r_scr=0, scr = scr, K=k) +
                            specificity(scr) * (1-prob_crc_cit(age)) * utility_cit(age, crc=0, r_scr=0, scr = scr, K=k)
                        for scr in scr_decision_patient] )

                    # Save the decision with highest expected utility.
                    argmax = np.argmax(arr)
                    count_arr[argmax] += 1
                    

                # Approximate the probability of each decision
                p_scr_K = count_arr / total_sim
            else:
                p_scr_K = [1]


            # Calculate the expected utility of the government for each incentive amount K
            v_x_K = [ p_scr_K[i] * (
                sensitivity(scr) * p_crc * utility_PM(age, crc=1, r_scr=1, scr = scr, K=k) +
                (1 - specificity(scr)) * p_no_crc * utility_PM(age, crc=0, r_scr=1, scr = scr, K=k)
            +  
                (1 - sensitivity(scr)) * p_crc * utility_PM(age, crc=1, r_scr=0, scr = scr, K=k) +
                specificity(scr) * p_no_crc * utility_PM(age, crc=0, r_scr=0, scr = scr, K=k)
            ) for i, scr in enumerate(scr_decision_patient)] 


            # pdb.set_trace()
            v_K += p_evidence * np.array(v_x_K)

        v_x_K_arr[ind_k] = sum(v_K)
       
    return v_x_K_arr



if __name__ == "__main__": 

    for limit in [False]: # [True, False]:

        # ---------------------- Run the Simulation ----------------------
        total_sim = 100

        n_K_points = 20
        upper_K = 200
        n_random_trials = 5

        v_scr_K_iter = []

        net2 = pysmile.Network()
        net2.read_file(f"models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
        net2.clear_all_evidence()

        df_test_w_util_lim = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)

        reader = XMLBIFReader("models/model_bn.xml")
        model = reader.get_model()

        best_options = get_all_combinations_id_w_optimal_scr(net2, df_test_w_util_lim, limit = limit)
        combinations_bn = get_all_combinations_bn(model)

        


        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_iteration, i, upper_K, n_K_points, total_sim, model, combinations_bn, best_options) for i in range(n_random_trials)]
            
            for future in tqdm(as_completed(futures), total=n_random_trials, desc="Processing iterations"):
                v_scr_K_iter.append(future.result())


        # ---------------------- Plot the Results ----------------------
        plt.figure(figsize=(6, 5))
        # plot each scr  in a differnte color
        plt.plot(np.linspace(0, upper_K, n_K_points), np.mean(np.stack(v_scr_K_iter), axis = 0))
        # Add best K in plot as a line and in legend
        opt_val_loc = np.argmax(np.mean(np.stack(v_scr_K_iter), axis = 0))
        opt_K = np.linspace(1, upper_K, n_K_points)[opt_val_loc]
        plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K:.2f}")
        plt.tight_layout()

        pd.DataFrame(v_scr_K_iter).to_csv(f"df_v_scr_K_iter_limit_{limit}.csv")

        plt.legend(loc='upper right')
        plt.xlabel("K")
        plt.ylabel(r"$\phi(I)$")
        plt.title(f"Marginalized Incentive Over The Entire Population")
        plt.savefig(f"outputs/overall_limit_{limit}_best_K_ARA.png", bbox_inches='tight')
        plt.close()


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
        # ----------------