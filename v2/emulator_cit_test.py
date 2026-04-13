import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
import random

from pgmpy.readwrite import XMLBIFReader
import itertools

import pysmile
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import pysmile_license
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import pdb

from costs_and_utilities import *
from patients import patient
# from v2.dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import xgboost as xgb
import joblib


def simulation_step(N, k, assigned_screening_individuals, model):

    simulated_df = pd.DataFrame(index=range(len(assigned_screening_individuals)), columns= list(assigned_screening_individuals.columns) + ["p_scr_K", "K"] )

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


        # Simulate utility function for the citizen
        s_opt = np.zeros((N,))
        for j_cit in range(N):

            ### Problem is (linear in utility?) thus I can avoid simulating for c and r.
            ### NOTE, here we are assuming all patients with the same characteristics have the same probabilities of CRC and make the same decisions.
            ### I think this is probably true in expectation.
            arr = np.array( [
                    sensitivity(scr_) * prob_crc_cit(age) * cost_cit(age, crc=1, r_scr=1, scr = scr_, K=k) +
                    (1 - specificity(scr_)) * (1-prob_crc_cit(age)) * cost_cit(age, crc=0, r_scr=1, scr = scr_, K=k)
                +
                    (1 - sensitivity(scr_)) * prob_crc_cit(age) * cost_cit(age, crc=1, r_scr=0, scr = scr_, K=k) +
                    specificity(scr_) * (1-prob_crc_cit(age)) * cost_cit(age, crc=0, r_scr=0, scr = scr_, K=k)
                for scr_ in scr_decision_patient.tolist()] )

            s_opt[j_cit] = np.argmax(arr)


            ### Other possibility
            # n_total_patients_with_chars = int(n_different_patients * p_evidence_arr[i])  # Approximate number of patients in the population with these characteristics
            # c_sim = np.array([np.random.binomial(1, p_crc[i]) for _ in range(n_total_patients_with_chars)])

            #for j_s, scr_ in enumerate(scr_decision_patient.tolist()):
            #    r_sim = np.array([np.random.binomial(1, sensitivity(scr_) * c_sim[j] + (1 - specificity(scr_)) * (1 - c_sim[j]))  for j in range(n_total_patients_with_chars)])
            #    cost_sim[j_s] = np.array([cost_SP(age=age, crc=c_sim[j], scr=scr_, r_scr=r_sim[j], K=k) for j in range(n_total_patients_with_chars)])
            
            # Average cost across the simulated patients with these characteristics
            #s_opt[j_cit, :] = np.argmax(cost_sim, axis=0)
            


        # Approximate the probability of each decision
        p_scr_K = s_opt.sum() / N

        simulated_df.loc[i, assigned_screening_individuals.columns] = assigned_screening_individuals.loc[i]
        simulated_df.loc[i, "p_scr_K"] = p_scr_K
        simulated_df.loc[i, "K"] = k

    return simulated_df  ### This is the unnormalized distribution over the Z grid for each K




if __name__ == "__main__":

    limit = False

    simulate = True

    if simulate:
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


        assigned_screening_individuals.drop(columns= assigned_screening_individuals.iloc[:, 7:-1].columns, inplace=True)

        # Define the number of simulations for the adversarial risk analysis
        N = 200

        # Define grid of incentives K to evaluate
        n_K_points = 50
        upper_K = 100
        K_grid = np.linspace(0, upper_K, n_K_points)

        # Run simulations for each incentive level K
        results = []
        for k in tqdm(K_grid):
            simulated_df = simulation_step(N, k, assigned_screening_individuals, model)
            results.append(simulated_df)

        # Combine results into a single DataFrame
        final_results = pd.concat(results, ignore_index=True)

        final_results.to_csv("models/simulated_data.csv", index=False)

    else: 
        final_results = pd.read_csv("models/simulated_data.csv")


    y = final_results["p_scr_K"].values
    X = final_results.drop(columns=["p_scr_K", 'best_option']) 

    pdb.set_trace()

    # 2. Convert string/object columns to pandas 'category' dtype
    # This is required for XGBoost's native categorical support
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # Persist feature schema and category vocabularies for robust inference.
    feature_metadata = {
        "feature_columns": X.columns.tolist(),
        "categorical_levels": {
            col: X[col].cat.categories.tolist() for col in cat_cols
        }
    }

    # 3. Initialize the Regressor with 'enable_categorical=True'
    # Objective 'binary:logistic' ensures output is [0, 1]
    model = xgb.XGBRegressor(
        objective='binary:logistic',
        enable_categorical=True, 
        tree_method='hist',  # Required for categorical support
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        max_depth=10
    )

    # 4. K-Fold Cross-Validation with multiple metrics
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)

    # 5. Final Training to get sample predictions
    # (We'll use a simple split here just to show the comparison sample)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 5. Output Metrics
    print("--- Cross-Validation Metrics ---")
    print(f"MAE:  {-cv_results['test_mae'].mean():.4f} (Avg error in probability)")
    print(f"RMSE: {-cv_results['test_rmse'].mean():.4f} (Penalizes large misses)")
    print(f"R2:    {cv_results['test_r2'].mean():.4f} (Variance explained - 1.0 is perfect)")

    print("\n--- Sample of True vs. Predicted Probabilities ---")
    comparison_df = pd.DataFrame({
        # 'Covariates': X_test.reset_index(drop=True),
        'Actual_Prob': y_test,
        'Predicted_Prob': y_pred,
        'Abs_Error': np.abs(y_test - y_pred)
    }).reset_index(drop=True)

    print(comparison_df.head(10))

    # Save trained model and feature metadata for future use
    joblib.dump(model, "models/xgb_cit_model.pkl")
    joblib.dump(feature_metadata, "models/xgb_cit_model_meta.pkl")
    

