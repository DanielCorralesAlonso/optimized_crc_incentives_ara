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

from utils import generate_grid


def simulation_step(J_c, N_ara, k, model, assigned_screening_profiles, n_assigned_screening, simulate_all = True):

    p_crc = np.zeros((len(assigned_screening_profiles),))
    p_scr_K = np.zeros((n_assigned_screening,))
    p_evidence_arr = np.zeros((len(assigned_screening_profiles),))

    infer = VariableElimination(model)
    result = infer.query(variables=list(model.get_parents("CRC")), joint=True)

    count = 0
    # NOTE, Can I parallelize this???
    for j_cit in range(J_c):
        for i, patient_chars in enumerate(assigned_screening_profiles.iloc[:,:7].to_dict(orient="records")):
            
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
                scr = assigned_screening_profiles.loc[i, "best_option"]
            except:
                scr = assigned_screening_profiles.loc[i, "best_option_w_lim"]
            

            scr_decision_patient = pd.unique(np.array(["No_screening", scr]))
            # -------------------------
            
            # ----- p_{SP}(s | I, x) ---- Calculate the probability for the citizen to accept screening given incentive K and covariates x
            # ----- This is done via simulation based on adversarial risk analysis.

            # Simulate utility function for the citizen
            n_total_patients_with_chars = int(assigned_screening_profiles.loc[i, "total_count"])  # Approximate number of patients in the population with these characteristics
            
            s_opt = np.zeros((n_total_patients_with_chars,))
            

            if simulate_all:
                # Vectorized simulation for all patients simultaneously (drawing their biases)
                
                p_c_crc = prob_crc_cit(p_crc[i], age) #NOTE, I assume all patients with the same characteristics have the same percieved probability of having CRC. 
                p_belief = np.array([p_c_crc for _ in range(n_total_patients_with_chars)])
                cit_comfort_noise = 100 * np.random.normal(1.0, 0.2, size=n_total_patients_with_chars)

                s_opt_count = np.zeros(n_total_patients_with_chars,)
                expected_utilities = np.zeros((len(scr_decision_patient), n_total_patients_with_chars))

                # pdb.set_trace()
                for f in range(N_ara):
                    for j_s, scr_ in enumerate(scr_decision_patient.tolist()):
                        
                        c_sim = np.random.binomial(1, p_belief)
                        r_sim = np.random.binomial(1, sensitivity(scr_) * c_sim + (1 - specificity(scr_)) * (1 - c_sim))
                        
                        # cost_cit vectorized equivalent
                        val_QALY = 30968 * EQ5D(age) * np.array([diff_QALY(c, r) for c, r in zip(c_sim, r_sim)])
                        if scr_ == "No_screening":
                            payoff = val_QALY
                        else:
                            payoff = val_QALY + k - (cit_comfort_noise / comfort(scr_)) - 1000 * r_sim

                        expected_utilities[j_s] = payoff  #random_utilities_cit(payoff) # Replace with random_utilities_cit(payoff) if non-linear
                    
                    s_opt_count += np.argmax(expected_utilities, axis=0)
                

            else:
                ### Simulate_per_profile (assuming identical outcome mapped over cohort)
                p_belief = prob_crc_cit(p_crc[i], age)
                cit_comfort_noise = 100 * np.random.normal(1.0, 1.0)

                expected_utilities = np.zeros((len(scr_decision_patient),))
                for j_s, scr_ in enumerate(scr_decision_patient.tolist()):
                    future_utilities = np.zeros(N_ara) # N_ara futures to calc expected utility
                    for f in range(N_ara):
                        c_sim = np.random.binomial(1, p_belief)
                        r_sim = np.random.binomial(1, sensitivity(scr_) * c_sim + (1 - specificity(scr_)) * (1 - c_sim))
                        
                        val_QALY = 30968 * EQ5D(age) * diff_QALY(c_sim, r_sim)
                        if scr_ == "No_screening":
                            payoff = val_QALY
                        else:
                            payoff = val_QALY + k - (cit_comfort_noise / comfort(scr_)) - 1000 * r_sim
                        future_utilities[f] = payoff # random_utilities_cit(payoff)
                        pdb.set_trace()

                    expected_utilities[j_s] = np.mean(future_utilities)
                
                s_opt += np.argmax(expected_utilities)


            p_scr_K[count:count + n_total_patients_with_chars] = s_opt_count / N_ara
            count += n_total_patients_with_chars
            # pdb.set_trace()
           

    # pdb.set_trace()
    return pd.DataFrame( data = p_scr_K , columns=[f"p_scr_K_{k:.2f}"] )  ### This is the unnormalized distribution over the Z grid for each K




def run_simulation(limit=False, J_c = 1, N_ara=1, n_K_points=1, upper_K=250):

    net2 = pysmile.Network()
    net2.read_file(f"models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    df_test_w_util_lim = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)

    reader = XMLBIFReader("models/model_bn.xml")
    model = reader.get_model()

    best_options = get_all_combinations_id_w_optimal_scr(net2, df_test_w_util_lim, limit = limit)

    try:
        assigned_screening_profiles = best_options[ best_options["best_option_w_lim"] != "No_screening" ].reset_index(drop=True).copy()
    except:
        assigned_screening_profiles = best_options[ best_options["best_option"] != "No_screening" ].reset_index(drop=True).copy()


    assigned_screening_profiles.drop(columns= assigned_screening_profiles.iloc[:, 11:-1].columns , inplace=True)

    # Define the number of simulations for the adversarial risk analysis
    # N = 1 (passed as arg)

    # Define grid of incentives K to evaluate
    full_grid = generate_grid(upper_K=upper_K, n_K_points=n_K_points)

    # Run simulations for each incentive level K
    n_assigned_screening = int(assigned_screening_profiles["total_count"].sum())
    simulated_df = pd.DataFrame(index=range(n_assigned_screening), columns= list(assigned_screening_profiles.iloc[:, :7].columns) )

    accumulated_count = 0
    for i, count in assigned_screening_profiles.iterrows():
        simulated_df.loc[accumulated_count:accumulated_count + count["total_count"] - 1, assigned_screening_profiles.columns[:7]] = assigned_screening_profiles.loc[i, assigned_screening_profiles.columns[:7]].values
        accumulated_count += count["total_count"]
        

    results = []
    for i in tqdm(range(len(full_grid))):
        k = full_grid.loc[i, "K_Incentive"]
        k = float(k)

        p_scr_K = simulation_step(
            J_c,
            N_ara, 
            k, 
            model, 
            assigned_screening_profiles, 
            n_assigned_screening=n_assigned_screening, 
            )
        
        simulated_df = pd.concat([simulated_df, p_scr_K], axis=1)

        # pdb.set_trace()

    simulated_df.to_csv("models/simulated_data.csv", index=False)

    return simulated_df


def train_emulator(simulated_df, full_grid):

    y = np.zeros((len(simulated_df)*len(full_grid),))
    X = pd.DataFrame(index=range(len(simulated_df)*len(full_grid)), columns= list(simulated_df.columns[:-len(full_grid)]) + ["K"] )
    for i in range(len(full_grid )) :
        k_val = float( full_grid.loc[i, "K_Incentive"])
        y[i*len(simulated_df):(i+1)*len(simulated_df)] = simulated_df[f"p_scr_K_{k_val:.2f}"].values
        X.loc[i*len(simulated_df):(i+1)*len(simulated_df)-1, "K"] = k_val
        X.loc[i*len(simulated_df):(i+1)*len(simulated_df)-1, list(simulated_df.columns[:-len(full_grid)])] = simulated_df[simulated_df.columns[:-len(full_grid)]].values  


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

    pdb.set_trace()

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
    

    return