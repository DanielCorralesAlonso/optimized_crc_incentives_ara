import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pysmile
import pysmile_license

from costs_and_utilities import sensitivity, specificity, random_utilities_cit, prob_crc_cit
from patients import patient



N_ara = 200  # Number of ARA samples (principal's uncertainty about this specific citizen's private parameters)
N_mc_evals = 50  # Number of MC draws the citizen uses to compute their own Expected Utility

def plot_histograms_count_distrib(P, net2, patient_chars):

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

    print(f"Optimal screening: {scr}")
    print(f"p_crc = {p_crc}")

    # We are evaluating ONE specific citizen defined by patient_chars
    age = patient_chars["Age"]

    for ind_k, k in enumerate([0 , 1, 50, 100, 200]):
        plt_arr = []

        # We calculate the probability of THIS specific citizen accepting screening.
        # We loop 20 times just to generate a histogram of that probability to verify stability.
        for _ in range(50):
            accept_count = 0
            
            # ARA loop: Simulate unknown private parameters characterizing THIS citizen
            for _ in range(N_ara):
                # Draw citizen's private beliefs and preferences
                p_belief = prob_crc_cit(p_crc, age) 
                cit_comfort_noise = 500 * np.random.normal(1.0, 0.2)
                
                expected_utilities = np.zeros((len(scr_decision_patient), ))

                sampled_alphas_cit = np.random.lognormal(mean=-2, sigma=0.1)
                
                # NOTE: MC loop or analytical calculation????
                #pdb.set_trace()
                expected_utilities = np.array( [
                        sensitivity(scr_) * prob_crc_cit(p_crc, age) * random_utilities_cit(sampled_alphas_cit, age, crc=1, r_scr=1, scr = scr_, K=k, cit_comfort_noise=cit_comfort_noise) +
                        (1 - specificity(scr_)) * (1-prob_crc_cit(p_crc, age)) * random_utilities_cit(sampled_alphas_cit, age, crc=0, r_scr=1, scr = scr_, K=k, cit_comfort_noise=cit_comfort_noise)
                    +
                        (1 - sensitivity(scr_)) * prob_crc_cit(p_crc, age) * random_utilities_cit(sampled_alphas_cit, age, crc=1, r_scr=0, scr = scr_, K=k, cit_comfort_noise=cit_comfort_noise) +
                        specificity(scr_) * (1-prob_crc_cit(p_crc, age)) * random_utilities_cit(sampled_alphas_cit, age, crc=0, r_scr=0, scr = scr_, K=k, cit_comfort_noise=cit_comfort_noise)
                    for scr_ in scr_decision_patient] )

                # The citizen evaluates the EXPECTED utility of their choices
                '''for f in range(N_mc_evals):
                    c_sim = np.random.binomial(1, p_belief)

                    for j_s, scr_ in enumerate(scr_decision_patient):
                        r_sim = np.random.binomial(1, sensitivity(scr_) * c_sim + (1 - specificity(scr_)) * (1 - c_sim))
                        payoff = cost_cit(age=age, crc=c_sim, scr=scr_, r_scr=r_sim, K=k, cit_comfort_noise=cit_comfort_noise)
                        
                        expected_utilities[j_s] += payoff # random_utilities_cit(payoff)
                
                expected_utilities /= N_mc_evals'''
                # pdb.set_trace()
                # Citizen's decision happens *once* based on their full Expected Utility
                if np.argmax(expected_utilities) == 1:
                    accept_count += 1

            # Probability THIS citizen accepts screening at this K
            p_scr = accept_count / N_ara
            plt_arr.append(p_scr)

        plt_arr = np.array(plt_arr)
        # for each K, plot the distribution of p_scr with a different color
        plt.hist(plt_arr, bins = 16 , alpha=0.5, label=str(k))
        plt.xlim(-0.05, 1.05)
        plt.legend(loc='upper right')
        plt.title("Age = " + age)
        print(f"K = {k}, mean_p_scr = {np.mean(plt_arr):.4f}, median_p_scr = {np.median(plt_arr):.4f}")
        
    plt.savefig(f"outputs/p_scr_K/p_scr_hist_age_patient_{P}.png")



if __name__ == "__main__":

    # read patient num from terminal as python main_test.py 1 if given, else 1
    patient_num = int(os.sys.argv[1]) if len(os.sys.argv) > 1 else 1


    # ---------------------- Load the Model (Model 2 containing Model 1) ----------------------
    net2 = pysmile.Network()
    net2.read_file(f"models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()


    # ---------------------- Set the Patient Characteristics ----------------------
    patient_chars = patient(patient_num=patient_num)

    plot_histograms_count_distrib(patient_num, net2, patient_chars)