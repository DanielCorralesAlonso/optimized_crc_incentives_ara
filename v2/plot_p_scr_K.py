import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pysmile
import pysmile_license

from costs_and_utilities import p_screen_ara
from patients import patient


N_ara = 200  # Number of ARA samples (principal's uncertainty about this specific citizen's private parameters)
N_mc_evals = 50  # Number of MC draws the citizen uses to compute their own Expected Utility

def plot_histograms_count_distrib(P, net2, patient_chars):

    for key, value in patient_chars.items():
        net2.set_evidence(key, value)

    net2.update_beliefs()

    p_crc = net2.get_node_value("CRC")[1]

    vars1 = net2.get_outcome_ids("Screening")
    arr = np.array(net2.get_node_value("Screening"))
    df_scr = pd.DataFrame(arr.reshape(1,-1), index=["Screening"], columns=vars1)

    # take column with the highest value
    scr = df_scr.idxmax(axis=1).values[0]
    scr_decision_patient = pd.unique(np.array(["No_screening", scr]))

    print(f"Optimal screening: {scr}")
    print(f"p_crc = {p_crc}")

    # We are evaluating ONE specific citizen defined by patient_chars
    age = patient_chars["Age"]

    for k in [0, 1, 20, 50, 100, 150, 200]:
        # Each outer iteration is one ARA experiment → distribution of p_scr estimates
        plt_arr = [
            p_screen_ara(p_crc, age, k, scr_decision_patient, N_ara)
            for _ in range(100)
        ]

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