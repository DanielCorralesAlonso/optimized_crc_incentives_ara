import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utilities import *

def plot_histograms_count_distrib(net2, patient_chars):

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

    total_sim = 100
    count = 0
    age = patient_chars["Age"]

    

    scr_decision_patient = ["No_screening", scr]


    for ind_k, k in enumerate([0, 1, 50, 100, 200]):
        plt_arr = []
        
        for _ in range(100):
            count_arr = np.zeros(len(scr_decision_patient))
            for _ in range(total_sim):
                arr = np.array( [
                    sensitivity(scr) * p_crc * utility_cit(age, crc=1, r_scr=1, scr = scr, K=k) +
                    (1 - specificity(scr)) * p_no_crc * utility_cit(age, crc=0, r_scr=1, scr = scr, K=k)
                +
                    (1 - sensitivity(scr)) * p_crc * utility_cit(age, crc=1, r_scr=0, scr = scr, K=k) +
                    specificity(scr) * p_no_crc * utility_cit(age, crc=0, r_scr=0, scr = scr, K=k)
                for scr in scr_decision_patient] )

                argmax = np.argmax(arr)
                count_arr[argmax] += 1
                    
            p_scr = count_arr / total_sim
            plt_arr.append(p_scr)
        
        plt_arr = np.array(plt_arr)
        # for each K, plot the distribution of p_scr with a different color
        plt.hist(plt_arr[:, 1], bins = 16 , alpha=0.5, label=str(k))
        plt.legend(loc='upper right')
        plt.title("Age = " + age)
        print(f"K = {k}, mean_p_scr = {np.mean(plt_arr[:,1])}, median_p_scr = {np.median(plt_arr[:,1])}")
                    

    plt.savefig("outputs/histograms.png")