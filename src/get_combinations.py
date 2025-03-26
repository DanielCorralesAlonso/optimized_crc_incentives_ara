
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BDsScore
from pgmpy.factors.discrete import State
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


def get_all_combinations_bn(bn_model):
        nodes = bn_model.get_parents("CRC")
        states = bn_model.states
        outcome_lists = [states[node] for node in nodes]
        
        # Compute the Cartesian product of all outcome states
        all_combinations = list(itertools.product(*outcome_lists))
        
        # Convert to a DataFrame
        df_bn = pd.DataFrame(all_combinations, columns=nodes)

        return df_bn



def get_all_combinations_id_w_optimal_scr(id_net, df_test_w_util_lim, limit = False):

    net2 = pysmile.Network()
    net2.read_file(f"../models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    df_test_w_util_lim = pd.read_csv("../models/df_test_new_w_lim.csv", index_col=0)

    reader = XMLBIFReader("../models/model_bn.xml")
    model = reader.get_model()

    id_net = net2

    if limit == False:
        opt_col_name = "best_option"
    else:
        opt_col_name = "best_option_w_lim"

    # Step 1: Define the grouping columns
    parent_columns = id_net.get_parent_ids("CRC").copy()  # List of column names

    # Step 2: Count occurrences of each "best_option" within each multi-index
    counts = df_test_w_util_lim.groupby(parent_columns + [opt_col_name]).size().reset_index(name="count")

    # Step 3: Pivot the table to have each best_option as a separate column
    pivot_table = counts.pivot_table(index=parent_columns, columns=opt_col_name, values='count', fill_value=0)

    # Step 4: Compute total counts per multi-index (for relative frequency)
    pivot_table['total_count'] = pivot_table.sum(axis=1)

    # Step 5: Assign relative frequency for each best_option
    for col in pivot_table.columns:
        if col != 'total_count':
            pivot_table[f'relative_freq_{col}'] = pivot_table[col] / pivot_table['total_count']

    # Step 6: Reset index to convert MultiIndex to columns
    pivot_table = pivot_table.reset_index()

    # Step 7: Generate all possible multi-index combinations
    all_possible_combinations = pd.MultiIndex.from_product(
        [df_test_w_util_lim[col].unique() for col in parent_columns], names=parent_columns
    )

    # Step 8: Reindex to ensure all combinations exist
    pivot_table = pivot_table.set_index(parent_columns).reindex(all_possible_combinations, fill_value=0).reset_index()

    # Step 9: Select the best option per multi-index (highest count or "No_scr_no_col" if all are 0)
    pivot_table[opt_col_name] = pivot_table.apply(
        lambda row: "No_scr_no_col" if row['total_count'] == 0 else row[pivot_table.columns.difference(parent_columns + ['total_count'])].idxmax(),
        axis=1
    )

    # Step 10: Apply model's best option to missing profile
    missing_profiles = pivot_table[pivot_table["total_count"] == 0].copy()

    net2 = pysmile.Network()
    net2.read_file(f"../models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    for patient_chars in missing_profiles.iloc[:,:7].to_dict(orient="records"):

        for key, value in patient_chars.items():
            net2.set_evidence(key, value)

        net2.update_beliefs()

        # Calculate the probability of CRC for the patient. (Model 1)
        p_crc = net2.get_node_value("CRC")[1]
        p_no_crc = net2.get_node_value("CRC")[0]

        vars1 = net2.get_outcome_ids("Screening")
        arr = np.array(net2.get_node_value("Screening"))
        df_scr = pd.DataFrame(arr.reshape(1,-1), index=["Screening"], columns=vars1)

        # Take the asigned screening decision for the patient. (Model 2)
        scr = df_scr.idxmax(axis=1).values[0]

        pivot_table.loc[pivot_table[parent_columns].eq(list(patient_chars.values())).all(axis=1), opt_col_name] = scr

    # rename in best_option all No_scr_no_col to No_screening
    pivot_table[opt_col_name] = pivot_table[opt_col_name].replace("No_scr_no_col", "No_screening")

    return pivot_table