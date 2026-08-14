"""
Dump the (age, p_crc, scr, n) profiles to profiles.csv.

Run from the REPO ROOT (get_combinations.py reads "models/..." relatively):
    python v2/dump_profiles.py
"""
import sys
import pandas as pd

sys.path.insert(0, "v2")

import pysmile
import pysmile_license
from pgmpy.readwrite import XMLBIFReader
from get_combinations import get_all_combinations_id_w_optimal_scr
from simulator_cit_p_scr import build_profiles

net2 = pysmile.Network()
net2.read_file("models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
net2.clear_all_evidence()
df_test = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)
model = XMLBIFReader("models/model_bn.xml").get_model()

# Emit BOTH assignments.  limit=True gives best_option_w_lim (capacity-limited,
# FIT-dominant); limit=False gives best_option (unlimited, Stool_DNA-dominant).
# The test mix differs enormously between them and drives the programme costs.
for limit, tag in [(True, "wlim"), (False, "nolim")]:
    best = get_all_combinations_id_w_optimal_scr(net2, df_test, limit=limit)
    col = "best_option_w_lim" if "best_option_w_lim" in best.columns else "best_option"
    assigned = best[best[col] != "No_screening"].reset_index(drop=True).copy()

    out = pd.DataFrame(build_profiles(model, assigned),
                       columns=["age", "p_crc", "scr", "n"])
    path = f"profiles_{tag}.csv"
    out.to_csv(path, index=False)

    print("\n" + "=" * 60)
    print(f"limit={limit}   column={col}   ->  {path}")
    print("=" * 60)
    print(f"{len(out)} profiles, N = {out['n'].sum():,}")
    print("by test:", dict(out.groupby("scr")["n"].sum()))
    print("by age :", dict(out.groupby("age")["n"].sum()))
    print(f"p_crc  min {out['p_crc'].min():.5f}   median {out['p_crc'].median():.5f}"
          f"   max {out['p_crc'].max():.5f}")
    print("n-weighted mean p_crc = "
          f"{(out['p_crc'] * out['n']).sum() / out['n'].sum():.5f}")
