import numpy as np
import pandas as pd


def generate_grid(z1_start = None, z1_stop = None, z1_step = None, 
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