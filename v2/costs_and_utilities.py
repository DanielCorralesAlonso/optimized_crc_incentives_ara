import pdb

import numpy as np


u_EQ5D = {
    "age_1_young_adult": 0.966,
    "age_2_young": 0.963,
    "age_3_young_adult": 0.939,
    "age_4_adult": 0.911,
    "age_5_old_adult": 0.884,
}

def EQ5D(age):
    return u_EQ5D[age]


scr_costs_dict = {
    "No_screening": 0,
    "gFOBT": 12.14,
    "FIT": 14.34,
    "Blood_based": 123.13,
    "Stool_DNA": 236.88,
    "CTC": 95.41,
    "CC": 510.24,
    "Colonoscopy": 1000
}

def scr_costs(scr):
    return scr_costs_dict[scr]

sensitivity_dict = {
    "No_screening": 0,
    "gFOBT": 0.45,
    "FIT": 0.75,
    "Blood_based": 0.66,
    "Stool_DNA": 0.923,
    "CTC": 0.8,
    "CC": 0.87,
    "Colonoscopy": 0.97
}

def sensitivity(scr):
    return sensitivity_dict[scr]

specificity_dict = {
    "No_screening": 1,
    "gFOBT": 0.978,
    "FIT": 0.966,
    "Blood_based": 0.91,
    "Stool_DNA": 0.866,
    "CTC": 0.89,
    "CC": 0.92,
    "Colonoscopy": 0.99
}

def specificity(scr):
    return specificity_dict[scr]

comfort_dict = {
    "No_screening": 4,
    "gFOBT": 3,
    "FIT": 3,
    "Blood_based": 3,
    "Stool_DNA": 3,
    "CTC": 2,
    "CC": 2,
    "Colonoscopy": 1
}

def comfort(scr):
    return comfort_dict[scr]


def diff_QALY(crc, r_scr):
    if crc == 1 and r_scr == 1:
        return np.random.uniform(5, 10)
    elif crc == 1 and r_scr == 0:
        return - np.random.uniform(3, 5)
    else:
        return 0 
    

def cost_PM(age, crc, scr, r_scr, K):
    if scr == "No_screening":
        return 30968*EQ5D(age)*diff_QALY(crc, r_scr)
    else:
        return 30968*EQ5D(age)*diff_QALY(crc, r_scr) - K - scr_costs(scr) - 25955*crc*r_scr
    
def cost_SP(age, crc, scr, scr_decision, r_scr, K):
    if scr == "No_screening" or scr_decision == 0:
        return 30968*EQ5D(age)*diff_QALY(crc, r_scr)
    else:
        return 30968*EQ5D(age)*diff_QALY(crc, r_scr) - K - scr_costs(scr) - 25955*crc*r_scr 

def cost_cit(age, crc, scr, r_scr, K, cit_comfort_noise=None):
    if isinstance(crc, np.ndarray):
        val_QALY = 30968 * EQ5D(age) * np.array([diff_QALY(c, r) for c, r in zip(crc, r_scr)])
    else:
        val_QALY = 30968 * EQ5D(age) * diff_QALY(crc, r_scr)
    if scr == "No_screening":
        return val_QALY
    else:
        return val_QALY + K - (cit_comfort_noise / comfort(scr)) - 3000 * r_scr



# Define how much each age group underestimates their risk
age_underestimation_factors = {
    "age_1_young_adult": 0.1,  # Perceives 10% of their actual risk
    "age_2_young": 0.2,        # Perceives 20% of their actual risk
    "age_3_young_adult": 0.3,  # Perceives 30% of their actual risk
    "age_4_adult": 0.5,        # Perceives 50% of their actual risk
    "age_5_old_adult": 0.8     # Perceives 80% of their actual risk
}

def prob_crc_cit(true_p_crc, age):
    base_factor = age_underestimation_factors[age]
    
    # Add a little uniform noise so not everyone in the age group is identical
    factor = base_factor * np.random.uniform(0.9999, 1.0001)
    factor = min(max(factor, 0.01), 0.99)
    
    r = true_p_crc * factor
    if r <= 0:
        return 0
        
    var = (r * 0.01)**2
    
    d1 = ((1-r)/var - 1/r)*r**2
    d2 = d1*(1/r - 1)
    
    if d1 <= 0 or d2 <= 0:
        return r
        
    return np.random.beta(d1, d2)



def random_utilities_SP(cost, mu_alpha=-20, sigma_alpha=0.1):
    # 2. Sample the parameter internally
    #sampled_alphas = np.random.lognormal(mean=mu_alpha, sigma=sigma_alpha)
    
    # 3. Calculate the utility using the sampled parameters
    # u_sampled_SP = 1 - np.exp(-sampled_alphas * cost)
    u_sampled_SP = cost
    return u_sampled_SP

def random_utilities_cit(sampled_alphas, age, crc, scr, r_scr, K, cit_comfort_noise=None):
    cost = cost_cit(age, crc, scr, r_scr, K, cit_comfort_noise)
    
    #exponent = np.clip(-sampled_alphas * cost, -700, 700)

    #u_sampled_SP = 1 - np.exp(exponent)
    u_sampled_SP = cost
    
    return u_sampled_SP