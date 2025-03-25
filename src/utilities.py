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
    "No_screening": 0,
    "gFOBT": 1,
    "FIT": 1,
    "Blood_based": 1,
    "Stool_DNA": 1,
    "CTC": 2,
    "CC": 2,
    "Colonoscopy": 3
}

def comfort(scr):
    return comfort_dict[scr]


def diff_QALY(crc, r_scr):
    if crc == 1 and r_scr == 1:
        return np.random.uniform(5, 15)
    elif crc == 1 and r_scr == 0:
        return - np.random.uniform(2, 5)
    else:
        return 0 


def prob_crc_cit(age):
    if age == "age_5_old_adult":
        r = 0.0001
        var = (0.00005)**2
    else:
        r = 0.00001
        var = (0.000005)**2

    d1 = ((1-r)/var - 1/r)*r**2
    d2 = d1*(1/r - 1)
    return np.random.beta(d1, d2)


def utility_gov(age, crc, scr, r_scr, K):
    if scr == "No_screening":
        return 30000*EQ5D(age)*diff_QALY(crc, r_scr)
    else:
        return 30000*EQ5D(age)*diff_QALY(crc, r_scr) - K*EQ5D(age) - scr_costs(scr) - 30000*crc*r_scr

def utility_cit(age, crc, scr, r_scr, K):
    if scr == "No_screening":
        return 30000*EQ5D(age)*diff_QALY(crc, r_scr)
    else:
        return 30000*EQ5D(age)*diff_QALY(crc, r_scr) + K*EQ5D(age) - scr_costs(scr)*np.random.uniform(1 - 0.9/comfort(scr), 1 - 0.6/comfort(scr))
