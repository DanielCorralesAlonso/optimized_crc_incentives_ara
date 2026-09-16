# Example patients.  The PM's risk uses the seven observed covariates (PA, SD,
# Smoking, BMI, Alcohol, Sex, Age) and the three comorbidities (Diabetes,
# Hypertension, Hyperchol_); the citizen's risk uses the seven only.  A patient must
# specify all ten.

patient_1 = {
    "Age": "age_5_old_adult",
    "Sex": "M",
    "SD": "SD_2_normal",
    "PA": "PA_2",
    "Smoking": "sm_1_not_smoker",
    "BMI": "bmi_2_normal",
    "Alcohol": "low",
    "Diabetes": "False",
    "Hypertension": "False",
    "Hyperchol_": "False",
    }

patient_2 = {
    "Age": "age_4_adult",
    "Sex": "M",
    "SD": "SD_2_normal",
    "PA": "PA_2",
    "Smoking": "sm_1_not_smoker",
    "BMI": "bmi_2_normal",
    "Alcohol": "low",
    "Diabetes": "True",
    "Hypertension": "True",
    "Hyperchol_": "False",
    }

# Comorbid: the PM's risk is above the citizen's.
patient_3 = {
    "Age": "age_4_adult",
    "Sex": "M",
    "SD": "SD_2_normal",
    "PA": "PA_1",
    "Smoking": "sm_3_ex_smoker",
    "BMI": "bmi_3_overweight",
    "Alcohol": "high",
    "Diabetes": "False",
    "Hypertension": "True",
    "Hyperchol_": "True",
    }

patient_4 = {
    "Age": "age_4_adult",
    "Sex": "M",
    "SD": "SD_2_normal",
    "PA": "PA_2",
    "Smoking": "sm_1_not_smoker",
    "BMI": "bmi_2_normal",
    "Alcohol": "low",
    "Diabetes": "False",
    "Hypertension": "False",
    "Hyperchol_": "False",
    }

patient_5 = {
    "Age": "age_5_old_adult",
    "Sex": "W",
    "SD": "SD_2_normal",
    "PA": "PA_2",
    "Smoking": "sm_1_not_smoker",
    "BMI": "bmi_2_normal",
    "Alcohol": "low",
    "Diabetes": "False",
    "Hypertension": "False",
    "Hyperchol_": "False",
    }


def patient(patient_num):
    patients = {1: patient_1, 2: patient_2, 3: patient_3, 4: patient_4, 5: patient_5}
    if patient_num not in patients:
        raise KeyError(f"patient {patient_num} not found")
    return patients[patient_num]
