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
    }


patient_3 = {
    "Age": "age_4_adult", 
    "Sex": "M",
    "SD": "SD_2_normal",
    "PA": "PA_1",
    "Smoking": "sm_3_ex_smoker",
    "BMI": "bmi_3_overweight",
    "Alcohol": "high",
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
    }


def patient(patient_num):
    if patient_num == 1:
        return patient_1
    elif patient_num == 2:
        return patient_2
    elif patient_num == 3:
        return patient_3
    elif patient_num == 4:
        return patient_4
    else:
        return print("Patient number not found")