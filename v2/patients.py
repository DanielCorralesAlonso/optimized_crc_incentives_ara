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


# Patient 5 is specified by EXACTLY the seven covariates the screening decision
# observes (the informational predecessors of Screening in the influence
# diagram: PA, SD, Smoking, BMI, Alcohol, Sex, Age).  Nothing else is set, so
# the diagram marginalises over the rest -- which is what the policy actually
# does, and it makes this patient a well-defined cell of the assignment rather
# than an average over several.
#
# From models/screening_assignment.csv: the LARGEST cell that is assigned
# Stool_DNA after the operational cap (n = 1,617; p_crc = 0.003508).  It gets
# its unconstrained first choice -- capacity is not binding here -- so it is the
# complement of patient 1, whose Stool_DNA is taken away by the cap.
#   EU: Stool_DNA 0.146930 > FIT 0.144795 > No_screening 0.141984
patient_5 = {
    "Age": "age_5_old_adult",
    "Sex": "W",
    "SD": "SD_2_normal",
    "PA": "PA_2",
    "Smoking": "sm_1_not_smoker",
    "BMI": "bmi_2_normal",
    "Alcohol": "low",
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
    elif patient_num == 5:
        return patient_5
    else:
        return print("Patient number not found")