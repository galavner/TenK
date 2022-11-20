import os

import numpy as np
import pandas as pd

# Should be changes to something more elegant, unnecessary
feature_inx = {
    'gender': 0,
    'age': 1,
    'chol': 2,
    'hdl': 3,
    'sbp': 4,
    'smoking': 5,
    'diabetes': 6
}

# The risk score function
def _calculate_framingham_risk_score(row):

    # Load the different parameters from the 'row' series, should be changed to something more elegant
    gender = row[feature_inx['gender']]
    age = row[feature_inx['age']]
    total_cholesterol = row[feature_inx['chol']]
    hdl = row[feature_inx['hdl']]
    sbp = row[feature_inx['sbp']]
    smoking = row[feature_inx['smoking']]
    # diabetes = row[feature_inx['diabetes']]

    points = 0
    percent_risk = 0

    # Males:
    if gender == 1:
        #calculate age points
        if  age <= 34:
            points+=0
        if  35 <= age <= 39:
            points+=2
        if  40 <= age <= 44:
            points+=5
        if  45 <= age <= 49:
            points+=6
        if  50 <= age <= 54:
            points+=8
        if  55 <= age <= 59:
            points+=10
        if  60 <= age <= 64:
            points+=11
        if  65 <= age <= 69:
            points+=12
        if  70 <= age <= 74:
            points+=14
        if  75 <= age:
            points+=15

        #calculate total cholesterol points:
        if total_cholesterol < 160:
            points += 0
        if 160 <= total_cholesterol <= 199:
            points += 1
        if 200 <= total_cholesterol <= 239:
            points += 2
        if 240 <= total_cholesterol <= 279:
            points += 3
        if total_cholesterol > 280:
            points += 4

        #calculate hdl points:
        if hdl > 60:
            points-=2
        if 50 <= hdl <= 59:
            points-=1
        if 45 <= hdl <= 49:
            points+=0
        if 35 <= hdl <= 44:
            points+=1
        if hdl < 35:
            points+=2

        #calculate sbp points:
        # Todo: insert 'if not blood_pressure_med_treatment:' before the following
        #  if we find treated people, and else another point calculations
        if sbp < 120:
            points += 0
        if 120 <= sbp <= 129:
            points += 2
        if 130 <= sbp <= 139:
            points += 3
        if 140 <= sbp <= 159:
            points += 4
        if sbp >= 160:
            points += 5

        # calculate smoking points:
        if smoking:
            points += 4

        # calculate diabetes points:
        # if diabetes:
        #     points += 3


        # calulate % risk for males
        # it's supposed to be <1%, 0 is just a replacement
        if points <= -3:
            percent_risk = 0
        elif points == -2:
            percent_risk = 1.1

        elif points == -1:
            percent_risk = 1.4

        elif points == 0:
            percent_risk = 1.6

        elif points == 1:
            percent_risk = 1.9

        elif points == 2:
            percent_risk = 2.3

        elif points == 3:
            percent_risk = 2.8

        elif points == 4:
            percent_risk = 3.3

        elif points == 5:
            percent_risk = 3.9

        elif points == 6:
            percent_risk = 4.7

        elif points == 7:
            percent_risk = 5.6

        elif points == 8:
            percent_risk = 6.7

        elif points == 9:
            percent_risk = 7.9

        elif points == 10:
            percent_risk = 9.4

        elif points == 11:
            percent_risk = 11.2

        elif points == 12:
            percent_risk = 13.2

        elif points == 13:
            percent_risk = 15.6

        elif points == 14:
            percent_risk = 18.4

        elif points == 15:
            percent_risk = 21.6

        elif points == 16:
            percent_risk = 25.3

        elif points == 17:
            percent_risk = 29.4

        # it's supposed to be >30%, 35 is just a replacement
        elif points >= 18:
            percent_risk = 35







    # Females
    else:
        # calculate age points
        if age <= 34:
            points += 0
        if 35 <= age <= 39:
            points += 2
        if 40 <= age <= 44:
            points += 4
        if 45 <= age <= 49:
            points += 5
        if 50 <= age <= 54:
            points += 7
        if 55 <= age <= 59:
            points += 8
        if 60 <= age <= 64:
            points += 9
        if 65 <= age <= 69:
            points += 10
        if 70 <= age <= 74:
            points += 11
        if 75 <= age:
            points += 12

        # calculate total cholesterol points:
        if total_cholesterol < 160:
            points += 0
        if 160 <= total_cholesterol <= 199:
            points += 1
        if 200 <= total_cholesterol <= 239:
            points += 3
        if 240 <= total_cholesterol <= 279:
            points += 4
        if total_cholesterol > 280:
            points += 5

        # calculate hdl points:
        if hdl > 60:
            points -= 2
        if 50 <= hdl <= 59:
            points -= 1
        if 45 <= hdl <= 49:
            points += 0
        if 35 <= hdl <= 44:
            points += 1
        if hdl < 35:
            points += 2

        # calculate sbp points:
        # Todo: insert 'if not blood_pressure_med_treatment:' before the following
        #  if we find treated people, and else another point calculations. Currently not treated
        if sbp < 120:
            points -= 3
        if 120 <= sbp <= 129:
            points += 0
        if 130 <= sbp <= 139:
            points += 1
        if 140 <= sbp <= 159:
            points += 2
        if sbp >= 160:
            points += 5

        # calculate smoking points:
        if smoking:
            points += 3

        # calculate diabetes points:
        # if diabetes:
        #     points += 4


        # calulate % risk for females
        # it's supposed to be <1%, 0 is just a replacement
        if points <= -2:
            percent_risk = 0

        elif points == -1:
            percent_risk = 1.0

        elif points == 0:
            percent_risk = 1.2

        elif points == 1:
            percent_risk = 1.5

        elif points == 2:
            percent_risk = 1.7

        elif points == 3:
            percent_risk = 2.0

        elif points == 4:
            percent_risk = 2.4

        elif points == 5:
            percent_risk = 2.7

        elif points == 6:
            percent_risk = 3.3

        elif points == 7:
            percent_risk = 3.9

        elif points == 8:
            percent_risk = 4.5

        elif points == 9:
            percent_risk = 5.3

        elif points == 10:
            percent_risk = 6.3

        elif points == 11:
            percent_risk = 7.3

        elif points == 12:
            percent_risk = 8.6

        elif points == 13:
            percent_risk = 10.0

        elif points == 14:
            percent_risk = 11.7

        elif points == 15:
            percent_risk = 13.7

        elif points == 16:
            percent_risk = 15.9

        elif points == 17:
            percent_risk = 18.5

        elif points == 18:
            percent_risk = 21.5

        elif points == 19:
            percent_risk = 24.8

        elif points == 20:
            percent_risk = 28.5

        # it's supposed to be >30%, 35 is just a replacement
        elif points >= 21:
            percent_risk = 35

    return points, percent_risk
