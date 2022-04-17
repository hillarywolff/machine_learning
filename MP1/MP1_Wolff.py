#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:13:50 2022

@author: hillarywolff
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

PATH = r"/Users/hillarywolff/Documents/GitHub/machine_learning/MP1/"
 


def read_data(fname):
    df = pd.read_csv(os.path.join(PATH, fname)) 
    return df

def interaction_terms(terms):
    for term in terms:
        df[f'EDUx{term}'] = df['EDUCDC']*df[f'{term}']

fname = 'usa_00001.csv'

df = read_data(fname)

crosswalk_fname = 'PPHA_30545_MP01-Crosswalk.csv'

crosswalk = read_data(crosswalk_fname)
crosswalk = crosswalk.set_index('educd').T
crosswalk = crosswalk.to_dict('list')


df['EDUCDC'] = df['EDUCD']
df = df.replace({'EDUCDC':crosswalk})

df['hsdip'] = np.where((df['EDUC']==62)| 
                       (df['EDUCD']==63)|
                       (df['EDUCD']==64), 1, 0)

df['coldip'] = np.where((df['EDUCD']==101)|
                        (df['EDUCD']==114)|
                        (df['EDUCD']==115)|
                        (df['EDUCD']==115)|
                        (df['EDUCD']==116), 1, 0)

df['White'] = np.where(df['RACE'] == 1, 1, 0)

df['Black'] = np.where(df['RACE'] == 2, 1, 0)

df['hispanic'] = np.where((df['HISPAN']==1)|
                          (df['HISPAN']==2)|
                          (df['HISPAN']==3)|
                          (df['HISPAN']==4), 1, 0)

df['married']= np.where((df['MARST']==1)|
                        (df['MARST']==2), 1, 0)

df['female']= np.where((df['SEX']==2), 1, 0)

df['vet'] =np.where((df['VETSTAT']==2), 1, 0)


df['EDUxHISPAN'] = df['EDUCDC']*df['hispanic']
df['EDUxHSDIP'] = df['EDUCDC']*df['hsdip']
df['EDUxCOLDIP'] = df['EDUCDC']*df['coldip']
df['EDUxWHITE'] = df['EDUCDC']*df['White']
df['EDUxBLACK'] = df['EDUCDC']*df['Black']
df['EDUxMARRIED']= df['EDUCDC']*df['married']
df['EDUxFEMALE'] = df['EDUCDC']*df['female']
df['EDUxVET'] = df['EDUCDC']*df['vet']

df['age_sq'] = np.power(df['AGE'], 2)
df['INCWAGE_log'] = np.log(df['INCWAGE'])

describe_cols = ['EDUxHISPAN', 'EDUxHSDIP', 'EDUxCOLDIP', 'EDUxWHITE', 'EDUxBLACK',
            'EDUxMARRIED', 'EDUxFEMALE', 'EDUxVET', 'vet', 'female', 'Black',
            'White', 'hispanic', 'coldip', 'hsdip', 'EDUCDC', 'age_sq', 'AGE', 
            'INCWAGE','INCWAGE_log', 'NCHILD', 'YEAR']

# Q4 pt1:
describe_df=pd.DataFrame()
for col in describe_cols:
    describe_df.append(df[col].describe())


# Q4 pt2:
plt.scatter(df['EDUCDC'], df['INCWAGE_log'])
plt.title('ln(wage) vs. Educational Attainment')
plt.ylabel('ln(wage) ($10,000)')
plt.xlabel('Education level')


# Q4 pt3:
df = df.drop(np.where(df['INCWAGE_log']<0)[0])
reg_cols = ['EDUCDC', 'female', 'AGE', 'age_sq', 'White', 'Black', 
            'hispanic', 'married', 'NCHILD', 'vet']
reg_cols = ' + '.join(reg_cols)

result = smf.ols('INCWAGE_log ~ {}'.format(reg_cols), data=df).fit().summary()

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:            INCWAGE_log   R-squared:                       0.298
# Model:                            OLS   Adj. R-squared:                  0.298
# Method:                 Least Squares   F-statistic:                     336.9
# Date:                Sun, 17 Apr 2022   Prob (F-statistic):               0.00
# Time:                        15:18:08   Log-Likelihood:                -10490.
# No. Observations:                7928   AIC:                         2.100e+04
# Df Residuals:                    7917   BIC:                         2.108e+04
# Df Model:                          10                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      5.7064      0.123     46.267      0.000       5.465       5.948
# EDUCDC         0.0980      0.004     27.174      0.000       0.091       0.105
# female        -0.3991      0.021    -19.193      0.000      -0.440      -0.358
# AGE            0.1607      0.006     26.567      0.000       0.149       0.173
# age_sq        -0.0017   7.13e-05    -23.489      0.000      -0.002      -0.002
# White          0.0395      0.029      1.383      0.167      -0.016       0.096
# Black         -0.1178      0.044     -2.657      0.008      -0.205      -0.031
# hispanic      -0.0398      0.034     -1.158      0.247      -0.107       0.028
# married        0.2057      0.024      8.528      0.000       0.158       0.253
# NCHILD        -0.0029      0.011     -0.271      0.786      -0.024       0.018
# vet           -0.0200      0.052     -0.387      0.699      -0.121       0.081
# ==============================================================================
# Omnibus:                     2297.997   Durbin-Watson:                   1.855
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10723.509
# Skew:                          -1.334   Prob(JB):                         0.00
# Kurtosis:                       8.034   Cond. No.                     2.69e+04
# ==============================================================================


# report results
# A. what fraction of the variation in log wages does the model explain?
# 29.8% of the variation in log wages is explained by our model
# 
# B. test the hypothesis that all betas = 0, and that no betas=0 at an alpha of 0.10
# 
#
# C. what is the return to an additional year of education? is it significant?
# the coefficient indicates that an additional year of education will increase
# log wages by 9 units. this is statistically significant with a p-value of 0.00
#
# D. at what age does the model predict an individual will achieve the highest wage?
#
#
# E. does the model predict that men or women will have higher wages? why?
# the model predicts higher wages for men since the coefficient on the female 
# variable is negative indicating a decrease in wages 
#
# F. Interpret the coefficients on the white, black, and hispanic variables
# the coefficient on the variable for white indicated an increase in wages, while
# the coefficients on Black and Hispanic variables are negative. However, only the
# coefficient for Black is statistically significant.
#
# G. test the hypothesis that race has no effect on wages. state the null
# and alternative hypothesis 


## Q4 pt4:
# graph ln wage and education with three linear lines for no diploma, high school
# diploma, and college degree
















































