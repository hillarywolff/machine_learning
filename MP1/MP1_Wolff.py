#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:13:50 2022

@author: hillarywolff with Jason Winik
"""

import pandas as pd
import os
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns

PATH = r"/Users/hillarywolff/Documents/GitHub/machine_learning/MP1/"
 


def read_data(fname):
    df = pd.read_csv(os.path.join(PATH, fname)) 
    return df

def interaction_terms(terms):
    for term in terms:
        df[f'EDUx{term}'] = df['EDUCDC']*df[f'{term}']


## Q3
# 1.
fname = 'usa_00001.csv'

df = read_data(fname)


# 2. 
# a. 
crosswalk_fname = 'PPHA_30545_MP01-Crosswalk.csv'

crosswalk = read_data(crosswalk_fname)
crosswalk = crosswalk.set_index('educd').T
crosswalk = crosswalk.to_dict('list')


df['EDUCDC'] = df['EDUCD']
df = df.replace({'EDUCDC':crosswalk})

# b.
df['hsdip'] = np.where((df['EDUC']==62)| 
                       (df['EDUCD']==63)|
                       (df['EDUCD']==64), 1, 0)

df['coldip'] = np.where((df['EDUCD']==101)|
                        (df['EDUCD']==114)|
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

# c. 
df['EDUxHSDIP'] = df['EDUCDC']*df['hsdip']
df['EDUxCOLDIP'] = df['EDUCDC']*df['coldip']

# d.
df['age_sq'] = np.power(df['AGE'], 2)
df['INCWAGE_log'] = np.log(df['INCWAGE'])


# Q4 pt1:
describe_cols = ['EDUxHSDIP', 'EDUxCOLDIP','vet', 'female', 'Black',
            'White', 'hispanic', 'coldip', 'hsdip', 'EDUCDC', 'age_sq', 'AGE', 
            'INCWAGE','INCWAGE_log', 'NCHILD', 'YEAR']

describe_df=pd.DataFrame()
for col in describe_cols:
    describe_df.append(df[col].describe())


# Q4 pt2:
ax = sns.regplot(x='EDUCDC', y='INCWAGE_log', data=df, scatter_kws={'alpha':0.3})
ax.set(xlabel=('Years of Education'), ylabel=('ln(wage) ($10,000)'), title=('Years of education vs. log wages'))

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
# With a significance threshold of 0.1, we can rule in favor of the alternative
# hypothesis since we have statistically significant values for all but four 
# regressors
#
# C. what is the return to an additional year of education? is it significant?
# the coefficient indicates that an additional year of education will increase
# log wages by 10.3 percent. this is statistically significant with a p-value of 0.00
# and practically significant since a 10% increase is fairly large

100*(np.exp(0.098)-1)

# D. at what age does the model predict an individual will achieve the highest wage?
# y = ax2 + bx + c
# y = 2(-.0017)X + (.1607) 
# y = (-.0034)X + (.1607)
# 0 = (-.0034)X + .1607
# -.1607= -.0034X
# X= 47
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
# Null: all betas for race = 0
# alternate: all betas for race are not 0

results = smf.ols('INCWAGE_log ~ hispanic + White + Black', data=df).fit().summary()

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:            INCWAGE_log   R-squared:                       0.016
# Model:                            OLS   Adj. R-squared:                  0.016
# Method:                 Least Squares   F-statistic:                     42.69
# Date:                Mon, 18 Apr 2022   Prob (F-statistic):           2.32e-27
# Time:                        18:28:34   Log-Likelihood:                -11832.
# No. Observations:                7928   AIC:                         2.367e+04
# Df Residuals:                    7924   BIC:                         2.370e+04
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     10.5528      0.032    333.612      0.000      10.491      10.615
# hispanic      -0.2814      0.040     -7.057      0.000      -0.360      -0.203
# White          0.0539      0.034      1.595      0.111      -0.012       0.120
# Black         -0.2823      0.052     -5.420      0.000      -0.384      -0.180
# ==============================================================================
# Omnibus:                     1777.241   Durbin-Watson:                   1.759
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5323.596
# Skew:                          -1.158   Prob(JB):                         0.00
# Kurtosis:                       6.279   Cond. No.                         6.79
# ==============================================================================
#
#
# since the F statistic is high enough that we can rule
# in favor of the alternate hypothesis. 


## Q4 pt4:
# graph ln wage and education with three linear lines for no diploma, high school
# diploma, and college degree
degree_level = [df['coldip'] == 1, df['hsdip']==1, df['hsdip']==0]
degree_name = ['College Diploma', 'HS Diploma', 'No HS Diploma']
df['degree'] = np.select(degree_level, degree_name)

ax = sns.lmplot(x='EDUCDC', y='INCWAGE_log', hue='degree', data=df);
ax.set(xlabel='years of education', ylabel='ln(wage) in $10,0000', 
       title='log wages by education level')


## Q5.
degree_numeric = [3, 2, 1]
degree_new = [df['degree'] == 'College Diploma', df['degree']=='HS Diploma', df['degree']=='No HS Diploma']
df['degree_level'] = np.select(degree_new, degree_numeric)

reg2_cols = ['EDUCDC', 'female', 'AGE', 'age_sq', 'White', 'Black', 
            'hispanic', 'married', 'NCHILD', 'vet', 'C(degree)']
reg2_cols = ' + '.join(reg2_cols)

results = smf.ols('INCWAGE_log ~ {}'.format(reg2_cols), data=df).fit().summary()

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:            INCWAGE_log   R-squared:                       0.312
# Model:                            OLS   Adj. R-squared:                  0.311
# Method:                 Least Squares   F-statistic:                     326.3
# Date:                Mon, 18 Apr 2022   Prob (F-statistic):               0.00
# Time:                        18:41:15   Log-Likelihood:                -10413.
# No. Observations:                7928   AIC:                         2.085e+04
# Df Residuals:                    7916   BIC:                         2.093e+04
# Df Model:                          11                                         
# Covariance Type:            nonrobust                                         
# ================================================================================
#                    coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------
# Intercept        5.9146      0.123     47.973      0.000       5.673       6.156
# EDUCDC           0.0640      0.004     14.246      0.000       0.055       0.073
# female          -0.3931      0.021    -19.081      0.000      -0.433      -0.353
# AGE              0.1552      0.006     25.838      0.000       0.143       0.167
# age_sq          -0.0016   7.08e-05    -22.822      0.000      -0.002      -0.001
# White            0.0513      0.028      1.812      0.070      -0.004       0.107
# Black           -0.0830      0.044     -1.886      0.059      -0.169       0.003
# hispanic        -0.0197      0.034     -0.576      0.564      -0.087       0.047
# married          0.1975      0.024      8.268      0.000       0.151       0.244
# NCHILD          -0.0007      0.011     -0.068      0.946      -0.022       0.020
# vet              0.0012      0.051      0.024      0.981      -0.099       0.101
# degree_level     0.1863      0.015     12.454      0.000       0.157       0.216
# ==============================================================================
# Omnibus:                     2367.652   Durbin-Watson:                   1.865
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            11092.715
# Skew:                          -1.377   Prob(JB):                         0.00
# Kurtosis:                       8.099   Cond. No.                     2.71e+04
# ==============================================================================


# I believe this is a model that can give the President the best idea of returns
# to education by degree level, as I have categorized our three degree levels 
# and assigned them a numeric value so they could be best evaluated in the regression.
# Intuitively, it makes sense that we would have no high school diploma coded to
# 1, with high school at 2 and college at 3. The 'one unit increase' line for our
# regression interpretation will refer to the subject escalating from one level 
# to another, accompanied by a larger salary.

## Q6:
# a. 22 year old, female, 0 for race, 0 married, 0 children, 0 vet, has college dip

# college:
# 5.9 + 0.06*16 -.393 + .155*22 -.0016*(22*22) + .186*3
# log(y) = 9.6
# y=$15,687.2

# high school:
# 5.9 + 0.06*12 -.393 + .155*22 -.0016*(22*22) + .186*3
# log(y) = 9.4
# y=$12,340 


# b. 
# Based on intuition,I can guess that those with college degrees will have higher
# predicted wages by about 20% as each degree level increases. So it will be 
# a 40% higher salary for someone with a college degree compared to someone
# without a high school diploma
100*(np.exp(.186)-1)



#c. 
# I would advise the president to expand access to college education since there
# are major benefits for having a college education compared to high school
# or less. 

## Q7
# To improve this model, I would want the following variables available:
# work sector (which could be very expansive, but categrocial), state/city of
# residence, and recent work history to determine if a person had a break in 
# their career. There are also other terms that could be added to this model,
# but I think those would help increase the R^2 without overfitting.

















