#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:19:54 2022

@author: hillarywolff
"""
# Ch. 2 #3

# *sketch*

# test error: decreases as flexibility increases until a point where the model starts to overfit the data, the opposite of training error
# training error: decreases as flexibility increases since it will begin to overfit the model
# variance: stays relatively flat with a slight increase as flexibility increases until a point where variance beomes higher since the model gets less robust
# bayes error: a constant for this graph since it depends on the dataset
# bias: decreases as flexibility increases since it will overfit the model, leaving zero or little bias
# 
# #5
# advantages for very flexible model: low bias, can represent non-linear relationships
# disadvantages for very flexible model: overfitting of the dataset (low training error and high test error) and high variance
# when is flexibility preferred to less flexible: if we have a more complex, non-linear problem that won't be overfitted
# when is less flexibility preferred to very flexible: linear problems where we aren't concered with a model overfitting
# 


import pandas as pd
import os
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf

 
def read_data(fname):
    df = pd.read_csv(os.path.join(PATH, fname)) 
    return df

PATH = r"/Users/hillarywolff/Documents/GitHub/machine_learning/PS1/Boston"
fname = 'Boston.csv'


data = read_data(fname)
# Ch. 2 #10
# a. rows = 506
#    columns = 14
#    what do rows and columns represent? 
rows_col = data.info()

# b. 
cols = list(data.columns)
pair_plot = sns.pairplot(data[cols])


# describe findings:
# relationships: ageXnox (positive), disXnox (negative), LSTATXnox (weak positive)
# mdev X RM (positive), lstatXRM (negative), ageXlstat (weak positive), disXage (weak negative), mdev Xlstat (negative)

    
# c. are any predictors assiciated with per capita crime rate?
# weak association with age, dis, lstat, mdev

# D. do any of the census tracts of Boston appear to have particularly high crime rate?
# tax rate? pupil-teacher ratio?
high_crime = data['CRIM'].nlargest(n=5)
data['CRIM'].plot(use_index=True)
# highest crime are tracts: 380, 418, 405, 414
high_tax = data['TAX'].nlargest(n=5)
data['TAX'].plot(use_index=True)
# highest tax rate are tracts: 488, 489, 490, 491, 492
high_stu_teach = data['PTRATIO'].nlargest(n=5)
data['PTRATIO'].plot(use_index=True)
# 354, 355, 127, 128, 129


# E. how many of the census tracts in this data set bound the charles river
data['CHAS'].value_counts()
# 35 bound the charles river


# F. what is the median pupil-teacher ratio amount the towns in this data set
data['PTRATIO'].median()
# 19.05 


# G. which cencus tract has the lowest median value of owner-occupied homes?
# What are the values of the other predictors for that census tract, and how do 
# those values compare to the overall ranges for those predictors?

low_val = data['MDEV'].nsmallest(1)
# 398
tract = pd.DataFrame(data.iloc[low_val])

# CRIM        38.3518
# ZN           0.0000
# INDUS       18.1000
# CHAS         0.0000
# NOX          0.6930
# RM           5.4530
# AGE        100.0000
# DIS          1.4896
# RAD         24.0000
# TAX        666.0000
# PTRATIO     20.2000
# B          396.9000
# LSTAT       30.5900
# MDEV         5.0000

compare_df  = data[cols].describe()
tract_df = tract.describe()

# comparing this tract to the dataset as a whole,
# it seems this specific tract is equal to the minimum value for variables ZN, 
# CHAS, and MDEV. This tract is equal to the max value at AGE, B, and RAD. for
# the remainder of the variables, CRIM is far above the mean and median, but 
# less than the maximum. INDUS falls inbetween the mean and max,
# while LSTAT, PTRATIO, and NOX are more aligned with the mean. Finally,
# TAX is closer to the maximum value and DIS is closer to the minimum value. 


# H. in this data set, how many of the census tracts average more than seven rooms
# per dwelling? more than eight? 

dwell7_df = data[data['RM'] > 7]
dwell7_df.info()
# more than 7: 64

dwell8_df = data[data['RM'] > 8]
dwell8_df.info()
# more than 8: 13


# Chapter 3 #3
# A. 
## which answer is correct and why?
#
# iii because the interaction term for level is negative, so at the threshold of 
# GPA = 3.5, high school graduates can earn more than college graduates

## B.
# predict the salary of a college graduate with an IQ of 110 and a GPA of 4.0
#
# y = 50 + 20*gpa + .07*iq + 35*level + .01*gpa*iq + (-10)*gpa*level
# 50 + 20*4.0 + .07*110 + 35*1 + .01*4*110 + (-10)*4*1
# y = $137,100 starting salary

## C. 
# true or false: since the coefficient for the GPA/IQ interaction term is very small,
# there is very little evidence of an interaction effect
#
# False, the effect size is small, but the interaction term involves multiplying
# GPA and IQ, creating some large numbers. So this coefficient might be small, 
# the overall impact of this interaction term could be large and significant. 

# 15. 
## A. 

col_list = data.columns.difference(['CRIM'])
results_df = pd.DataFrame()
for col in col_list:
    results_summary = smf.ols(formula=f'CRIM ~ {col}', data=data).fit().summary()
    results_as_html = results_summary.tables[1].as_html()
    result_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    results_df = results_df.append(result_df)
results_df = results_df.reset_index()

# describe results

# AGE: significant, coef 0.107
# B: significant, negative relationship, small effect (-0.035)
# CHAS: insignificant, negative (-1.87)
# DIS: significant, negative (-1.54)
# INDUS: significant, positive 0.5
# LSTAT: significant, small effect,  0.54
# MDEV: significant, negative, small effect (-0.36)
# NOX: significant, large effect, 30.97
# PTRATIO: significant, 1.14
# RAD: significant, small effect, 0.6
# RM: signficiant, negative, relatively larger (-2.69)
# TAX: significant, small, 0.029
# ZN: significant, negative, small (-0.7)

## plots
# for col in col_list:
#     plt.figure(figsize=(10,8))
#     sns.regplot(x=f'{col}', y='CRIM', data=data)


## B. 
# multi variate model
predictors = ' + '.join(col_list)
result = smf.ols('CRIM ~ {}'.format(predictors), data=data).fit().summary()
results_as_html = result.tables[1].as_html()
multi_df = (pd.read_html(results_as_html, header=0, index_col=0)[0]).reset_index()

#         index     coef  std err      t  P>|t|  [0.025  0.975]
# 0   Intercept  17.4184    7.270  2.396  0.017   3.135  31.702
# 1         AGE   0.0020    0.018  0.112  0.911  -0.033   0.037
# 2           B  -0.0069    0.004 -1.857  0.064  -0.014   0.000
# 3        CHAS  -0.7414    1.186 -0.625  0.532  -3.071   1.588
# 4         DIS  -0.9950    0.283 -3.514  0.000  -1.551  -0.439
# 5       INDUS  -0.0616    0.084 -0.735  0.463  -0.226   0.103
# 6       LSTAT   0.1213    0.076  1.594  0.112  -0.028   0.271
# 7        MDEV  -0.1992    0.061 -3.276  0.001  -0.319  -0.080
# 8         NOX -10.6455    5.301 -2.008  0.045 -21.061  -0.230
# 9     PTRATIO  -0.2787    0.187 -1.488  0.137  -0.647   0.089
# 10        RAD   0.5888    0.088  6.656  0.000   0.415   0.763
# 11         RM   0.3811    0.616  0.619  0.536  -0.829   1.591
# 12        TAX  -0.0037    0.005 -0.723  0.470  -0.014   0.006
# 13         ZN   0.0449    0.019  2.386  0.017   0.008   0.082
 #
 
## For which predictors can we reject the null?
# DIS, MDEV, NOX, RAD, ZN


## C. 
# plot
unvar_reg = results_df[~results_df['index'].str.contains('Intercept')]
multi_reg = multi_df[~multi_df['index'].str.contains('Intercept')]

# plt.figure(figsize=(10,8))
# sns.regplot(x=unvar_reg['coef'], y=multi_reg['coef'])
# plt.xlabel("Unvariate", fontsize=15)
# plt.ylabel("Multi Variate", fontsize=15)
# plt.title('Unvariate vs. Multi Variate', fontsize=20)

# D. 

non_linear_df = pd.DataFrame()
for col in col_list:
    data_subset = data[['CRIM', f'{col}']]
    data_subset[f'{col}_squared'] = np.power(data_subset[f'{col}'], 2)
    data_subset[f'{col}_cubed'] = np.power(data_subset[f'{col}'], 3)
    data_summary = smf.ols(formula=f'CRIM ~ {col} + {col}_squared + {col}_cubed', data=data_subset).fit().summary()
    data_as_html = data_summary.tables[1].as_html()
    data_df = pd.read_html(data_as_html, header=0, index_col=0)[0]
    non_linear_df = non_linear_df.append(data_df)

non_linear_df = non_linear_df.reset_index()
non_linear_df = non_linear_df[~non_linear_df['index'].str.contains('Intercept')]

#               index          coef       std err       t  P>|t|        [0.025  \
# 1               AGE  2.743000e-01  1.860000e-01   1.471  0.142 -9.200000e-02   
# 2       AGE_squared -7.200000e-03  4.000000e-03  -1.987  0.047 -1.400000e-02   
# 3         AGE_cubed  5.737000e-05  2.110000e-05   2.719  0.007  1.590000e-05   
# 5                 B -8.450000e-02  5.600000e-02  -1.497  0.135 -1.960000e-01   
# 6         B_squared  2.000000e-04  0.000000e+00   0.760  0.447 -0.000000e+00   
# 7           B_cubed -2.895000e-07  4.380000e-07  -0.661  0.509 -1.150000e-06   
# 9              CHAS -6.238000e-01  5.020000e-01  -1.243  0.214 -1.610000e+00   
# 10     CHAS_squared -6.238000e-01  5.020000e-01  -1.243  0.214 -1.610000e+00   
# 11       CHAS_cubed -6.238000e-01  5.020000e-01  -1.243  0.214 -1.610000e+00   
# 13              DIS -1.551720e+01  1.737000e+00  -8.931  0.000 -1.893100e+01   
# 14      DIS_squared  2.447900e+00  3.470000e-01   7.061  0.000  1.767000e+00   
# 15        DIS_cubed -1.185000e-01  2.000000e-02  -5.802  0.000 -1.590000e-01   
# 17            INDUS -1.953300e+00  4.830000e-01  -4.047  0.000 -2.901000e+00   
# 18    INDUS_squared  2.504000e-01  3.900000e-02   6.361  0.000  1.730000e-01   
# 19      INDUS_cubed -6.900000e-03  1.000000e-03  -7.239  0.000 -9.000000e-03   
# 21            LSTAT -4.133000e-01  4.660000e-01  -0.887  0.375 -1.328000e+00   
# 22    LSTAT_squared  5.300000e-02  3.000000e-02   1.758  0.079 -6.000000e-03   
# 23      LSTAT_cubed -8.000000e-04  1.000000e-03  -1.423  0.155 -2.000000e-03   
# 25             MDEV -5.077400e+00  4.350000e-01 -11.668  0.000 -5.932000e+00   
# 26     MDEV_squared  1.551000e-01  1.700000e-02   8.995  0.000  1.210000e-01   
# 27       MDEV_cubed -1.500000e-03  0.000000e+00  -7.277  0.000 -2.000000e-03   
# 29              NOX -1.264102e+03  1.708600e+02  -7.398  0.000 -1.599791e+03   
# 30      NOX_squared  2.223227e+03  2.806590e+02   7.921  0.000  1.671816e+03   
# 31        NOX_cubed -1.232389e+03  1.496870e+02  -8.233  0.000 -1.526479e+03   
# 33          PTRATIO -8.180890e+01  2.764900e+01  -2.959  0.003 -1.361310e+02   
# 34  PTRATIO_squared  4.603900e+00  1.609000e+00   2.862  0.004  1.444000e+00   
# 35    PTRATIO_cubed -8.420000e-02  3.100000e-02  -2.724  0.007 -1.450000e-01   
# 37              RAD  5.122000e-01  1.047000e+00   0.489  0.625 -1.545000e+00   
# 38      RAD_squared -7.500000e-02  1.490000e-01  -0.504  0.615 -3.680000e-01   
# 39        RAD_cubed  3.200000e-03  5.000000e-03   0.699  0.485 -6.000000e-03   
# 41               RM -3.870400e+01  3.128400e+01  -1.237  0.217 -1.001670e+02   
# 42       RM_squared  4.465500e+00  5.005000e+00   0.892  0.373 -5.369000e+00   
# 43         RM_cubed -1.694000e-01  2.640000e-01  -0.643  0.521 -6.870000e-01   
# 45              TAX -1.524000e-01  9.600000e-02  -1.589  0.113 -3.410000e-01   
# 46      TAX_squared  4.000000e-04  0.000000e+00   1.476  0.141 -0.000000e+00   
# 47        TAX_cubed -2.193000e-07  1.890000e-07  -1.158  0.247 -5.910000e-07   
# 49               ZN -3.303000e-01  1.100000e-01  -3.008  0.003 -5.460000e-01   
# 50       ZN_squared  6.400000e-03  4.000000e-03   1.670  0.096 -1.000000e-03   
# 51         ZN_cubed -3.753000e-05  3.140000e-05  -1.196  0.232 -9.920000e-05   


#           0.975]  
# 1   6.410000e-01  
# 2  -8.250000e-05  
# 3   9.880000e-05  
# 5   2.600000e-02  
# 6   1.000000e-03  
# 7   5.700000e-07  
# 9   3.620000e-01  
# 10  3.620000e-01  
# 11  3.620000e-01  
# 13 -1.210400e+01  
# 14  3.129000e+00  
# 15 -7.800000e-02  
# 17 -1.005000e+00  
# 18  3.280000e-01  
# 19 -5.000000e-03  
# 21  5.020000e-01  
# 22  1.120000e-01  
# 23  0.000000e+00  
# 25 -4.222000e+00  
# 26  1.890000e-01  
# 27 -1.000000e-03  
# 29 -9.284140e+02  
# 30  2.774637e+03  
# 31 -9.383000e+02  
# 33 -2.748700e+01  
# 34  7.764000e+00  
# 35 -2.300000e-02  
# 37  2.569000e+00  
# 38  2.180000e-01  
# 39  1.200000e-02  
# 41  2.275900e+01  
# 42  1.430000e+01  
# 43  3.480000e-01  
# 45  3.600000e-02  
# 46  1.000000e-03  
# 47  1.530000e-07  
# 49 -1.150000e-01  
# 50  1.400000e-02  
# 51  2.410000e-05  





































