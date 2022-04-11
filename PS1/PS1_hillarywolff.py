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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf



# Variables in order:
#  CRIM     per capita crime rate by town
#  ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  INDUS    proportion of non-retail business acres per town
#  CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  NOX      nitric oxides concentration (parts per 10 million)
#  RM       average number of rooms per dwelling
#  AGE      proportion of owner-occupied units built prior to 1940
#  DIS      weighted distances to five Boston employment centres
#  RAD      index of accessibility to radial highways
#  TAX      full-value property-tax rate per $10,000
#  PTRATIO  pupil-teacher ratio by town
#  B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#  LSTAT    % lower status of the population
#  MEDV     Median value of owner-occupied homes in $1000's

 
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

# comparing this tract to the min, max, and mean of the dataset as a whole,
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

dwell8_df = data[data['RM'] > 8]
dwell8_df.info()

# more than 7: 64
# more than 8: 13



# Chapter 3 #3
# which answer is correct and why?
# iii because the interaction term for level is negative, so at the threshold of 
# GPA = 3.5, high school graduates can earn more than college graduates

# predict the salary of a college graduate with an IQ of 110 and a GPA of 4.0
# y = 50 + 20*gpa + .07*iq + 35*level + .01*gpa*iq + (-10)*gpa*level
50 + 20*4.0 + .07*110 + 35*1 + .01*4*110 + (-10)*4*1
# y = $137,100 starting salary

# true or false: since the coefficient for the GPA/IQ interaction term is very small,
# there is very little evidence of an interaction effect
# True because any (realistic) IQ level and GPA 

# 15. 
col_list = data.columns.difference(['name', 'CRIM'])
results_df = pd.DataFrame()
for col in col_list:
    results_summary = smf.ols(formula=f'CRIM ~ {col}', data=data).fit().summary()
    results_as_html = results_summary.tables[1].as_html()
    result_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    results_df = results_df.append(result_df)
results_df = results_df.reset_index()


# a. describe results

# AGE: significant, coef 0.107
# B: significant, negative relationship (-0.035)
# CHAS: insignificant, negative (-1.87)
# DIS: significant, negative (-1.54)
# INDUS: significant, positive 0.5
# LSTAT: significant, 0.54
# MDEV: significant, negative (-0.36)
# NOX: significant, large, 30.97
# PTRATIO: significant, 1.14
# RAD: significant, 0.6
# RM: signficiant, negative (-2.69)
# TAX: significant, small, 0.029
# ZN: significant, small (-0.7)


## plots
for col in col_list:
    plt.figure(figsize=(10,8))
    sns.regplot(x=f'{col}', y='CRIM', data=data)



# multi variate model
predictors = ' + '.join(data.columns.difference(['name', 'CRIM']))
result = smf.ols('CRIM ~ {}'.format(predictors), data=data).fit().summary()
results_as_html = result.tables[1].as_html()
multi_df = (pd.read_html(results_as_html, header=0, index_col=0)[0]).reset_index()
print(multi_df)

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



# plot
unvar_reg = results_df[~results_df['index'].str.contains('Intercept')]
multi_reg = multi_df[~multi_df['index'].str.contains('Intercept')]

plt.figure(figsize=(10,8))
sns.regplot(x=unvar_reg['coef'], y=multi_reg['coef'])
plt.xlabel("Unvariate", fontsize=15)
plt.ylabel("Multi Variate", fontsize=15)
plt.title('Unvariate vs. Multi Variate', fontsize=20)
# b. describe results

# lopp for plotting? 


























































