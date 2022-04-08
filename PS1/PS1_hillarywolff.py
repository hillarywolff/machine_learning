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
tract = pd.DataFrame(data.iloc[398])

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

col_max = data[cols].max()
col_max = col_max.rename('MAX', inplace=True)
col_min = data[cols].min()
col_min = col_min.rename('MIN', inplace=True)
col_mean = data[cols].mean()
col_mean = col_mean.rename('MEAN', inplace=True)
col_med = data[cols].median()
col_med = col_med.rename('MEDIAN', inplace=True)

min_max = pd.concat([col_min, col_max, col_mean, col_med, tract], axis=1)



# comparing this tract to the min, max, and mean of the dataset as a whole,
# it seems this specific tract is equal to the minimum value for variables ZN, 
# CHAS, and MDEV. This tract is equal to the max value at AGE, B, and RAD. for
# the remainder of the variables, CRIM is far above the mean and median, but 
# less than the maximum. INDUS falls inbetween the mean and max,
# while LSTAT, PTRATIO, and NOX are more aligned with the mean. Finally,
# TAX is closer to the maximum value and DIS is closer to the minimum value. 


# H. in this data set, how many of the census tracts average more than seven rooms
# per dwelling? more than eight? 

dwell_df = data[data['RM'] > 7]
dwell_df.info()

dwell_df = data[data['RM'] > 8]
dwell_df.info()

# more than 7: 64
# more than 8: 13



# Chapter 3 #3
# which answer is correct and why?
# predict the salary of a college graduate with an IQ of 110 and a GPA of 4.0
# true or false: since the coefficient for the GPA/IQ interaction term is very small,
# there is very little evidence of an interaction effect

# 15. 













