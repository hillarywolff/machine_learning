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
    
# c. are any predictors assiciated with per capita crime rate?

# D. do any of the census tracts of Boston appear to have particularly high crime rate?
# tax rate? pupil-teacher ratio?

# E. how many of the census tracts in this data set bound the charles river

# F. what is the median pupil-teacher ratio amount the towns in this data set

# G. which cencus tract has the lowest median value of owner-occupied homes?
# What are the values of the other predictors for that census tract, and how do 
# those values compare to the overall ranges for those predictors?

# H. in this data set, how many of the census tracts average more than seven rooms
# per dwelling? more than eight? 




# Chapter 3 #3
# which answer is correct and why?
# predict the salary of a college graduate with an IQ of 110 and a GPA of 4.0
# true or false: since the coefficient for the GPA/IQ interaction term is very small,
# there is very little evidence of an interaction effect

# 15. 













