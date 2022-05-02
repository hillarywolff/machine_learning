#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:06:08 2022

@author: hillarywolff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

#%matplotlib inline
plt.style.use('seaborn-white')

# 5. difference between LDA and QDA

# A. If the Bayes decision boundary is linear, do we expect LDA or QDA to perform
# better on the training set? on the test set?

# training set: QDA since the training set needs more flexibility
# test set: LDA 

# B. If the bayes decision boundary is non-linear, do we expect LDA or QDA to 
# perform better on the training set? on the test set?

# training set: QDA
# test set: QDA

# C. In general, as the sample size n increases, do we expect the test predction
# accuracy of QDA relative to LDA to improve, decline, or be unchanged?

# when sample size increases, we expect QDA to perform better since more
# flexibility with larger sample sizes can offset variance

# D. True or False: Even if the Bayes decision boundary for a given problem is 
# linear, we will probably achieve a superior test error rate using QDA rather 
# than LDA because QDA is flexible enough to model a linear decision boundary

# False, if the sample size is too small, QDA will give higher variance than LDA,
# giving it an inferior test error rate

# 6. Suppose we collect data for a group of students in a statistics class with
# variables X1=hours studied, X2=undergrad GPA, and Y=receive an A. We fit a 
# logistic regression and produce estimated coefficient B0=-6, B1=0.05, B2=1

# A. estimate the probability that a student who studies for 40 hours with a GPA 
# of 3.5 gets an A in the class
# P(x) = e^(-6+(0.05*40)+(1*3.5))/ 1+(" ")

# where (np.exp(-6+(.05*40)+3.5)) = -0.5
# np.exp(-.5)/1+(np.exp(-.5))
# 37.7% chance

# B. how many hours would the student in part A need to study to have a 50% 
# chance of getting an A in the class?

# (np.exp(-6+(.05X)+3.5))/1+(np.exp(-6+(.05X)+3.5)) = .5
# np.exp(-6+(.05*40)+3.5) = 1
# log(np.exp(-6+(.05*40)+3.5)) = log(1)
# -6+.05X+3.5 = 0
# X = 50
# need to study 50 hours to have a 50% chance of an A


# 7. suppose that we wish to predict whether a given stock will issue a dividend
# this year (yes or no) based on X, last year's percent profit. The mean value 
# of X for companies that issued a dividend was X=10, while the mean for companies
# that did NOT issue dividends was X=0. in addition, the variance of X for these 
# two classes of companies was sig_sq=36. Finally, 80% of companies issued dividends.
# Assuming that X follows a normal distribution, predict the probability that a 
# company will issue a divided this year given that its percentage profit was X=4
# last year. 
# ******************

discrim_1 = np.log(.8) - ((10^2)/(2*36)) + ((10/36)*4)

discrim_2 = np.log(.2) - ((0^2)/(2*36)) + ((0/36)*4)

prediction = (np.exp(discrim_1))/((np.exp(discrim_1))+(np.exp(discrim_2)))

# Pyes(X=4) = 91.7%



# 14.
# A. 
PATH = r"/Users/hillarywolff/Documents/GitHub/machine_learning/PS2/"
df = pd.read_csv(PATH + 'Data-Auto.csv')

median_mpg = df['mpg'].median() # 22.75 is median
df['mpg01'] = np.where((df['mpg'] > median_mpg), 1, 0)

# B. 
cols = list(df.columns)
pair_plot = sns.pairplot(df[cols])

# displacement, horsepower, weight are predictors of mpg01

# C. 
lda_model = LinearDiscriminantAnalysis(solver='svd')

X = df[['displacement', 'horsepower', 'weight']]
y = df['mpg01']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

lda_model.fit(X_train, y_train)
y_pred = lda_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
# accuracy 90%, test error 10%

# E. 
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
y_pred = qda_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
# accuracy 88.9%, test error 12%

# F. 
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred_log = logisticRegr.predict_proba(X_test)[:,1]

y_pred_log_bin = np.where(y_pred_log>0.5, 1, 0)
print('OLS Accuracy:', accuracy_score(y_test, y_pred_log_bin))
# accuracy 87%, test error 13%

# G
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print('Naive Bayes Accuracy:', accuracy_score(y_test, y_pred))
# accuracy 88.9%, test error 12%


# Chapter 5
# 5. estimating test error of logistic regression model using validation set approach

df = pd.read_csv(PATH + 'Data-Default.csv')
df['default'] = df['default'].eq('Yes').mul(1)

X = df[['income', 'balance']]
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

logisticRegr.fit(X_train, y_train)


for i, seed in enumerate([16, 78, 244]):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    y_pred = lda_model.predict(X_val)
    validation_set_error = 1 - accuracy_score(y_val, y_pred)
    
    print("Validation set error with set", i+1 ," is: " , validation_set_error)
    
# Validation set error with set 1  is:  0.026000000000000023
# Validation set error with set 2  is:  0.02733333333333332
# Validation set error with set 3  is:  0.02833333333333332










