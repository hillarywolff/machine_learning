#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:06:08 2022

@author: hillarywolff
"""

import numpy as np

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
np.exp(-.5)/1+(np.exp(-.5))
# 1.21?
# 37.7% chance


# (np.exp(-6+(.05X)+3.5))/1+(np.exp(-6+(.05X)+3.5)) = .5
# np.exp(-6+(.05*40)+3.5) = 1
# log(np.exp(-6+(.05*40)+3.5)) = log(1)
# -6+.05X+3.5 = 0
# X = 50
# need to study 50 hours to have a 50% chance of an A

# B. how many hours would the student in part A need to study to have a 50% 
# chance of getting an A in the class?
# *****

# 7. suppose that we wish to predict whether a given stock will issue a dividend
# this year (yes or no) based on X, last year's percent profit. The mean value 
# of X for companies that issued a dividend was X=10, while the mean for companies
# that did NOT issue dividends was X=0. in addition, the variance of X for these 
# two classes of companies was sig_sq=36. Finally, 80% of companies issued dividends.
# Assuming that X follows a normal distribution, predict the probability that a 
# company will issue a divided this year given that its percentage profit was X=4
# last year. 


# Pyes(X=4) = .8*np.exp(-(1/72)(4-10)^2)/ .8*np.exp(-(1/72)(4-10)^2) + .2*np.exp(-(1/72)(4-0)^2)
# Pyes(X=4) = 75.2%














