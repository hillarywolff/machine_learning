#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:19:40 2022

@author: hillarywolff
"""
import numpy as np
import pandas as pd
from numpy import ravel

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import combinations
from sklearn import model_selection
from sklearn.preprocessing import scale 

def heat_map(cm):
    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, annot=True, 
                cmap='Blues')
    
    ax.set_title('Confusion Matrix with labels\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    return fig


PATH = r"/Users/hillarywolff/Documents/GitHub/machine_learning/PS4/"
# ch. 6, #9
# a. split the data set into a training set and a test set
college = pd.read_csv(PATH+'Data-College.csv')
college = college.drop('Unnamed: 0', axis=1)
college['Private'] = np.where(college['Private'].str.contains('Yes'), 1, 0)
X = college.drop('Apps', axis=1)
y = college[['Apps']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=37)

# b. fit a linear model using least squares on the training set and report the
# test error
model = LinearRegression().fit(X_train, y_train)
y_pred_ols = model.predict(X_test)
mean_squared_error(y_test, y_pred_ols)
# 1382455

# e. fit a PCR model on the training set, with M chosen by cross-validation.
# report the test error obtained, along with the value of M selected by X-val
pca = PCA()
X_reduced_train = pca.fit_transform(scale(X_train))
n = len(X_reduced_train)
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=37)
model = LinearRegression()
mse = []

score = -1*model_selection.cross_val_score(model, np.ones((n,1)), 
        y_train.values.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
mse.append(score)

for i in np.arange(2, 18):
    score = -1*model_selection.cross_val_score(model, X_reduced_train[:,:i], 
    y_train.values.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)

min(mse)
# M=17, MSE = 1649769.019

X_reduced_test = pca.transform(scale(X_test))[:,:17]

# Train regression model on training data 
model = LinearRegression()
model.fit(X_reduced_train[:,:17], y_train)

# Prediction with test data
y_pred = model.predict(X_reduced_test)
mean_squared_error(y_test, y_pred)
# MSE = 1370097.41


# f. fit a PLS model on the training set, with M chosen by cross-validation.
# report the test error obtained, along with the value of M selected by X-val
n = len(X_train)
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=37)
mse = []

for i in np.arange(2, 18):
    pls = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls, scale(X_train), y_train, 
    cv = kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(-score)

min(mse)
# M=9, MSE=1649557.56

pls = PLSRegression(n_components=9)
pls.fit(scale(X_train), y_train)

mean_squared_error(y_test, pls.predict(scale(X_test)))
# MSE = 1407576.20


# g. comment on the results obtained. how accurrately can we predict the number
# of college applications received? is there much difference among the test
# errors resulting from these five approaches?

# looking at the three tests run, we end up with MSE for OLS at 1382455, PCR at
# 1370097, and PLS at 1407576. This means that our PCR is the best predictor
# of college applications received since it has the lowest MSE. The three
# tests are not that far from each other in MSE result, specifically comparing
# OLS to PCR since they are very similar to each other. 


# ch. 8, #4
# a. sketch the tree corresponding to the partition of the predictor space illustrated
# in the left-handed panel of figure 8.14. the numbers inside the boxes indicate 
# the mean of Y within each region.



# b. Create a diagram similar to the left-handed panel of figure 8.14, using 
# the tree illustrated in the right=hand panel of the same figure. you should
# divide up the predictor space into the correct regions, and indicate the mean
# for each region




# modified question 9

# i. Create a training set containing a random sample of 800 observations, and a test
# set containing the remaining observations.

oj = pd.read_csv(PATH+'Data-OJ.csv')
oj['Store7'] = np.where(oj['Store7'].str.contains('Yes'), 1, 0)

train = oj.sample(n=800)
test = oj[~oj.isin(train)]

# ii. Fit a full, unpruned tree to the training data, with Purchase as the response and the
# other variables as predictors. What is the training error rate?

X_train = train.drop(['Purchase'], axis=1).dropna()
y_train = train[['Purchase']].dropna()

X_test = test.drop(['Purchase'], axis=1).dropna()
y_test = test[['Purchase']].dropna()

model = DecisionTreeClassifier(random_state=1, criterion='gini')
model.fit(X_train, y_train)


# iii. Create a plot of the tree The plot is a mess, isn’t it? For the purposes of this
# question, fit another tree with the max_depth parameter set to 3 in order to get an
# interpretable plot. How many terminal nodes does the tree have? Pick one of the
# terminal nodes, and interpret the information displayed.
#
plt.figure(figsize=(12,12))
tree.plot_tree(model, fontsize=10)
plt.show()

model = DecisionTreeClassifier(max_depth = 3, random_state=1, criterion='gini')
model.fit(X_train, y_train)

plt.figure(figsize=(12,12))
tree.plot_tree(model, fontsize=10)
plt.show()


# how many terminal nodes: 8
# pick one and interpret the results
# X[16] <=3.5
# gini = 0.435
# samples = 25
# value = [8, 17]


# iv. Predict the response on the test data, and produce a confusion matrix comparing
# the test labels to the predicted test labels. What is the test error rate?
#
y_predict = model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
heat_map(cm)

print('\nTest error rate: ', 1 - accuracy_score(y_test, y_predict))
# 0.255

# v. Determine the optimal tree size by tuning the ccp_alpha argument in scikit-
# learn’s DecisionTreeClassifier You can use GridSearchCV for this pur-
# pose.
#
tree_size = np.arange(2,20)
print(tree_size)
parameters = {'max_depth': tree_size}
cv_tree = GridSearchCV(model, parameters)
cv_tree.fit(X, y)


cv_scores = []
for mean_score in zip(cv_tree.cv_results_["mean_test_score"]):
    cv_scores.append(mean_score[0])
print(cv_scores)

# vi. Produce a plot with tree size on the x-axis and cross-validated classification error
# rate on the y-axis calculated using the method in the previous question. Which tree
# size corresponds to the lowest cross-validated classification error rate?
#
plt.figure(figsize=(10,8))
sns.lineplot(x=tree_size, y=cv_scores)
plt.xlabel("Tree size", fontsize= 16)


# vii. Produce a pruned tree corresponding to the optimal tree size obtained using cross-
# validation. If cross-validation does not lead to selection of a pruned tree, then
# create a pruned tree with five terminal nodes.
#



# viii. Compare the training error rates between the pruned and unpruned trees. Which is
# higher? Briefly explain.
#



# ix. Compare the test error rates between the pruned and unpruned trees. Which is
# higher? Briefly explain.






















































