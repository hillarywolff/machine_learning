#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:51:32 2022

@author: hillarywolff
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import svm

PATH = r"/Users/hillarywolff/Documents/GitHub/machine_learning/MP2/"
df = pd.read_csv(PATH + 'audit.csv')



# 1. 
# check for the variance of each predictor
cols = df.columns
for col in cols:
    print(f'{col} variance: ', df[col].var)

# Sector score: 3.89
# PARA A: 4.18
# PARA B: 2.5
# RIsk_B: .5
# money value: 3.38
# risk_D: .676
# score: 2.4
# inherent risk variance: 8.574
# audit_risk:1.714
# risk: 1

# is wildly uneven variance across the attributes is a cause for concern?

# 2. 
# are there circumstances under which you would not be concerned with the 
# wildly uneven variance?

# 3. 
# which attributes do you suspect might interact as far as a firms ex ante 
# probability of tax evasion is concered? 

# without creating an interaction variable, what advantage does KNN have over 
# LPM if the interaction is important?


##########################
# data analysis

# 4. use the first half of the data to train LPM. apply model to the second 
# half of the data to predict the probability a firm cheated. 

df = df.dropna()

X = df.loc[:, df.columns != 'Risk']
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=16)


# For firms with a predicted probability of tax evasion greater than 0.5, what 
# proportion of firms evaded taxes? (add confusion matricies)

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
y_pred_ols = model.predict(X_test)
sns.histplot(y_pred_ols)

y_pred_ols_bin = np.where(y_pred_ols>0.5, 1, 0)
cm_ols = confusion_matrix(y_test, y_pred_ols_bin, normalize ='true')
#print(cm_ols)
# [[0.97959184 0.02040816]
#  [0.21678322 0.78321678]]

ax = sns.heatmap(cm_ols, annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('OLS: Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

# our OLS caught 78.32% of predicted evaders

# for firms with predicted probability over 0.8, what fraction evaded taxes?
# (add confusion matricies)
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
y_pred_ols = model.predict(X_test)
sns.histplot(y_pred_ols)

y_pred_ols_bin = np.where(y_pred_ols>0.8, 1, 0)
cm_ols = confusion_matrix(y_test, y_pred_ols_bin, normalize ='true')
#print(cm_ols)
# [[0.97959184 0.02040816]
#  [0.21678322 0.78321678]]

ax = sns.heatmap(cm_ols, annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('OLS: Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

# with a threshold of 0.8, our model only predicted 51% of those who committed
# tax fraud. 

# 5. using the first half of the data as training data, fit a KNN model on the 
# testing data with k=1. with and without normalizing

#normalize the data
X = df.drop(['Risk'], axis=1)
X_norm = pd.DataFrame(preprocessing.scale(X))
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.5, random_state=16)


knn_1 = KNeighborsClassifier(n_neighbors = 1)
knn_1.fit(X_norm_train, y_train)
y_pred_knn1 = knn_1.predict(X_norm_test)
accuracy_score(y_test, y_pred_knn1)

cm_knn1 = confusion_matrix(y_test, y_pred_knn1, normalize = 'true')
print(cm_knn1)
print(confusion_matrix(y_test, y_pred_knn1))

ax = sns.heatmap(cm_knn1, annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('OLS: Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

# the confusion matrix with normalized x shows that our model predicted correctly 93% of 
# those who did commit tax fraud

X = df.drop(['Risk'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=16)


knn_1 = KNeighborsClassifier(n_neighbors = 1)
knn_1.fit(X_train, y_train)
y_pred_knn1 = knn_1.predict(X_test)
accuracy_score(y_test, y_pred_knn1)

cm_knn1 = confusion_matrix(y_test, y_pred_knn1, normalize = 'true')
print(cm_knn1)
print(confusion_matrix(y_test, y_pred_knn1))

ax = sns.heatmap(cm_knn1, annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('OLS: Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

# the confusion matrix with non-normalized x shows that our model predicted
# 95% of those who commited tax fraud

# KNN performed btter with the non-normalized X, but not by a lot

# 6. k=7
# normalized:
X = df.drop(['Risk'], axis=1)
X_norm = pd.DataFrame(preprocessing.scale(X))
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.5, random_state=16)

knn_5 = KNeighborsClassifier(n_neighbors = 5)
knn_5.fit(X_norm_train, y_train)
y_pred_knn5 = knn_5.predict(X_norm_test)
accuracy_score(y_test, y_pred_knn5)
# accuracy of 95%

cm_knn_5 = confusion_matrix(y_test, y_pred_knn5, normalize='true')
ax = sns.heatmap(cm_knn_5, annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('OLS: Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()


# non-normalized
X = df.drop(['Risk'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=16)

knn_5 = KNeighborsClassifier(n_neighbors = 5)
knn_5.fit(X_train, y_train)
y_pred_knn5 = knn_5.predict(X_test)
accuracy_score(y_test, y_pred_knn5)
# accuracy of 96%

cm_knn_5 = confusion_matrix(y_test, y_pred_knn5, normalize='true')
ax = sns.heatmap(cm_knn_5, annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('OLS: Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()


# 7.
X = df.drop(['Risk'], axis=1)
X_norm = pd.DataFrame(preprocessing.scale(X))
X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.5, random_state=16)

def odd(n):
    return list(range(1, 2*n, 2))

ks = odd(194)

ac_rate = []
for i in ks:
     knn = KNeighborsClassifier(n_neighbors=i)
     knn.fit(X_norm_train, y_train)
     pred_i = knn.predict(X_norm_test)
     ac_rate.append(np.mean(pred_i == y_test))

max_value = max(ac_rate)
print(max_value)
# .966
opt_k = ac_rate.index(max_value)
print(opt_k)
# 0?

knni = KNeighborsClassifier()
para = {'n_neighbors':ks}
knn_cv = GridSearchCV(knni, para, cv = KFold(5, random_state=13, shuffle=True))
knn_cv.fit(X_norm, y)
print(knn_cv.best_params_)
# n_neighbors:1 
# where k = 1 is the best k
print(knn_cv.best_score_)
# .967

# 8. why should a false negative rate matter as much as a false positive rate?

































