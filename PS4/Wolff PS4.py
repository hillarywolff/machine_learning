#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:19:40 2022

@author: hillarywolff
"""

# ch. 6, #9
# a. split the data set into a training set and a test set



# b. fit a linear model using least squares on the training set and report the
# test error



# e. fit a PCR model on the training set, with M chosen by cross-validation.
# report the test error obtained, along with the value of M selected by X-val



# f. fit a PLS model on the training set, with M chosen by cross-validation.
# report the test error obtained, along with the value of M selected by X-val



# g. commont on the results obtained. how accurrately can we predict the number
# of college applications received? is there much difference among the test
# errors resulting from these five approaches?



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
#




# ii. Fit a full, unpruned tree to the training data, with Purchase as the response and the
# other variables as predictors. What is the training error rate?
#









# iii. Create a plot of the tree.2 The plot is a mess, isn’t it? For the purposes of this
# question, fit another tree with the max_depth parameter set to 3 in order to get an
# interpretable plot. How many terminal nodes does the tree have? Pick one of the
# terminal nodes, and interpret the information displayed.
#









# iv. Predict the response on the test data, and produce a confusion matrix comparing
# the test labels to the predicted test labels. What is the test error rate?
#









# v. Determine the optimal tree size by tuning the ccp_alpha argument in scikit-
# learn’s DecisionTreeClassifier.3 You can use GridSearchCV for this pur-
# pose.
#







# vi. Produce a plot with tree size on the x-axis and cross-validated classification error
# rate on the y-axis calculated using the method in the previous question. Which tree
# size corresponds to the lowest cross-validated classification error rate?
#






# vii. Produce a pruned tree corresponding to the optimal tree size obtained using cross-
# validation. If cross-validation does not lead to selection of a pruned tree, then
# create a pruned tree with five terminal nodes.
#





# viii. Compare the training error rates between the pruned and unpruned trees. Which is
# higher? Briefly explain.
#






# ix. Compare the test error rates between the pruned and unpruned trees. Which is
# higher? Briefly explain.






















































