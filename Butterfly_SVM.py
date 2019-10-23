# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:03:21 2019

@author: jiabx
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix

## The Edited_Data removed column obs, palp, stams, plug, butt#

## Read in the data using pandas as a dataframe
but = pd.read_csv("Edited_Data.csv")
## Print the dataframe
#print(but)

## Removed all missing and non-numeric values 
but = but.dropna()
print(but)

## Check column names
for col in but.columns: 
    print(col) 

## Check data types 
but.dtypes

def stat_info(x):
    print("Statistics information for " + x.name + ":" + "\n")
    print("mean:")
    print(np.mean(x))
    print("median:")
    print(np.median(x))
    print("max:")
    print(np.max(x))
    print("min:")
    print(np.min(x))
    print("variance:")
    print(np.var(x))
    print("standard deviation:")
    print(np.std(x))
    print("\n")

#for col in but.columns:
##    print(but[col]) 
#    if type(col) != "category":
#        stat_info(but[col])
#stat_info(but.WC)
#but = but[(but['WC'] <= 4)]
#stat_info(but.WC)
#
#but.WC.unique()

#for i in range(len(but.WC)):
#    if but.WC[i] < 3:
#        but.WC[i] = 1 ## good condition
#    else:
#        but.WC[i] = 0 ## bad condition
##    print(but.WC[i])
#
#print(but.WC[0])

but.WC_Grouped.unique()
but = but.drop("WC", axis=1)
print(but.head())
but.columns
## Change data types to more appropriate ones
#but['WC'] = pd.Categorical(but.WC)
#but['WC'] = but['WC'].astype('float')
#but['LOCATION'] = pd.Categorical(but.LOCATION)
#but['GENDER'] = pd.Categorical(but.GENDER)
## Check data types 
#but.dtypes

data = but.drop("WC_Grouped", axis=1)
print(data)
data.columns
    
labels = but.WC_Grouped
print(labels)
###### Split the data ######
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.2)
###### SVM ######
#svr = SVR()
#svr.fit(data_train, labels_train)
#print(svr.score(data_train, labels_train))
#print(svr.score(data_test, labels_test))
svm = SVC(kernel='linear', C=0.5)
pred_labels_svm = svm.fit(data_train,labels_train).predict(data_test)
print("Confusion matrix for SVM:")
print(confusion_matrix(labels_test, pred_labels_svm))
print("Accuracy for SVM:", metrics.accuracy_score(labels_test, pred_labels_svm))