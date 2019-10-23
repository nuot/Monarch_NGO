#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:01:22 2019

@author: xinlu
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_excel("All_Cleaned_Data.xlsx")

df.dtypes

columnName=df.columns.values
## Print the number of missing values in all variables 
for name in columnName:
    print(name, ': ')
    total_nas = df[name].isna().sum()
    print(total_nas)

drop_elements = ['# Gen.','OBS', 'PLUG','STAMS','PALP']
df1 = df.drop(drop_elements, axis = 1)

df1 = df1.apply(pd.to_numeric, errors = 'coerce')
df1 = df1.dropna()

seed = 7

###############################################################################
###################### decision tree regressor model for female ###############
###############################################################################
df_f = df1[df1['GENDER'] == 2]


x_f = df_f.drop(['BMI'], axis = 1)
y_f = df_f['BMI']

# split dataset into training and testing
x_f_train, x_f_test, y_f_train, y_f_test = train_test_split(x_f, y_f, 
                                                    test_size=0.3, random_state=seed)
model_f = DecisionTreeRegressor(criterion="mse",min_samples_leaf=5,random_state = seed)
model_f.fit(x_f_train, y_f_train)

y_f_predict = model_f.predict(x_f_test)

score_f = model_f.score(x_f_train, y_f_train)
# Have a look at R sq to give an idea of the fit

print("coefficient of determination R^2 of the prediction: ",score_f)
# coefficient of determination R^2 of the prediction:  0.9780724433309413

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_f_test, y_f_predict))
# Mean squared error: 0.00

# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(y_f_test, y_f_predict))
# Test Variance score: 0.91

# So let's run the model against the test data


fig, ax = plt.subplots()
ax.scatter(y_f_test, y_f_predict, edgecolors=(0, 0, 0))
ax.plot([y_f_test.min(), y_f_test.max()], [y_f_test.min(), y_f_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()



###############################################################################
###################### decision tree regressor model for male #################
###############################################################################
df_m = df1[df1['GENDER'] == 1] 


x_m = df_m.drop(['BMI'], axis = 1)
y_m = df_m['BMI']

# split dataset into training and testing
x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(x_m, y_m, 
                                                    test_size=0.3, random_state=seed)
model_m = DecisionTreeRegressor(criterion="mse",min_samples_leaf=5,random_state = seed)
model_m.fit(x_m_train, y_m_train)

y_m_predict = model_m.predict(x_m_test)

score_m = model_m.score(x_m_train, y_m_train)
# Have a look at R sq to give an idea of the fit ,
print("coefficient of determination R^2 of the prediction: ",score_m)
# coefficient of determination R^2 of the prediction:  0.9865864512173849

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_m_test, y_m_predict))
# Mean squared error: 0.00

# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(y_m_test, y_m_predict))
# Test Variance score: 0.95

# So let's run the model against the test data

fig, ax = plt.subplots()
ax.scatter(y_m_test, y_m_predict, edgecolors=(0, 0, 0))
ax.plot([y_m_test.min(), y_m_test.max()], [y_m_test.min(), y_m_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()
















