#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:11:47 2020

@author: h4rrydog
"""

# %% Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Importing the dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# %% Encoding categorical data

rom sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# California: 1 0 0
# Florida:    0 1 0
# New York:   0 0 1

# %% Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# %% Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %% Predicting the Test set results

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %% Bonus question #1
# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# %% Bonus question #2
print(regressor.coef_)
print(regressor.intercept_)

# Profit = 86.6 * Dummy State 1 - 873 * Dummy State 2 + 786 * Dummy State 3 - 0.773 * R&D Spend + 0.0329 * Administration + 0.0366 * Marketing Spend + 42467.53
