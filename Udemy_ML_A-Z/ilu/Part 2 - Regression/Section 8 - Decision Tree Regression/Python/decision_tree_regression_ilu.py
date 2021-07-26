#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:09:07 2021

@author: h4rrydog
"""

# %% Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %% Importing the Dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)


# %% Training the Decision Tree Regression model on the whole dataset

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# %% Predicting a new result

salary = regressor.predict([[6.5]])


# %% Visualising the Decision Tree Regression result (higher resolution)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

