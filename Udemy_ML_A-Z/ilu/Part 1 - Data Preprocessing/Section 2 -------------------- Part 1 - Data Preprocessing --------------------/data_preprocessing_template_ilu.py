# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # pd.read_csv.iloc returns a subset of the pandas frame
y = dataset.iloc[:, 3].values

# Taking care of missing data
"""
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean')
missingvalues = missingvalues.fit(X[:, 1:3]) # SimpleImputer.fit() returns missing values
X[:, 1:3] = missingvalues.transform(X[:, 1:3]) # SimpleImputer.transform() puts them back in X
"""

# Encoding categorical data
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
"""

# LabelEncoder
"""
labelencoder_X = LabelEncoder() # encodes, but with order 1, 2, 3
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # LabelEncoder.fit_transform does fit + transform
"""

# OneHotEncoder encodes in 'one hot' fashion, with multiple columns per category
"""
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
"""

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""