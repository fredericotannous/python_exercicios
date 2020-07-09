# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:43:02 2020

@author: Frederico
"""


import pandas as pd
import numpy as np

# read into pandas}
df = pd.read_csv("approval.csv", header = None)
print(df)

# statistics
df_desc = df.describe()
df_info = df.info()

print(df_desc)
print(df_info)

# missing values
print(df.tail(17))   

# replacing '?' with Nan
df = df.replace('?', np.nan)    

# impute the missing values with mean imputation
df = df.fillna(df.mean())   

# how many NaN do we have in the df?
print(df.isnull().sum())

# impute missing values with the most frequent values in the respective columns
for col in df.columns:
    if df[col].dtypes == 'object':
        df = df.fillna(df[col].value_counts().index[0])
print(df.isnull().sum())

#converting non-numeric into numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for col in df.columns:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col])
        
# import train test split
from sklearn.model_selection import train_test_split

#convert to NumPy array
df = df.drop([11, 13], axis = 1)
df = df.values

# segregate features and labels into separate variables
X, y = df[:, 0:13], df[:,13]

# split into train and test sets   
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42
                                                    )

# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# import LogisticRegression
from sklearn.linear_model import LogisticRegression

# initiate classifier 
logreg = LogisticRegression()

# fit logreg to the train set
logreg.fit(rescaledX_train, y_train)

# import confison_matrix()
from sklearn.metrics import confusion_matrix

#predict instance
y_pred = logreg.predict(rescaledX_test)

# accuracy score
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# import GridSearchCV
from sklearn.model_selection import GridSearchCV

# define the grid of values
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# create dictionary
param_grid = dict(tol=tol, max_iter=max_iter)

# initiate GridSearchCV
grid_model = GridSearchCV(estimator = logreg, param_grid=param_grid, cv = 5)

# rescale X
rescaledX = scaler.transform(X)

# fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))










