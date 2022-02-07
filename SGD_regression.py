##Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('insurance.csv')

# x stands for the features or input
x = dataset.iloc[:, :-1].values

# y stands for the target or output
y = dataset.iloc[:, -1].values

print(x)

"""## Encoding categorical data"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The second, fifth, and sixth columns are categorical
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1, 4, 5])], remainder = 'passthrough')
x_onehot = np.array(ct.fit_transform(x))

print(x_onehot[0])

"""## Feature scaling"""

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

x_scaled = sc_x.fit_transform(x_onehot)

# Reshape y because StandardScaler expects a 2D array
y = y.reshape(len(y),1)
y_scaled = sc_y.fit_transform(y)
y_scaled = y_scaled.flatten()

print(x_scaled[0])

"""## Splitting the dataset into training set and test set"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.2, random_state = 0)

"""## Training the SGD Regression model on the training set"""

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RandomizedSearchCV

model = SGDRegressor()

space = dict()
space['penalty'] = ['l2', 'l1', 'elasticnet']
space['alpha'] = [0.0001, 0.001, 0.01, 0.1]
space['max_iter'] = [1000, 1500, 2000]
space['epsilon'] = [0.0001, 0.001, 0.01, 0.1]
space['learning_rate'] = ['constant', 'optimal', 'invscaling', 'adaptive']

search = RandomizedSearchCV(model, space, scoring='neg_mean_squared_error', n_jobs=-1, cv=3, random_state=1)
result = search.fit(x_train, y_train)

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

"""## Predicting the targets of test set"""

y_pred = search.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Calculating the mean squared error of the test set prediction"""

from sklearn.metrics import mean_squared_error

score = mean_squared_error(y_test, y_pred)

print(score)