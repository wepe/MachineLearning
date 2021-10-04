import numpy as np
import pandas as pd

from LinearRegression import LinearRegression as LinearRegression_myImplementation
from sklearn.linear_model import LinearRegression as LinearRegression_sklearn

from sklearn import datasets

boston = datasets.load_boston()
boston

data = np.concatenate([boston['data'], boston['target'][:, None]], axis = 1)
cols = list(boston['feature_names'])+['target']
df = pd.DataFrame(data, columns = cols)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis = 1),df['target'], test_size = 0.1)

my_model = LinearRegression_myImplementation(learning_rate = 1e-8, verbose = False) # Noticed a huge increase in time with verbose = True so better keep it False if you're in a hurry
my_model.fit(X_train,y_train)

sklearn_model = LinearRegression_sklearn()
sklearn_model.fit(X_train ,y_train)

print('My model parameters: ', my_model.theta)
print('Sklearn model parameters: ', sklearn_model.intercept_, sklearn_model.coef_)

print('My model predictions: ', my_model.predict(X_test))
print('Sklearn model predictions: ', sklearn_model.predict(X_test))