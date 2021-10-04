import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression as LinearRegression_myImplementation
from sklearn.linear_model import LinearRegression as LinearRegression_sklearn

number_of_examples = 50
points = np.matrix([np.array([i for i in range(number_of_examples)]), np.array([0.4*i+3+6*np.random.random() for i in range(number_of_examples)])]).T
df = pd.DataFrame(points, columns = ['X', 'y'])

X = np.array(df['X'])
y = np.array(df['y'])


my_model = LinearRegression_myImplementation(verbose = False) # Noticed a huge increase in time with verbose = True so better keep it False if you're in a hurry
my_model.fit(X,y)

sklearn_model = LinearRegression_sklearn()
sklearn_model.fit(X.reshape(-1, 1) ,y)

print('My model parameters: ', my_model.theta)
print('Sklearn model parameters: ', sklearn_model.intercept_, sklearn_model.coef_)

plt.style.use('dark_background')
slope = np.sum(sklearn_model.coef_[0])
intercept = np.sum(sklearn_model.intercept_)
xplt = np.array([i for i in range(number_of_examples)])
yplt = slope*xplt + intercept
plt.plot(xplt, yplt, 'w-', label = 'Sklearn Model')

slope = np.sum(my_model.theta[1])
intercept = np.sum(my_model.theta[0])
xplt = np.array([i for i in range(number_of_examples)])
yplt = slope*xplt + intercept
plt.plot(xplt, yplt, 'r-', label = 'My Model')
plt.plot(df['X'],df['y'], 'go', label = 'Data')

plt.legend()
plt.show() # Both the lines overlap because they are pretty much the same so only the red line shall be visible