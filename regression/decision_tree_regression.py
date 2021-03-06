#  Decision Tree regression
#import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#fitting the Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
#predicting a new result
y_pred = regressor.predict([[6.5]])
#visualising the Decision Tree Regression result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff(Decision Tree Regression)')
plt.xlabel('Salary level')
plt.ylabel('Salary')
plt.show()