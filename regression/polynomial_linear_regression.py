#polynomial regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#fitting linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
#fitting polynomial regression to the dataset
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)
#visualising the linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#visualising the polynomial linear regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff(polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#predicting a new result with linear regression
linear_regressor.predict([[6.5]])
#predicting a new result with polynomial linear regression
linear_regressor_2.predict(poly_regressor.fit_transform([[6.5]]))