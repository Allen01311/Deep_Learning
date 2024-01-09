import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

#load 'data.mat'
data = loadmat('data.mat')
x = data['x'].flatten()
y = data['y'].flatten()

#1.1 Plot the data
plt.scatter(x, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot the data')
plt.legend()
plt.savefig("Q1_1.png")
plt.show()

#----------------------------------------------------------------------------------------------------------

#1.2 計算 the least square line
lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y)
theta0_line = lr.intercept_
theta1_line = lr.coef_[0]

#Plot the data
plt.scatter(x, y, label='Data')
plt.plot(x, theta0_line + x * theta1_line, color='red', label='Least Square Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Line Fit')
plt.legend()
plt.savefig("Q1_2.png")
plt.show()


#----------------------------------------------------------------------------------------------------------
#1.3 計算 the least square parabola
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x.reshape(-1, 1))
lr.fit(X_poly, y)
theta0_parabola = lr.intercept_
theta1_parabola, theta2_parabola = lr.coef_[1:]

#Plot the data
plt.scatter(x, y, label='Data')
plt.plot(x, theta0_parabola + x * theta1_parabola + x**2 * theta2_parabola, color='green', label='Least Square Parabola')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Parabola Fit')
plt.legend()
plt.savefig("Q1_3.png")
plt.show()

#----------------------------------------------------------------------------------------------------------
#1.4 計算 the least square quartic curve
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(x.reshape(-1, 1))
lr.fit(X_poly, y)
theta_quartic = lr.coef_
y_quartic = np.polyval(theta_quartic, x)

#Plot the data
plt.scatter(x, y, label='Data')
plt.plot(x, y_quartic, color='purple', label='Least Square Quartic Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Quartic Curve Fit')
plt.legend()
plt.savefig("Q1_4.png")
plt.show()

#----------------------------------------------------------------------------------------------------------
#1.5 計算3種多項式的 Mean Square Errors
mse_line = mean_squared_error(y, theta0_line + x * theta1_line)
mse_parabola = mean_squared_error(y, theta0_parabola + x * theta1_parabola + x**2 * theta2_parabola)
mse_quartic = mean_squared_error(y, y_quartic)

print(f"Mean Square Error (Line): {mse_line}")
print(f"Mean Square Error (Parabola): {mse_parabola}")
print(f"Mean Square Error (Quartic Curve): {mse_quartic}")