import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# load data.mat
data = loadmat('data.mat')
x = data['x'].flatten()
y = data['y'].flatten()

# 設array來儲存結果
lines = []
quartic_curves = []

# 隨機樣本數 = 30
num_samples = 30

# 重複隨機抽樣次數 = 200
num_iterations = 200

for _ in range(num_iterations):
    # 隨機選取30筆資料
    random_indices = np.random.choice(len(x), num_samples, replace=False)
    x_sampled = x[random_indices]
    y_sampled = y[random_indices]

    # Fit a line
    lr = LinearRegression()
    lr.fit(x_sampled.reshape(-1, 1), y_sampled)
    theta0_line = lr.intercept_
    theta1_line = lr.coef_[0]
    print('line:',theta0_line,theta1_line)
   

    # Fit a quartic curve
    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(x_sampled.reshape(-1, 1))
    lr.fit(X_poly, y_sampled)
    theta_quartic = lr.coef_
    print('quartic_curves',theta_quartic)
    print('----------------------------')
     
    # 計算 the fitted lines and quartic curves
    line_fit = theta0_line + x * theta1_line
    quartic_fit = np.polyval(theta_quartic, x)

    lines.append(line_fit)
    quartic_curves.append(quartic_fit)

# Plot the lines
plt.figure(figsize=(10, 5))
plt.title('Q2_Lines')
for line_fit in lines:
    plt.plot(x, line_fit, color='blue', alpha=0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Q2_Lines.png")
plt.show()

# Plot the quartic curves
plt.figure(figsize=(10, 5))
plt.title('Q2_Quartic Curves')
for quartic_fit in quartic_curves:
    plt.plot(x, quartic_fit, color='green', alpha=0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Q2_Quartic_Curves.png")
plt.show()