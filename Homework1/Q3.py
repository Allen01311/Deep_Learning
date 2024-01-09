import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data
train_data = loadmat('train.mat')
X_train = np.column_stack((train_data['x1'], train_data['x2']))
Y_train = train_data['y'].flatten()

# Load test data
test_data = loadmat('test.mat')
X_test = np.column_stack((test_data['x1'], test_data['x2']))
Y_test = test_data['y'].flatten()

# Initialize and train a logistic regression model
logistic_reg = LogisticRegression(solver='liblinear')
logistic_reg.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = logistic_reg.predict(X_test)

# Calculate the test error (percentage of misclassified test samples)
test_error = 100 * (1 - accuracy_score(Y_test, Y_pred))

# Report the test error
print("-----------------------------------")
print(f"Test Error: {test_error:.2f}%")
print("-----------------------------------")
