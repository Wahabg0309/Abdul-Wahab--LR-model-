import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/MasteR/Desktop/multidata.csv")
df = df.drop_duplicates()

# Data Splitting
X = df.iloc[:, :7]
Y = df.iloc[:, 7]

# Data Normalization
mean_X = np.mean(X, axis=0)
mean_Y = np.mean(Y)
std_X = np.std(X, axis=0)
std_Y = np.std(Y)

X = (X - mean_X) / std_X
Y = (Y - mean_Y) / std_Y

# Making 2d arrays
Y = Y.to_numpy()[:, np.newaxis]
Y = Y.reshape(-1, 1)
m, col = X.shape
ones = np.ones((m, 1))
X = np.hstack((ones, X))

theta = np.zeros([8, 1])
iterations = 3000
alpha = 0.001

# Get cost function
def cost_function(X, Y, theta):
    error = (np.dot(X, theta)) - Y
    sqr_error = np.power(error, 2)
    sum_sqrError = np.sum(sqr_error)
    j = (1 / (2 * m)) * sum_sqrError
    return j 

# Gradient Descent Algorithm
def gradient_decent(X, Y, theta, alpha, m, iterations):
    history = np.zeros((iterations, 1))
    for i in range(iterations):
        error = (np.dot(X, theta)) - Y
        loss = (np.dot(X.T, error)) * alpha / m
        theta = theta - loss
        history[i] = cost_function(X, Y, theta)
    return (history,theta)

(h, theta) = gradient_decent(X, Y, theta, alpha, m, iterations)

print(h)
print(theta)

#Visualize iterations with respect to training loss
plt.plot(range(iterations), h, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.show()
plt.close()