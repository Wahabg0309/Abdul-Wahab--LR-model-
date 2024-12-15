import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Read the data
df = pd.read_csv("C:/Users/MasteR/Desktop/dataset.csv", header=None)
df = df.drop_duplicates() #Data pre-processing

X = df.iloc[:,0] #input feature
Y = df.iloc[:,1] #Actual output

#Data Splitting
#X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=40)

# #Data Normalization
mean_X = np.mean(X)
mean_Y = np.mean(Y)
std_X = np.std(X)
std_Y = np.std(Y)

X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

#Data visualization
plt.scatter(X, Y)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Scatter Plot of Experience vs Salary')
plt.show()
plt.close()
#Making 2d arrays
X = X.to_numpy()[:,np.newaxis]
Y = Y.to_numpy()[:,np.newaxis]
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
m,col = X.shape
ones = np.ones((m,1))
X = np.hstack((ones,X))
theta = np.zeros((2,1))
iterations = 5000
alpha = 0.001

#Compute Cost function
def cost_function(X,Y,theta):
    error = (np.dot(X,theta))-Y
    sqr_error = np.power(error,2)
    sum_sqrError = np.sum(sqr_error)
    loss_func = (1/(2*m))*sum_sqrError
    return loss_func
final_loss = cost_function(X,Y,theta)
#print(final_loss)
#Gradient Decent Algorithn
def gradient_decent_algo(X,Y,theta,alpha,m,iterations):
    history = np.zeros((iterations,1))
    for i in range(iterations):
        error = (np.dot(X,theta))-Y
        loss = (np.dot(X.T,error))*alpha/m
        theta = theta - loss
        history[i] = cost_function(X,Y,theta)
    return (history,theta)
(h, theta) = gradient_decent_algo(X,Y,theta,alpha,m,iterations)
print(h)

#Get the best fit line
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], np.dot(X, theta),color ='g')
plt.xlabel("Years")
plt.ylabel("Salary")
plt.show()
plt.close()

#Visualize iterations with respect to training loss
plt.plot(range(iterations), h, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.show()
plt.close()


#Get prediction
def get_prediction(x,theta):
    x = (x - mean_X)/std_X
    y = theta[0,0] + theta[1,0]*x
    y = (y*std_Y+mean_Y)
    return y





experience = float(input("Enter your experience in months"))
print("The expected salary is ", (get_prediction(experience,theta)))