'''
Here is an example of implimentation of logistic regression 
Here is the example of the data should be predected: 
				----------------------------------
				Same as usual, we start with importing of libraries and the dataset. 
				This dataset contains 2 different test score of students and their status of admission 
				into the university. We are asked to predict if a student gets admitted into a university
				 based on their test scores.
				----------------------------------
Helmi GHANMI 
Dec 2018

'''

# import package 

import sys, os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
print('The packages are well imported !!!')

# create forlder for output test results 

if  not os.path.exists('output_result'):
	os.makedirs('output_result')

# Read the bdd file 
df = pd.read_csv("ex2data1.txt", header=None)
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print('--------------------------------')
print(df.head())
print('--------------------------------')
print(df.describe())

#Plot an example of independant data(first split the into pos and neg , the define the list)
pos, neg = (y==1).reshape(100,1), (y==0).reshape(100,1)

plt.figure(0)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o", s=10)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted", "Not admitted"],loc=0)
plt.savefig('output_result/Res_Example_1.png')
plt.show()
#------------------Def the sgmaoid function which differ the logistic from the linear regression 
def sigmaoid(z):
	"""
	Return the sigmaoid 
	"""
	#value_function_sig = 1/(1+np.exp(-z))
	#return value_function_sig
	return 1/(1+np.exp(-z))

# test and plot sigmaoid function
print('sigmaoid(0) is : ',sigmaoid(0))

z = np.array([-np.inf, 0., np.inf])
sigma = sigmaoid(z)
print(sigma)
plt.figure(1)
plt.plot(sigma)
plt.xlabel("x")
plt.ylabel("Sigmaoid(x)")
plt.legend(["Sigmaoid function"],loc=0)
plt.savefig('output_result/sgmaoid_function.png')
plt.show()

#To compute the cost function J(Θ) and gradient (partial derivative of J(Θ) with respect to each Θ)
def costFunction(theta, X, y):
	"""
	Tokes in numpy array, x and y and return the logistic 
	regression const function and gradient 
	"""
	m = len(y)
	preduction = sigmaoid(np.dot(X,theta))
	error = (-y*np.log(preduction))-((1-y)*np.log(1-preduction)) # the croos entropy function 

	cost = 1/m*sum(error)
	grad = 1/m*np.dot(X.transpose(), (preduction-y))

	return cost[0], grad



#Feature normalization 
def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    
    X_norm = (X - mean)/std
    
    return X_norm , mean , std

#Setting the initial_theta and test the cost function
m , n = X.shape[0], X.shape[1]
X, X_mean, X_std = featureNormalization(X)
X= np.append(np.ones((m,1)),X,axis=1)
y=y.reshape(m,1)
initial_theta = np.zeros((n+1,1))
cost, grad= costFunction(initial_theta,X,y)
print("Cost of initial theta is",cost)
print("Gradient at initial theta (zeros):",grad)


# Gradient Descent method 
def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each iteration
    """
    
    m=len(y)
    J_history =[]
    
    for i in range(num_iters):
        cost, grad = costFunction(theta,X,y)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    
    return theta , J_history

# plot gradient descent method 
alpha=1
num_iters=400
theta , J_history = gradientDescent(X,y,initial_theta,alpha,num_iters)

plt.figure(2)
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.legend(["Gradient descent for alpah = {},number iteration = {}".format(alpha, num_iters)],loc=0)
plt.savefig('output_result/Gradient_descent')
plt.show()

print("Theta optimized by gradiebnt descent is :", theta)
print("The cost of the optimized thata is :", J_history[-1])

# plotting of the decision boundary using the optimized theta
plt.scatter(X[pos[:,0],1],X[pos[:,0],2],c="r",marker="+",label="Admitted")
plt.scatter(X[neg[:,0],1],X[neg[:,0],2],c="b",marker="x",label="Not admitted")
x_value= np.array([np.min(X[:,1]),np.max(X[:,1])])
y_value=-(theta[0] +theta[1]*x_value)/theta[2]
plt.plot(x_value,y_value, "r")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)
plt.savefig('output_result/decision_boundary.png')
plt.show()

#----------------Add preducntion function -----------#
def classifierPredict(theta,X):
    """
    take in numpy array of theta and X and predict the class 
    """
    predictions = X.dot(theta)
    
    return predictions>0

p=classifierPredict(theta,X)
print("Train Accuracy:", sum(p==y)[0],"%")



print('-----------------Processing finished!----------------')
