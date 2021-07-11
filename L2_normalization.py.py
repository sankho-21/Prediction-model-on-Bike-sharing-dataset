import numpy as np      #Importing the required modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import copy 
#Reading the training and testing data to pandas dataset
data = pd.read_csv(r'training_dataset.csv')
testdata = pd.read_csv(r'testing_dataset.csv')
#Essential constants; number of parameters(paranum)), learning rate(alpha), number of iterations(iter_num)
#lamdba value for regularization (lam)
paranum = 36
alpha = 0.001
lam = 30
iter_num = 1000
#Declaring the matrices we'll need to carry out the calcultaions and in training of the model
#X is the matrix corresponding to the training dataset and testX to testing dataset
X = np.zeros([10886, paranum - 1], dtype=np.dtype('f8'))
testX = np.zeros([10886, paranum - 1], dtype=np.dtype('f8'))
#Using the dataset, setting values of the matrix elements accordingly(for both training and test data)
for i in range(len(data)):
    if(data['holiday'][i] != 0):
        X[i][0] = 1
    X[i][1] = data['temp'][i]
    if(data['season'][i] != 4):
        X[i][data['season'][i] + 1] = 1
    X[i][5] = data['workingday'][i]
    if(data['hour'][i] != 23):
        X[i][data['hour'][i] + 6] = 1
    X[i][29] = data['year'][i]
    if(data['weekday'][i] != 6):
        X[i][data['weekday'][i] + 30] = 1 
    X[i][36] = data['humidity'][i]
    if[data['month'][i] != 12]:
        X[i][data['month'][i] + 36]
    X[i][48] = data['windspeed'][i]
    if(data['weather'][i] != 4):
        X[i][data['weather'][i] + 48] = 1

for i in range(len(testdata)):
    if(testdata['holiday'][i] != 0):
        testX[i][0] = 1
    testX[i][1] = testdata['temp'][i]
    if(testdata['season'][i] != 4):
        testX[i][testdata['season'][i] + 1] = 1
    testX[i][2] = testdata['workingday'][i]
    if(testdata['hour'][i] != 23):
        testX[i][testdata['hour'][i] + 7] = 1
    testX[i][29] = testdata['year'][i]
    if(testdata['weekday'][i] != 6):
        testX[i][testdata['weekday'][i] + 30] = 1 
    testX[i][36] = testdata['humidity'][i]
    if[testdata['month'][i] != 12]:
        testX[i][testdata['month'][i] + 36]
    testX[i][48] = testdata['windspeed'][i]
    if(testdata['weather'][i] != 4):
        testX[i][testdata['weather'][i] + 48] = 1
#SPlitting the training dataset into training set and validation set
trainData = X[:8704]
validData = X[8704:]
#Declaring required matrices and vectors
Y = np.transpose(np.array(data['count'][:8704]))  #Count values of the training dataset
testY = np.transpose(np.array(data['count']))     #COunt valus of the testing dataset
Yv = np.transpose(np.array(data['count'][8704:]))  #Count values of the validation dataset
W = np.zeros(paranum, dtype=np.dtype('f8'))   #A matrix to store the weights corresponding to the parameters we are considering

Xt = (np.column_stack((np.ones(8704),trainData))) # Adding a column of ones to the training, test, validation data for the bias 
testX = (np.column_stack((np.ones(len(testX)),testX)))
Xv = (np.column_stack((np.ones(2182),validData))) 

grad = np.zeros(paranum, dtype=np.dtype('f8')) # A matrix to store the gradient values in each iteration
W_hist = copy.deepcopy(W)  #A matrix to store the previous values of W_hist

def loss(w, X, y): # Declaring a function to evaluate the loss function with given X, y and W 
    y_pred = np.exp(np.dot(X, w))
    loss = ((y_pred - y) ** 2.0)
    return loss.mean(axis=None)
#Running the iteration for training through the whole training dataset 
for a in range(iter_num):
    grad = np.zeros(paranum, dtype=np.dtype('f8')) #Setting grad to zeros at the start of each iteration
    for i in range(paranum):
        for j in range(len(trainData)):
            grad[i] += (Y[j] - np.exp(np.dot(np.transpose(W), Xt[j]))) * Xt[j][i] #Upgrading the grad values
        grad[i] += 2 * lam * W_hist[i]  #Adding the L2 regularization term
    W += (alpha / 8704) * grad #Updating the weights using the grad vlaue evaluated and printing loss function and weights
    W_hist = copy.deepcopy(W)  #copying the values of weights into W_hist
    print("iter", a, (alpha / 8704) * grad, "\nloss fn: ", loss(W, Xt, Y))

print("\nweights.....\n", W)
print(np.exp(np.dot(Xv[0], W)))

print("\nweights.....\n", W)
print(np.exp(np.dot(Xv[0], W)))  #Printing the final weights

Yp = np.zeros(2182)              #Making a vector to store the predictions
Yp = np.exp(np.dot(Xv, W))        #predicting the values
for k in range(len(Yp)):         #Taking the floor for all the predictions
    Yp[k] = math.floor(Yp[k])
#Printing the accuracy as the RMSE of the predictions made on the testing dataset
print("\n\n*******ACCURACY****\n\n")

print("RMSE: ", loss(W, Xv, Yv) ** (0.5))
