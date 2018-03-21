import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import csv



def sigmoid(z):
    y = 1/(1+np.exp(-z))
    return y


def load_data(csvname):
    reader = csv.reader(open(csvname, "rb"), delimiter=",")
    d = list(reader)
    data = np.array(d).astype("float")
    X = data[:,0:8]
    y = data[:,8]
    y.shape = (len(y),)
    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    X = X.T
    return X,y


def softmax(X,y,w):
    alpha = 10**-2
    iteration_cap = 20
    
    gradient = 1
    misclassifications=[]
    iterator = 1
    while np.linalg.norm(w) > 10**-5 and  iterator <= iteration_cap:
        sig = sigmoid(np.multiply(-y, np.dot(X.T, w)))
        r = np.multiply(-y, sig)
        gradient = np.dot(X, r)
        hessian =  np.zeros((len(X),len(X)))
        for i in range(len(y)):
            hessian += sig[i] * (1-sig[i]) * np.matmul(mat(X[:,i]).T,mat(X[:,i]))
        w = w - np.dot(pinv(hessian),gradient)
        classi = sign(np.multiply(-y, np.dot(X.T, w)))
        misclassifications.append(list(classi).count(1))
        iterator +=1
    return misclassifications


def square_margin(X,y,w):
    iteration_cap = 20
    iterator = 1
    misclassifications = []
    gradient = 1
    while np.linalg.norm(w) > 10**-5 and  iterator <= iteration_cap:
        margin = 1-y*np.dot(X.T, w)
        r = np.zeros(len(y))
        for i in range(len(y)):
            r[i] = max(0,margin[i])
        r = -2*y*r
        gradient = np.dot(X,r)
        hessian =  np.zeros((len(X),len(X)))
        for i in range(len(y)):
            if margin[i]>0:
                hessian += np.matmul(mat(X[:,i]).T,mat(X[:,i]))
        hessian = 2*hessian
        w = w - np.dot(pinv(hessian),gradient)
        classi = sign(np.multiply(-y, np.dot(X.T, w)))
        misclassifications.append(list(classi).count(1))
        iterator += 1
    return misclassifications


X,y = load_data('breast_cancer_data.csv')
w = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001])
softmax_error=softmax(X,y,w)
sq_error=square_margin(X,y,w)
iterations=np.arange(2,22)
ps=plt.plot(iterations,softmax_error, color='black', label='Softmax')
pm=plt.plot(iterations,sq_error, color='purple', label='Square Margin')
plt.xlim(2, 20)
plt.xlabel('Iterations')
plt.ylabel('# of Misclassifications')
plt.legend()
plt.show()