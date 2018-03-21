import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data1(csvname):
    data = np.array(np.genfromtxt(csvname, delimiter=','))
    np.random.shuffle(data)
    x = np.reshape(data[:,:-1],(10000, 496))
    y = np.reshape(data[:,-1],(np.size(data[:,0]),))
    x = compactNotation(x)
    return x,y


def compactNotation(X):
    return np.hstack([np.ones([X.shape[0], 1]), X])


def plot(costs_stand, costs_stoch):
	s= np.arange(0,100)
	plt.plot(s, costs_stand, label ='Standard GD')
	plt.plot(s, costs_stoch, label ='Stochastic GD')
	plt.xlabel('Iterations')
	plt.ylabel('Objective Value')
	plt.legend(loc='upper right')
	plt.show()


def sigmoid(t):
	return 1/(1+np.exp(-t))


#Softmax cost calcualtion
def softmax_cost(w,X,y):
    return np.sum(np.log(1+ np.exp(-y * np.dot(X,w))))


def standardGD(X,y):
	iteration=1
	max_iter=100
	costs = []
	#Optimal step length calculation
	alpha=1/(((np.linalg.norm(X)**2))/4)

	W = np.zeros((len(X[0]),1))
	y = np.reshape(y, (len(y), 1))
	while iteration <= max_iter:
		sig = sigmoid(-y*np.dot(X,W))
		r= -y*sig
		gradient = np.dot(X.T, r)
		W = W-alpha*gradient
		costs.append(softmax_cost(W,X,y))
		print iteration
		iteration+=1
	return costs


def stochasticGD(X,y):
	iteration =1
	max_iter = 100
	costs = []
	w=np.zeros((len(X[0]),1))
	y = np.reshape(y, (len(y),1))
	while iteration <= max_iter:
		alpha = 1.0/iteration
		print iteration
		for i in range(X.shape[0]):
			xp = np.reshape(X[i], (1, len(X[i])))
			gradient = sigmoid(-y[i]*np.dot(xp,w))*-y[i]*xp
			w = w-alpha*gradient.T
		costs.append(softmax_cost(w,X,y))
		iteration+=1
	return costs



X,y = load_data1('feat_face_data.csv')
costs_stand = standardGD(X,y)
costs_stoch = stochasticGD(X,y)
plot(costs_stand,costs_stoch)
