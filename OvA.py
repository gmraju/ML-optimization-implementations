from __future__ import print_function
import numpy as np
import csv

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def checkSize(w, X, y):
	# w and y are column vector, shape [N, 1] not [N,]
	# X is a matrix where rows are data sample
	assert X.shape[0] == y.shape[0]
	assert X.shape[1] == w.shape[0]
	assert len(y.shape) == 2
	assert len(w.shape) == 2
	assert w.shape[1] == 10
	assert y.shape[1] == 10

def compactNotation(X):
	return np.hstack([np.ones([X.shape[0], 1]), X])

def readData(path):
	"""
	Read data from path (either path to MNIST train or test)
	return X in compact notation (has one appended)
	return Y in with shape [10000,1] and starts from 0 instead of 1
	"""
	reader = csv.reader(open(path, "rb"), delimiter=",")
	d = list(reader)
	# import data and reshape appropriately
	data = np.array(d).astype("float")
	X = data[:,0:783]
	y = data[:,784]
	y.shape = (len(y),1)  
	# pad data with ones for more compact gradient computation
	X = compactNotation(X)
	return X,y



def softmaxGrad(w, X, y):
	checkSize(w, X, y)
	X = X.T
	xtw = np.dot(X.T,w)
	sig_arg = np.multiply(-y, xtw)
	sig = sigmoid(sig_arg)
	r = np.multiply(-y, sig)
	gradient = np.dot(X, r)
	### RETURN GRADIENT
	return gradient




def accuracy(OVA, X, y):
	"""
	Calculate accuracy using matrix operations
	"""
	correct = 0.0
	total = float(y.shape[0])
	Yout = np.empty_like(y)
	for i in range(len(y)):
		x = X[i,:]
		out = np.dot(OVA.T,x)
		if(int(y[i][0]) == np.argmax(out)):
			correct += 1
	return correct/total




def gradientDescent(grad, w0, *args, **kwargs):
	max_iter = 500
	alpha = 0.001
	eps = 10^(-5)

	w = w0
	iter = 0
	while True:
		gradient = grad(w, *args)
		w = w - alpha * gradient

		if iter > max_iter or np.linalg.norm(gradient) < eps:
			break

		if iter  % 100 == 0:
			print("Iteration: %d " % iter)

		iter += 1
	return w





def oneVersusAll(Y, value):
	"""
	generate label Yout, 
	where Y == value then Yout would be 1
	otherwise Yout would be -1
	"""
	Y_new = np.empty_like(Y)
	Y_new = (Y == value).astype(int)
	Y_new *= 2
	Y_new -= 1
	return Y_new




if __name__=="__main__":

	trainX, trainY = readData('MNIST_data/MNIST_train_data.csv')
	trainY -= 1
	testX, testY = readData('MNIST_data/MNIST_test_data.csv')
	testY -= 1
	# # training individual classifier
	Nfeature = trainX.shape[1]
	Nclass = 10
	OVA = np.zeros((Nfeature, Nclass))
	Yclass = np.zeros((trainY.shape[0], Nclass))

	for i in range(Nclass):
		Yclass[:,i] = oneVersusAll(trainY, i)[:,0]
	w0 = np.zeros((Nfeature, Nclass))
	OVA = gradientDescent(softmaxGrad, w0, trainX, Yclass)
	print("The accuracy for test data is: "+str(accuracy(OVA,testX,testY)))
	print("The accuracy for training data is: "+str(accuracy(OVA, trainX, trainY)))



	

