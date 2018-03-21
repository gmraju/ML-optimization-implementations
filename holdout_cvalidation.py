import numpy as np
import math
import random
import matplotlib.pyplot as plt

#loads data from file
def load_data():
    data = np.array(np.genfromtxt('wavy_data.csv', delimiter=','))
    return data

#Splitting input data into x and y
def get_input_output(data):
	x = np.reshape(data[:,0],(np.size(data[:,0]),))
	y = np.reshape(data[:,1],(np.size(data[:,1]),1))
	return x,y

#Randomizing and splitting data into k folds
def split_data(data, k):
	split_data = []
	fold_size = len(data)/k
	np.random.shuffle(data)
	for i in range(k):
		split_data.append(data[i*fold_size:(i+1)*fold_size,:])
	return split_data

#Split the data into training and validation sets
def get_train_test(data, k, validation_k_id):
	split = split_data(data, k)	 
	xtest, ytest = get_input_output(split.pop(validation_k_id))
	xtrain, ytrain = get_input_output(np.concatenate(split))
	return xtrain, ytrain, xtest, ytest


#Calculates matrix contiang Fourier feature vector of all input points
def fourier_features(x,D):
	g = lambda (i, x): np.sin(2*math.pi*(i+1)*x) if i%2==0 else np.cos(2*math.pi*i*x) 
	F = []
	x = np.reshape(x, [len(x),1])
	F = np.repeat(x, 2*D+1, axis=1)
	for i in range (len(x)):
		F[i,:] = np.reshape(map(g, enumerate(F[i,:])), [1, 2*D+1])
	F[1,:] = 1
	return F.T


#Perform hold-out cross validation
def hold_out_cross_validation(xtrain,ytrain, xtest, ytest, D):
	mses_train = []
	mses_test = []

	for d in range(1, D+1):
		F_train = fourier_features(xtrain,d)
		F_test = fourier_features(xtest,d)
		w = np.dot(np.dot(np.linalg.pinv(np.dot(F_train, F_train.T)), F_train), ytrain)
		#mse_train = np.linalg.norm(np.dot(F_train.T, w)-ytrain)/np.size(ytrain)
		#mse_test = np.linalg.norm(np.dot(F_test.T, w)-ytest)/np.size(ytest)
		mse_train = (np.sum((np.dot(F_train.T,w)-ytrain)**2))/len(xtrain)
		mse_test = (np.sum((np.dot(F_test.T,w)-ytest)**2))/len(xtest)	

		mses_train.append(mse_train)
		mses_test.append(mse_test)

	optimal_D = mses_test.index(min(mses_test))
	x = np.concatenate((xtrain,xtest))
	y = np.concatenate((ytrain,ytest))
	F = fourier_features(x, optimal_D)
	w = np.dot(np.linalg.pinv(np.dot(F, F.T)), np.dot(F, y))
	mse = np.linalg.norm(np.dot(F.T, w)-y)/np.size(y)
	print 'The best degree of Fourier basis, in terms of validation error, is %d' % (mses_test.index(min(mses_test))+1)
	make_plot(np.arange(1,D+1), mses_train, mses_test)

#Draws MSE vs. Degree plot
def make_plot(D, MSE_train, MSE_val):
	plt.figure()
	train, = plt.plot(D, MSE_train, 'yv--')
	val, = plt.plot(D, MSE_val, 'bv--')
	plt.legend(handles=[train, val], labels=['training_error', 'validation error'], loc='upper left')
	plt.xlabel('Degree of Fourier basis')
	plt.ylabel('Error in log scale')
	plt.yscale('log')
	plt.show()


data = load_data()
xtrain, ytrain, xtest, ytest = get_train_test(data, 3, 2)
hold_out_cross_validation(xtrain, ytrain, xtest, ytest, 8)


