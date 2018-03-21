import numpy as np
import math
import random
import matplotlib.pyplot as plt

#load data from the file
def load_data():
    data = np.array(np.genfromtxt('galileo_ramp_data.csv', delimiter=','))
    return data

#Splitting input data into input and corrsponding output
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

#Splits the data into training and testing/validation sets
def get_train_test(split, validation_k_id):	 
	xtest, ytest = get_input_output(split[validation_k_id])
	xtrain, ytrain = get_input_output(np.concatenate(split[:validation_k_id]+split[validation_k_id+1:]))
	return xtrain, ytrain, xtest, ytest

#Creates matrix holding polynomial feature vectors for all inputs 
def poly_features(x,D):
    F = []
    x = np.reshape(x, [1,len(x)])
    F = np.repeat(x, D+1, axis=0)
    for i in range (D+1):
    	F[i,:] = np.power(F[i,:], i)
    return F

#Draw MSE vs. Degree plot
def make_plot(D, MSE_train, MSE_val):
	plt.figure()
	train, = plt.plot(D, MSE_train, 'yv--')
	val, = plt.plot(D, MSE_val, 'bv--')
	plt.legend(handles=[train, val], labels=['training_error', 'validation error'], loc='upper left')
	plt.xlabel('Degree of Polynomial basis')
	plt.ylabel('Error in log scale')
	plt.yscale('log')
	plt.show()


#Performs k-fold cross validation
def leave_one_out_cross_validation(data, D):
	split = split_data(data, D)
	mses_train = []
	mses_test = []

	k = len(data)
	for d in range(1,D+1):
		mse_test = 0
		mse_train = 0
		rmse_train = 0
		rmse_test = 0
		for i in range(k):
			xtrain, ytrain, xtest, ytest = get_train_test(split, i)
			F_train = poly_features(xtrain,d)
			F_test = poly_features(xtest,d)
			w = np.dot(np.linalg.pinv(np.dot(F_train, F_train.T)), np.dot(F_train, ytrain))
			#mse_train += np.linalg.norm(np.dot(F_train.T, w)-ytrain)
			#mse_test += np.linalg.norm(np.dot(F_test.T, w)-ytest)
			
			mse_train += (np.sum((np.dot(F_train.T,w)-ytrain)**2))/len(xtrain)
			mse_test += (np.sum((np.dot(F_test.T,w)-ytest)**2))/len(xtrain)			
			
			#np.testing.assert_almost_equal(rmse_train**2, mse_train, decimal = 3)
			#np.testing.assert_almost_equal(rmse_test**2, mse_test, decimal = 3)
		mses_train.append(mse_train/k)
		mses_test.append(mse_test/k)

	print 'The best degree of polynomial basis, in terms of validation error, is %d' % (mses_test.index(min(mses_test))+1)

	make_plot(np.arange(1,D+1), mses_train, mses_test)


data = load_data()
leave_one_out_cross_validation(data, 6)




