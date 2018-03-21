
from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

######Euclian function with looping

# def euclidean(C, pt1):
# 	out = []
# 	dif = np.empty_like(C)
# 	pt1 = np.reshape(pt1, (len(pt1),))
# 	for i in range(C.shape[1]):
# 		dif[:,i] = C[:,i] - pt1
# 		dif[:,i] = dif[:,i]**2
# 		out.append(np.sqrt(np.sum(dif[:,i])))
# 	pos = 10
# 	min_val = 10
# 	for i in range(len(out)):
# 		if(out[i]<min_val):
# 			pos = i
# 			min_val =out[i]
# 	return pos

#Euclidean function without looping
def euclidean(C,pt1):
	dif = (C-pt1)
	dif = dif**2
	dif_sum = np.sqrt(np.sum(dif, axis=0))
	return np.argmin(dif_sum)


def K_means(X, K, C):
 	W = np.zeros((K,X.shape[1]))
 	cluster_ind = 0
 	C_old = np.zeros((C.shape))
 	while(not np.array_equal(C, C_old)): 
 		C_old = np.copy(C)
        #Updating W
 		for p in range(X.shape[1]):
 			x = np.reshape(X[:,p], (C.shape[0],1))
 			cluster_ind = euclidean(C, x)
 			W[:,p] *= 0
 			W[cluster_ind,p] = 1

        #Updating C
 		for k in range(C0.shape[1]):
 			Sk = np.sum(W[k])
 			C[:,k] = np.sum((W[k]*X), axis=1)/Sk

	return C, W

def plot_results(X, C, W, C0):

    K = np.shape(C)[1]

    # plot original data
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    plt.scatter(X[0,:],X[1,:], s = 50, facecolors = 'k')
    plt.title('original data')
    ax1.set_xlim(-.55, .55)
    ax1.set_ylim(-.55, .55)
    ax1.set_aspect('equal')

    plt.scatter(C0[0,0],C0[1,0],s = 100, marker=(5, 2), facecolors = 'b')
    plt.scatter(C0[0,1],C0[1,1],s = 100, marker=(5, 2), facecolors = 'r')

    # plot clustered data
    ax2 = fig.add_subplot(122)
    colors = ['b','r']

    for k in np.arange(0,K):
        ind = np.nonzero(W[k][:]==1)[0]
        plt.scatter(X[0,ind],X[1,ind],s = 50, facecolors = colors[k])
        plt.scatter(C[0,k],C[1,k], s = 100, marker=(5, 2), facecolors = colors[k])

    plt.title('clustered data')
    ax2.set_xlim(-.55, .55)
    ax2.set_ylim(-.55, .55)
    ax2.set_aspect('equal')
    
# load data
X = np.array(np.genfromtxt('Kmeans_demo_data.csv', delimiter=','))

C0 = np.array([[0,0],[0,0.5]])   # initial centroid locations
###########Uncomment for incorrect result
#C0 = np.array([[0,0],[-0.5,0.5]])

# run K-means
K = np.shape(C0)[1]

C, W = K_means(X, K, C0)

# plot results
plot_results(X, C, W, C0)
plt.show()