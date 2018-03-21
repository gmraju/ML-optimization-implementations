from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


#recommender systems via matrix completion
def matrix_complete(Xcorrect, X, K):
    N = X.shape[0]
    P = X.shape[1]
    
    #Random values will results in slightly higher RMSE-ALS with around 1000 iterations
    #Uncomment regularization lines for random values
    C = np.ones((N,K))
    W = np.ones((K,P))
    Cold  = np.copy(C)
    Wold = np.copy(W)
    it = 0


    while(it < 100):
        Cold = np.copy(C)
        Wold = np.copy(W)
        wp = np.empty_like(W[:,0])
        for p in range(W.shape[1]):
            lhs=0
            rhs=0
            flag = 0
            for i in range(C.shape[0]):
                if(X[i,p] != 0):
                    flag+=1
                    Ci = np.reshape(C[i], (len(C[i]),1))
                    lhs +=  np.dot(Ci, Ci.T)
                    #Regularization
                    #lhs = lhs + np.identity(lhs.shape[0])
                    rhs +=  X[i,p]*Ci.T
            if(flag !=0):
                wp = np.dot(np.linalg.pinv(lhs), rhs.T)
                W[:,p] = np.reshape(wp, (len(wp),))


        for n in range(C.shape[0]):
            lhs=0
            rhs=0
            flag=0
            for j in range(W.shape[1]):
                if(X[n,j]!=0):
                    flag+=1
                    Wj = np.reshape(W[:,j], (len(W[:,j]), 1))
                    lhs += np.dot(Wj, Wj.T)
                    #Regularization
                    #lhs = lhs + np.identity(lhs.shape[0])
                    rhs += X[n,j]*Wj.T
            if(flag!=0):
                cn = np.dot(rhs, np.linalg.pinv(lhs))
                C[n] = cn
        it+=1
        #check_RMSE(Xcorrect, X, C, W)
    return C, W


def check_RMSE(X, X_corrupt, C, W):
    recon = np.dot(C,W)
    RMSE_mat = np.sqrt(np.linalg.norm(recon - X,'fro')/np.size(X))
    print RMSE_mat, 'rmse'


def plot_results(X, X_corrupt, C, W):

    gaps_x = np.arange(0,np.shape(X)[1])
    gaps_y = np.arange(0,np.shape(X)[0])

    # plot original matrix
    fig = plt.figure(facecolor = 'white',figsize = (30,10))
    ax1 = fig.add_subplot(131)
    plt.imshow(X,cmap = 'hot',vmin=0, vmax=20)
    plt.title('original')

    # plot corrupted matrix
    ax2 = fig.add_subplot(132)
    plt.imshow(X_corrupt,cmap = 'hot',vmin=0, vmax=20)
    plt.title('corrupted')

    # plot reconstructed matrix
    ax3 = fig.add_subplot(133)
    recon = np.dot(C,W)
    plt.imshow(recon,cmap = 'hot',vmin=0, vmax=20)
    RMSE_mat = np.sqrt(np.linalg.norm(recon - X,'fro')/np.size(X))

    title = 'RMSE-ALS = ' + str(RMSE_mat)
    plt.title(title,fontsize=10)
    
# load in data
X = np.array(np.genfromtxt('recommender_demo_data_true_matrix.csv', delimiter=','))
X_corrupt = np.array(np.genfromtxt('recommender_demo_data_dissolved_matrix.csv', delimiter=','))

K = np.linalg.matrix_rank(X)
#print K
#print X_corrupt.shape

# run ALS for matrix completion
C, W = matrix_complete(X, X_corrupt, K)

# plot results
plot_results(X, X_corrupt, C, W)
plt.show()