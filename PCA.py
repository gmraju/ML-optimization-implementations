from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


def PCA_svd(X, K):

#Method 1

    U,S,V = np.linalg.svd(X, full_matrices=True)
    Uk = U[:,:K]
    Vk = V[:,:K]
    Skk = np.zeros((U.shape[0], V.shape[0]))
    np.fill_diagonal(Skk, S)
    Skk = Skk[:K,:K]
    
    C = np.dot(Uk,Skk)
    W = Vk.T
    # print C, 'C'
    # print W, 'W'
    return C, W


def PCA_optimization(X,K):
#Method 2
    
    Ck = np.random.randn(X.shape[0], K)
    Wk = np.random.randn(K,X.shape[1])
    old_Ck = np.copy(Ck)
    old_Wk = np.copy(Wk)
    threshold = 0.0001
    #while(not (np.array_equal(Ck,old_Ck) and np.array_equal(Wk, old_Wk))):
    while(np.linalg.norm(old_Ck - Ck) < threshold and np.linalg.norm(old_Wk - Wk) < threshold):
        old_Ck = np.copy(Ck)
        old_Wk = np.copy(Wk)       
        Ck = np.dot(np.dot(X, Wk.T), np.linalg.pinv(np.dot(Wk, Wk.T)))
        Wk = np.dot(np.dot(np.linalg.pinv(np.dot(Ck.T,Ck)),Ck.T),X)
    C = Ck
    W = Wk
    # print ' '
    # print C,'C'
    # print W,'W'
    return C, W

# plot everything
def plot_results(X, C, str):

    # Print points and pcs
    fig = plt.figure(facecolor = 'white',figsize = (10,4))
    fig.suptitle(str)
    ax1 = fig.add_subplot(121)
    for j in np.arange(0,n):
        plt.scatter(X[0][:],X[1][:],color = 'lime',edgecolor = 'k')

    #s = np.arange(C[0,0],-C[0,0],.001)
    s = np.arange(-.5, .5, 0.100)
    m = C[1,0]/C[0,0]
    ax1.plot(s, m*s, color = 'k', linewidth = 2)
    ax1.set_xlim(-.5, .5)
    ax1.set_ylim(-.5, .5)
    ax1.axis('off')

    # Plot projected data
    ax2 = fig.add_subplot(122)
    X_proj = np.dot(C, np.linalg.solve(np.dot(C.T,C),np.dot(C.T,X)))
    for j in np.arange(0,n):
        plt.scatter(X_proj[0][:],X_proj[1][:],color = 'lime',edgecolor = 'k')

    ax2.set_xlim(-.5, .5)
    ax2.set_ylim(-.5, .5)
    ax2.axis('off')

    return

# load in data
X = np.matrix(np.genfromtxt('PCA_demo_data.csv', delimiter=','))
n = np.shape(X)[0]
means = np.matlib.repmat(np.mean(X,0), n, 1)
X = X - means  # center the data
X = X.T
K = 1

# run PCA    
C, W = PCA_svd(X, K)
plot_results(X, C, 'Method 1 - SVD')
plt.show()


C, W = PCA_optimization(X, K)
plot_results(X, C, 'Method 2 - Optimized according to footnotes')
plt.show()

