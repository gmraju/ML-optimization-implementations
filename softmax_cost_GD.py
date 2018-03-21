
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import csv


# sigmoid for softmax/logistic regression minimization
def sigmoid(z): 
    y = 1/(1+np.exp(-z))
    return y

    
# import training data 
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "rb"), delimiter=",")
    d = list(reader)
    # import data and reshape appropriately
    data = np.array(d).astype("float")
    X = data[:,0:2]
    y = data[:,2]
    y.shape = (len(y),1)  
    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    X = X.T
    return X,y


#gradient descent function for softmax cost/logistic regression 
def softmax_grad(X,y):
    alpha = 10**-2
    gradient = 1
    w = array([2, 2, 2])
    w.shape = (3, 1)
    max_its = 30000
    iterator = 1
    while linalg.norm(gradient) > 10**(-5) and iterator <= max_its:
        sig = sigmoid(np.multiply(-y, np.dot(X.T, w)))
        r = np.multiply(-1*y, sig)
        gradient = np.dot(X, r)
        w = w - alpha*gradient
        iterator+=1
    return w


# plots everything 
def plot_all(X,y,w):
    # custom colors for plotting points
    red = [1,0,0.4]  
    blue = [0,0.4,1]
    # scatter plot points
    fig = plt.figure(figsize = (4,4))
    ind = np.argwhere(y==1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1,ind],X[2,ind],color = red,edgecolor = 'k',s = 25)
    ind = np.argwhere(y==-1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1,ind],X[2,ind],color = blue,edgecolor = 'k',s = 25)
    plt.grid('off')
    # plot separator
    s = np.linspace(0,1,100) 
    plt.plot(s,(-w[0]-w[1]*s)/w[2],color = 'k',linewidth = 2)
    # clean up plot
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.show()
    

# load in data
X,y = load_data('imbalanced_2class.csv')
# run gradient descent
w = softmax_grad(X,y)
print 'w: '
print w
# plot points and separator
plot_all(X,y,w)