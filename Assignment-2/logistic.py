from __future__ import division, print_function

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm
from itertools import combinations

#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression

    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
        
        

    """
    TODO: add your code here
    """
    for i in range(max_iterations):
        z = np.dot(X, w)
        
        
        gradient = np.dot(X.T,  sigmoid(z) - y)
        gradient /= N
        gradient *= step_size
        w -= gradient

    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    
    N, D = X.shape
    preds = np.zeros(N)


    """
    TODO: add your code here
    """     
    for i in range(N):
        
        z = np.dot(X[i], w) +b
        
        if sigmoid(z) >= 0.5:
            preds[i]=1
        else:
            preds[i]=0
        
    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, 
                     w0=None, 
                     b0=None, 
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement a multinomial logistic regression for multiclass 
    classification. Keep in mind, that for this task you may need a 
    special (one-hot) representation of classification labels, where 
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to. 
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
        
    Y=np.zeros((N,C))
    Y[np.arange(N), y] = 1


    for i in range(max_iterations):
        for j in range(C):
            pp=np.zeros((1,D))
            for k in range(len(X)):
            
                z = np.dot(X[k], w[j].T)+b[j]
                makh=0
                sor=softmax(z)
                
                for l in range(C):
                    
                    makh+=softmax(np.dot(X[k], w[l].T)+b[l])
    #                    print(makh,'maaakh')
                
    #                print(sor,makh)
                    
                    
                z=float(sor)/float(makh)    
                
                
                    
                if y[k]==j:
    #                    print(y[k],j,'111')
                    
                    g=np.dot(X[k].T,z-1)
    #                    print(X[k],z-1,g,'111')
                else:
    #                    print(y[k],j,'222')
                    g=np.dot(X[k].T,z-0)
    #                    print(X[k],z-0,g,'222')
                    
                pp+=g
                
    #                print(pp,j)
                
    
            gradient = pp[0]/float(N)
    #            print(gradient,'graaad')
            gradient *= step_size
            
    #            print(gradient,'step')
    #            print(w[j],'before')
#            print(w[j],gradient,'graaad')
            w[j] -= gradient


#    for i in range(max_iterations):
#        
#        for j in range(C):
##            for p in range(len(X)):
##                for pp,ppp in fe:
##                    if ppp==j:
##                        
#                
#            
##            
##            components = w[1:] * X
##            predictions.append(sum(components) + self.w[0])
#            pp=[]
#            z = np.dot(X, w[j])
#            
#            for k in range(len(Y)):
#                pp.append(Y[k][j])
#            pp=np.array(pp)
#            
#                
#            gradient = np.dot(X.T,  softmax(z) - pp)
##            gradient /= N
#            gradient *= step_size
#            w[j] -= gradient
#            


    """
    TODO: add your code here
    """

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def softmax(x):
    
    return np.exp(x) 

def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 

    """
    TODO: add your code here
    """   
#    for i in range(N):
#        ind=[]
#        for j in range(C):
#            
#        
#            z = sigmoid(np.dot(X[i], w[j]) +b)
#            ind.append(z[0])
#        for p in range(len(ind)):
# 
#            if ind[p]==max(ind):
#                preds[i]=p
#            
#         
#            
    for i in range(N):
        result= np.dot(X[i],w.T)+b
        preds[i]=np.argmax(result)

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using binary classifier and 
    one-versus-rest strategy. Recall, that the OVR classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    Y=np.zeros((N,C))
    ww=[]
    Y[np.arange(N), y] = 1
#    for i in range(len(y)):
#        if y[i]==0:
#            Y[i][0]=1
#        elif y[i]==1:
#            Y[i][1]=1
#        else:
#            Y[i][2]=1
        
    
    for i in range(max_iterations):
        
        for j in range(C):
            pp=np.zeros((1,D))
            for k in range(len(X)):
            
                z = np.dot(X[k], w[j].T)+b[j]
                
#                print(X[k].shape,w[j].shape)
                z=sigmoid(z)
#                print(z.shape)
                if y[k]==j:
#                    print(y[k],j,'111')
                    
                    g=np.dot(X[k].T,z-1)
#                    print(X[k],z-1,g,'111')
                else:
#                    print(y[k],j,'222')
                    g=np.dot(X[k].T,z-0)
#                    print(X[k],z-0,g,'222')
                    
                pp+=g
                
#                print(pp,j)
                

            gradient = pp[0]/float(N)
#            print(gradient,'graaad')
            gradient *= step_size
            
#            print(gradient,'step')
#            print(w[j],'before')
        
            w[j] -= gradient
#            print(w[j],'after')
#    ww.append((w[0],w[1],w[2]))
        
#    print(w,'5555555555')

    """
    TODO: add your code here
    """
#    print(ww)
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and predictions from binary
    classifier. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
    
    """
    TODO: add your code here
    """
    
    
    for i in range(N):
        result= np.dot(X[i],w.T)+b
        preds[i]=np.argmax(result)
#        print(preds)
        
#        com = combinations([i for i in range(1,C+1)], 2)
#        ind=[]
#        
#        for p,o in com:
#            p=p-1
#            o=o-1
#            
#            z = np.dot(X[i], w[p]) +b
#            zz = np.dot(X[i], w[o]) +b
##            print('2222',z,zz)
#            if z[0]>zz[0]:
#                ind.append(p)
#            else:
#                ind.append(o)
#                
#        d={}
##        print(ind)
#        for h in ind:
#            d[h]=d.get(h,0)+1
##        print('111')
##        print(d)
#        m=max(d, key=d.get)
##        print('m',m)
#        preds[i]=m
##        print(preds[:5])
                
        

    assert preds.shape == (N,)
    return preds



#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
        