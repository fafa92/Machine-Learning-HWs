import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm

    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here
#    print(w,'kkkkkkkkk')
    obj=[]
    N=y.shape[0]
#    print(w,'tttttttt',w.shape)
#    print(w.shape,'8888888888888888888888')
    
    s=0
    for i in range(len(X)):
        
        
        a=np.dot(X[i].reshape(-1,1).T,w)
        b=1-np.dot(y[i],a)
#            print(y[i].shape,a.shape,b.shape,X[i].shape,w.shape,'mmmmmmmmmmmm',b)
        s+=max(b,0)
    s=s/float(N)
#            print(s,'sss')
    #    print(s,'s')
    #    print(s,'2')
    mi=float((lamb/2))*np.dot(w.T,w)
#            print(mi)
    #    print(mi,'mi')
    obj.append(s+mi)
#    obj=np.array(obj)
#    print(obj.argmin(),'323232',obj[obj.argmin()])
#    print(obj_value,'obj')
        
        
    
    

#    print(obj[0][0],'faraz')
#    print(s+mi,'55555')

    return  obj[0][0][0]


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w
    - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
#    print(w,'ffffff',w.shape,w.T.shape)
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []
    for iter in range(1, max_iterations + 1):
#        print(iter,'111111111111111')
        A=[]
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch
        for i in A_t:
#            print(np.dot(ytrain[i],np.dot(w.T,Xtrain[i])),'1111')
            if (ytrain[i]*np.dot(Xtrain[i].reshape(-1,1).T,w)) < 1:
                
#                print('2222222222')
                A.append(i)

        h=1/float(lamb*iter)
#        print(h,'2222',lamb,iter)
        z=0
        for j in A:
            z+=ytrain[j]*Xtrain[j].reshape(-1,1)
        ss=(h/float(k))*z
#        print(ss.shape,'jhjhjhjh',w.shape)
        res=(1-h*lamb)*w+ss
#        print(lamb,'11111',1/lamb**(1/2),'2222',np.linalg.norm(res),'333',sum(res))
#        print((1/(lamb**(1/2)))/sum(res))
#        print(res,'res',res.shape)
        ww=min(1,(1/(lamb**(1/2)))/np.linalg.norm(res))*res
        w=ww
#        print(w.shape,'saayaaaah')
        train_obj.append(objective_function(Xtrain, ytrain, w, lamb))
#        print(train_obj)
        
            
        # you need to fill in your solution here
#    print(w,'lllllllllllllllll',w.shape)
    
    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w, t = 0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here
    a=[]
    res=0
    Xtest=np.array(Xtest)
    for i in range(len(Xtest)):
        
        s=np.dot(w.T,Xtest[i].reshape(-1,1))
        if s<t:
            a.append(-1)
        else:
            a.append(1)
        
#    print(a)
    for i in range(len(ytest)):
        if ytest[i]  == a[i]:
            res+=1
#    print(res)
#    print(ytest)
    test_acc=res/float(len(ytest))

    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
