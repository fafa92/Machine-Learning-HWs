from typing import List

import numpy as np
import scipy

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return np.square(np.subtract(y_true, y_pred)).mean()
    assert len(y_true) == len(y_pred)

    raise NotImplementedError


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    tp=0
    fp=0
    tn=0
    fn=0
    f1=0
    
    for i,j in zip(real_labels,predicted_labels):
        if (i==j) and (i==1):
            tp+=1
        elif (i==j) and (i==0):
            tn+=1
        elif (j==0) and (i==1):
            fn+=1
            
        elif (j==1) and (i==0):
            fp+=1
#    print(tp,tn,fp,fn)
    if (tp+fp!=0) and (tp+fn!=0):
        
        pre=tp/(tp+fp)
        rec=tp/(tp+fn)
    else:
        return 0
    if (pre+rec)!=0:
        
        f1=float((2*(pre*rec)))/float(pre+rec)
    else:
        return 0
#    print(pre,rec,f1)
    return f1
    assert len(real_labels) == len(predicted_labels)

    raise NotImplementedError

def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    
    feature=[]
    f=[]
    for i in features:
        for p in range(k):
        
            for j in range(len(i)):
            
                feature.append(i[j]**(p+1))
        f.append(feature)
        feature=[]
 
    return f
    raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    aa=[]
    for i,j in zip(point1,point2):
        aa.append((i-j)**2)
        
    return np.sqrt(sum(aa))
    raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.dot(point1,point2)
    return scipy.spatial.distance.cosine(point1,point2)
    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]) -> float:
    p=0
    c=np.array(point1)-np.array(point2)
    
    for i in c:
        p+=(i**2)
    
#    e=euclidean_distance(point1,point2)
    return -1*np.exp(-1*p/2)
    raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        fe=[]
        t=[]
        c=0
        for i in features:
            for j in i:
                c+=j**2
            c=(c**(0.5))
            if c!=0:
                
                for j in i:
                    
                    fe.append(float(j)/float(c))
                t.append(fe)
                fe=[]
                c=0
            else:
                for j in i:
                    
                    fe.append(0)
                t.append(fe)
                fe=[]
                c=0
    
        return t
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [10, 0.1]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = scaler(test_features)
        # now test_features_scaled should be [20, 1]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        a=len(features)
        b=len(features[0])
        mi=[100000000]
        ma=[-100000000]
        
        for i in range(b):
            for j in range(a):
                if mi[-1]>features[j][i]:
                    mi[-1]=features[j][i]
                if ma[-1]<features[j][i]:
                    ma[-1]=features[j][i]
                        
                    
            mi.append(100000000)
            ma.append(-100000000)
        mi=mi[:-1]
        ma=ma[:-1]
        
        g=[]
        t=[]
        for i in range(a):
            for j in range(b):
                g.append(float(features[i][j]-mi[j])/float(ma[j]-mi[j]))
            t.append(g)
            g=[]
            
        return t    
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        raise NotImplementedError