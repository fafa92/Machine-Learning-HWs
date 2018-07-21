from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        
        self.errors = []
        
        
        for q in range(self.max_iteration):
            c = list(zip(features, labels))

            random.shuffle(c)
            
            features,labels = zip(*c)
            errors = 0
            for xi, target in zip(features, labels):
                p=0
                ss=np.dot(xi[1:], self.w[1:]) + self.w[0]*xi[0]
                if ss>=self.margin/2:
                    p=1
                else:
                    p=-1
                update =  (target -p)
                self.w[1:] += np.asarray(update) * xi[1:]
                self.w[0] += update
                if int(update)!=0:
                    
                    errors += int(update)
                    
            self.errors.append(errors)
            if self.errors[-1]==0:
                return True
        return False
        
        

        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        raise NotImplementedError
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        l=[]
        for i in features:
            ss=np.dot(i[1:], self.w[1:]) + self.w[0]*i[0]
            if ss>=self.margin/2:
                l.append(1)
            else:
                l.append(-1)
        return l
            
        
        
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        
        raise NotImplementedError

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    