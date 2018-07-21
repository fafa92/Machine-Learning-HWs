from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy
from operator import itemgetter
############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        
        self.train_labels=labels
        self.train_features=[]
        for i in range(len(features)):
            self.train_features.append((i,features[i]))
        return None
            
        
        
        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        if  'gaussian_kernel_distance'in str(self.distance_function):
            sett=[]
            target=[]
            zero=0
            one=0
            
            for i in features:
                for j,k in self.train_features:
                    if i!=k:
                        
                        sett.append((j,self.distance_function(i,k)))
                   
                        
                    
                sett.sort(key=itemgetter(1))
                kk=(-1*self.k)
                sett=sett[kk:]
                for o,p in sett:
                    if self.train_labels[o]==0:
                        zero+=1
                    else:
                        one+=1
                if zero>one:
                    target.append(0)
                else:
                    target.append(1)
                sett=[]
                zero=0
                one=0
            return target
        else:
            
            sett=[]
            target=[]
            zero=0
            one=0
            
            for i in features:
                for j,k in self.train_features:
                    if i!=k:
                        
                        sett.append((j,self.distance_function(i,k)))
                   
                        
                    
                sett.sort(key=itemgetter(1))
                sett=sett[:self.k]
                for o,p in sett:
                    if self.train_labels[o]==0:
                        zero+=1
                    else:
                        one+=1
                if zero>one:
                    target.append(0)
                else:
                    target.append(1)
                sett=[]
                zero=0
                one=0
            return target
            
            raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
