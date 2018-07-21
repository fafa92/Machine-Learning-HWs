from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
#        value=[]
#        for i in range(1,self.nb_features+1):
#            for j in values:
#                
#                value.append(j**(i))
#        values=value
#        print('1')
        
        ones = numpy.ones(len(features))
        features = numpy.column_stack((ones,features))
        self.product = numpy.dot(features.T, features) 
        self.w = numpy.dot(numpy.dot(numpy.linalg.pinv(self.product),features.T),values )
        
        return self.w
        
        """TODO : Complete this function"""
        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        predictions=[]
      
        
        for i in features:
            components = self.w[1:] * i
            predictions.append(sum(components) + self.w[0])
        
        
        return predictions
        
        """TODO : Complete this function"""
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        
        
        
        
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        raise NotImplementedError


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        ones = numpy.ones(len(features))
        features = numpy.column_stack((ones,features))
        
        
        Xt = numpy.identity(len(features.T))
        self.product = numpy.dot(features.T, features) 

        lambda_identity = self.alpha*Xt


        self.w = numpy.dot(numpy.dot(numpy.linalg.pinv(self.product+lambda_identity),features.T),values )
        return self.w
        
        """TODO : Complete this function"""
        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        predictions=[]
      
        
        for i in features:
            components = self.w[1:] * i
            predictions.append(sum(components) + self.w[0])
        
        return predictions
        """TODO : Complete this function"""
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
