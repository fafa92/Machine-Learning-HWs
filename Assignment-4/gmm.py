import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        self.NNN=x.shape[0]
        self.DDD=x.shape[1]


        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            
            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            centroids, membership, ppppp = k_means.fit(x)
            vari=[np.eye(D)] * self.n_cluster
            p_i=[1./self.n_cluster] * self.n_cluster
            log=[]
            resp=np.zeros((N, self.n_cluster))
            def calc(mean,sig):
                
                fshape=x.shape[1]/2.
                if np.linalg.det(sig)!=0:
#                    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
                    mynorm=np.linalg.det(sig)**-.5**(2 * np.pi)** (-fshape)
                    prod=np.dot(np.linalg.inv(sig) , (x - mean).T)
                    res=np.einsum('ij, ij -> i',x -  mean,prod.T)
                    resfinal=np.exp(-.5*res.T)
#                    print(mynorm*resfinal,'sssssssssssssssssssssssss')
                    return mynorm*resfinal
                else:
#                    print('sefreeeeeeeeeeeeeeeeeeeee')
                    sig+= 1/1000.0*np.identity(sig.shape[0])
                    if np.linalg.det(sig)!=0:
                        mynorm=np.linalg.det(sig)**-.5**(2 * np.pi)** (-fshape)
                        prod=np.dot(np.linalg.inv(sig) , (x - mean).T)
                        res=np.einsum('ij, ij -> i',x -  mean,prod.T)
                        resfinal=np.exp(-.5*res.T)
                        return mynorm*resfinal
                    else:
                        sig+= 1/1000.0*np.identity(sig.shape[0])
                        mynorm=np.linalg.det(sig)**-.5**(2 * np.pi)** (-fshape)
                        prod=np.dot(np.linalg.inv(sig) , (x - mean).T)
                        res=np.einsum('ij, ij -> i',x -  mean,prod.T)
                        resfinal=np.exp(-.5*res.T)
                        return mynorm*resfinal
            
            for iter in range(self.max_iter):
                

                if len(log)>=self.max_iter:
                    break
                else:
                    for k in range(self.n_cluster):
                        resp[:, k] = p_i[k] * calc(centroids[k], vari[k])
                    
                    
                    loglike=self.compute_log_likelihood(resp)
#                    print(loglike,'3444444444444444444444')
                    log.append(loglike)
#                    print('3555555555555555555')
                    resp = (resp.T / np.sum(resp, axis = 1)).T
                    
            
                    n_s = np.sum(resp, axis = 0)
                    
                    for k in range(self.n_cluster):
                        centroids[k] = 1. / n_s[k] * np.sum(resp[:, k] * x.T, axis = 1).T
                        xmeans = np.matrix(x - centroids[k])
                        vari[k] = np.array(1 / n_s[k] * np.dot(np.multiply(xmeans.T,  resp[:, k]), xmeans))
                        p_i[k] = 1. / N * n_s[k]
                    if len(log) < 2 : 
                        continue
#                    print(log)
                    
#                    print(np.abs(loglike - log[-2]),'878787')
                    if np.abs(loglike - log[-2]) < self.e: 
                        break
            
#            raise Exception(
#                'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            
            selection=np.random.choice(len(x),self.n_cluster)
            centroids = [x[ppp] for ppp in selection]
            vari=[np.eye(D)] * self.n_cluster
            p_i=[1./self.n_cluster] * self.n_cluster
            log=[]
            resp=np.zeros((N, self.n_cluster))
            def calc(mean,sig):
#                print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
                fshape=x.shape[1]/2.
                if np.linalg.det(sig)!=0:
                    mynorm=np.linalg.det(sig)**-.5**(2 * np.pi)** (-fshape)
                    prod=np.dot(np.linalg.inv(sig) , (x - mean).T)
                    res=np.einsum('ij, ij -> i',x -  mean,prod.T)
                    resfinal=np.exp(-.5*res.T)
                    return mynorm*resfinal
                else:
#                    print('sefreeeeeeeeeeeeeeeeeeeee')
                    sig+= 1/1000.0*np.identity(sig.shape[0])
                    if np.linalg.det(sig)!=0:
                        mynorm=np.linalg.det(sig)**-.5**(2 * np.pi)** (-fshape)
                        prod=np.dot(np.linalg.inv(sig) , (x - mean).T)
                        res=np.einsum('ij, ij -> i',x -  mean,prod.T)
                        resfinal=np.exp(-.5*res.T)
                        return mynorm*resfinal
                    else:
                        sig+= 1/1000.0*np.identity(sig.shape[0])
                        mynorm=np.linalg.det(sig)**-.5**(2 * np.pi)** (-fshape)
                        prod=np.dot(np.linalg.inv(sig) , (x - mean).T)
                        res=np.einsum('ij, ij -> i',x -  mean,prod.T)
                        resfinal=np.exp(-.5*res.T)
                        return mynorm*resfinal
                    
            for iter in range(self.max_iter):
                

                if len(log)>=self.max_iter:
                    break
                else:
                    for k in range(self.n_cluster):
                        resp[:, k] = p_i[k] * calc(centroids[k], vari[k])
                        
                    
                    
                    loglike=self.compute_log_likelihood(resp)
#                    print(loglike,'3444444444444444444444')
                    log.append(loglike)
                    resp = (resp.T / np.sum(resp, axis = 1)).T
            
                    n_s = np.sum(resp, axis = 0)
                    
                    for k in range(self.n_cluster):
                        centroids[k] = 1. / n_s[k] * np.sum(resp[:, k] * x.T, axis = 1).T
                        xmeans = np.matrix(x - centroids[k])
                        vari[k] = np.array(1 / n_s[k] * np.dot(np.multiply(xmeans.T,  resp[:, k]), xmeans))
                        p_i[k] = 1. / N * n_s[k]
                    if len(log) < 2 : 
                        continue
#                    print(log)
                    
#                    print(np.abs(loglike - log[-2]),'878787')
                    if np.abs(loglike - log[-2]) < self.e: 
                        break
            
            
#            raise Exception(
#                'Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        self.pi_k=np.array(p_i)
        self.variances=np.array(vari)
        self.means=np.array(centroids)
        return len(log)
        
#        raise Exception('Implement fit function (filename: gmm.py)')
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')
#        print(N,'ddddddddddddddd')
        
        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE

        samples=np.ones((N,self.DDD))
#        selection=np.random.choice(self.pi_k.shape[0],N)
#        print(selection,'seeeeeeeeeeeeeeeeeeeeeeeeeeee',len(selection))
#        print(self.pi_k.shape)
#        randpi = [self.pi_k[ppp] for ppp in selection]
#        print(self.variances.shape[1])
 
#        print(self.variances[0][0].shape)
#        print(self.means[1].shape,'lllll')
        
        selection=np.random.multinomial(N, self.pi_k, 1 )[0]
#        print(selection)
        ddddd=0
        for i in range(len(selection)):
            for j in range(selection[i]):
                samples[ddddd]=np.random.multivariate_normal(self.means[i], self.variances[i])
                ddddd+=1
#                ddddd+=1
#                print(samples[i],'saaaaaaaaaaaa',i,ddddd)
        return samples
                
        
        
#        for i in selection:
##            fff=0
##            print(self.means[i],'ggggggggggggggg',self.means[i].shape)
##            print(self.variances[i][0].T,self.variances[i][0].shape)
#            
#            
#            fff=np.sum(self.variances[i],axis=1)
#            print(fff,'fffffffffffff')
#            for j in range(len(fff)):
#                
##                print(self.pi_k[i]*np.random.normal(self.means[i],fff[j]),'ggggggggggggggg')
##                print(samples[i])
#                samples[i][j]=self.pi_k[i]*np.random.normal(self.means[i][j],fff[j])
#            print(samples[i],'saaaaaaaaaaaa')
#        return samples
                
        
        
#        raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        
        return  np.sum(np.log(np.sum(x, axis = 1)))
        
#        raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
