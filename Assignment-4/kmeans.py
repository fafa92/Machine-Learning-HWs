import numpy as np

import time

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        selection=np.random.choice(len(x),self.n_cluster)
        centroids = [x[ppp] for ppp in selection]
#        print(centroids,'lklklklklk')
        N, D = x.shape
        R=[]
        j=0
        j1=0
        
        c=0
#        
#        for i in range(self.n_cluster):
#            centroids[i] = x[i]
        
            
            
        for iter in range(self.max_iter):
            start=time.time()
#            print(iter,'iter')
            self.classes = {}
            R=[]
            j1=j
            
            for i in range(self.n_cluster):
                self.classes[i] = []
                
#            
#            print('a')  
#            
#            
#            print('fffffff')
##            R = np.array([np.argmin([np.dot(x_i-centroids[y_k], x_i-centroids[y_k]) for y_k in centroids]) for x_i in x])
#            print('yyyyyyyyyy')
#            [self.classes[([np.argmin([np.dot(x_i-centroids[y_k], x_i-centroids[y_k]) for y_k in centroids])])].append(x_i) for x_i in x]
            
#            print(self.classes,'oooooooooo')
#            
            for data in x:
                l=[]
#                print(data)t
#                print(np.array(centroids))
#                print(data,centroids)
                bestclass=np.argmin(np.sum(np.multiply(data-centroids,data-centroids),axis=1)**(1/2))
#                for center in centroids:
#                    print(centroids,'popopopoopopoo')
#                    diff=data-centroids[center]
#                    
#                    l.append(np.linalg.norm(diff))
#                print(l)
#                print(data)
#                bestclass = l.index(min(l))
#                R.append(bestclass)
                self.classes[bestclass].append(data)
                R.append(bestclass)
#            print(R,'pppppppppppp')
                        
#            print('c')
#            originalcentroids = list(centroids)
#            print(originalcentroids,centroids,'hhhhhhhhhhhhhh',cccc)
#            print(centroids,'gfgfgfgf')
#            for i in self.classes:
#                print(len(self.classes[i]))
            for myclass in self.classes:
#                print(self.classes[myclass],'mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')
                if len(self.classes[myclass])!=0:
#                    print(centroids[myclass],'before')
                    centroids[myclass] = np.average(self.classes[myclass],axis=0)
#                    print(centroids[myclass],'after')
                else:
#                    print('zakhaaa')
#                    print(centroids[myclass],'beforeeeeee')
                    centroids[myclass]=centroids[myclass]
#                    print(centroids[myclass],'afteeeeeer')
     
#                print(centroids[myclass])
#                print(originalcentroids[myclass],'llllllllll')
                
#            print(originalcentroids,centroids,'gggggggggggggggg',cccc)
#            print(originalcentroids)
#            print(centroids)
#            print('d')
            end=time.time()
#            print(end-start,'time1')
            start=time.time()
            j=0.0
            
        
            end = True
            for center in range(N):
#                print(originalcentroids,'iiiiiiiiiiiiiiiiiiiiii')
                j+=np.sum((x[center]-centroids[R[center]])**2)
#                original_centroid = originalcentroids[center]
#                currentcentroid = centroids[center]
#                print(original_centroid,'kkkkkkkkkkkkkkkkk')
#                print(currentcentroid,'lolololololo')
#                print('saaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
#                print(currentcentroid)
#                print(original_centroid)
                
#            print(j/N,'jjjjjjjjjjjjjjjjjjjjjjjj',j-j1/N)
            j/=N
            if abs(j-j1) > self.e:
#                print('cccccccccccccccccccc',j1-j)
#                    print(original_centroid)
#                    print(currentcentroid)
#                    print('salaaaaaaaaaaaaaaaaaaaaaaaaaaaam')
                c+=1
                end = False
#            print(centroids,'mkmkmkmkmkmk')
#            print('e')
            if end == True:
                break
            end=time.time()
#            print(end-start,'time2')
                
  ###############              #### BAYAD vaghti kelasi behesh nemikhore sefr bedeeeeeeee###############################################
            
            

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)
        means=[]
#        print(centroids,'kkkkkkkkkkk')
#        for i in centroids:
#            print(i,'uuuuuuuuuuuuu')
#            means.append(i)
#            print(means[i],'6665655656')
#        print(means,R,c)
        return(np.array(centroids),np.array(R),iter+1)
        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#            'Implement fit function in KMeans class (filename: kmeans.py')
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, i = k_means.fit(x)
#        print(x)
#        print(y)
#        
#        print('dooooooooooooooooooood')
#        print(centroids.shape,membership.shape,i)
#        print(np.unique(y))
#        print(np.unique(membership))
#        print(centroids)
#        print(membership)
#        print(x.shape)
#        print(self.n_cluster)
        d={}
        for i in range(len(np.unique(y))):
                d[i] = []
        
        for i in range(len(y)):
            d[y[i]].append(i)
        
        
            
        dd={}
        for i in range(self.n_cluster):
            dd[i] = []

        for i in d:
            for j in d[i]:
                dd[membership[j]].append(i)
        
        final=[]
        for i in dd:
            if len(dd[i])!=0:
                final.append((max(dd[i],key=dd[i].count)))
            else:
                final.append(0)
#        print(final,'finaaaaaaaaaaaa')
#        print(d)
#        print(dd,'ggggf')
#        print(final)
            
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
#        selection=np.random.choice(len(x),self.n_cluster)
#        centroids = [x[ppp] for ppp in selection]
#        centroid_labels=[0 for i in range(self.n_cluster)]
#        
        centroid_labels=np.array(final)        
#        raise Exception(
#            'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        predict=[]
        for data in x:
            bestclass=np.argmin(np.sum(np.multiply(data-self.centroids,data-self.centroids),axis=1)**(1/2))
            predict.append(self.centroid_labels[bestclass])
        return np.array(predict)
            
            
#        raise Exception(
#            'Implement predict function in KMeansClassifier class (filename: kmeans.py')
        # DONOT CHANGE CODE BELOW THIS LINE
