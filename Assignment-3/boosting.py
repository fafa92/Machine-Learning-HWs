import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
    


  # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        
        self.clfs = clfs
        self.num_clf = len(clfs)
        if T < 1:
            
            self.T = self.num_clf
        else:
            self.T = T
    
        self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
        self.betas = []       # list of weights beta_t for t=0,...,T-1
        return      
    
    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
#        print(features,'444444444')
        return
    

    def predict(self, features: List[List[float]]) -> List[int]:
        results=[]
        for i in range(len(features)):
            ft=[0.0]
            ft=ft*len(self.clfs_picked)
            for current in range(len(self.clfs_picked)):
                if current ==0:
                    ft[current]=self.betas[current]*float(self.clfs_picked[current].predict(features[i]))
                else:
                    ft[current]=float(ft[current-1]+self.betas[current]*self.clfs_picked[current].predict(features[i]))
            ft=[1 if x > 0 else -1 for x in ft]
                    
            
            
            first=np.multiply(self.betas,ft)
            sumoffirst=sum(first)
            if sumoffirst>0:
                results.append(1)
            else:
                results.append(-1)
        return results
            
                    
            
        return 
        
        ########################################################
        # TODO: implement "predict"
        ########################################################
        

class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        
        self.clf_name = "AdaBoost"
     
            
        return
        
    def train(self, features: List[List[float]], labels: List[int]):
        clfs=list(self.clfs)
#        print(len(clfs),'88888')
#        print(len(features),'99999')
        weights=[]
        for i in range(len(features)):
            weights.append(1/float(len(features)))
            
        
        for i in range(self.T):
            eachh=[]
            for j in range(len(clfs)):
                ss=0.0
                for k in range(len(features)):
                    if labels[k]!= clfs[j].predict(features[k]):
                        ss+=weights[k]
                eachh.append(ss)
#            print(eachh,'kkkk')
            h_t=np.argmin(np.array(eachh))
            e_t=float(eachh[h_t])
#            print(e_t,'5555555555')
            self.clfs_picked.append(clfs[h_t])
            
            if e_t >=1 or e_t<=0:
                b_t= 0
            else:
                b_t=(1/2.0)*np.log((1-float(e_t))/e_t)
            self.betas.append(b_t)
            for l in range(len(features)):
                if clfs[h_t].predict(features[l])!=labels[l]:
                    weights[l]*float(np.exp(b_t))
                else:
                    weights[l]*=np.exp(-1.0*b_t)
                    
            sumofall=float(sum(weights))
            for l in range(len(weights)):
                
                weights[l]/=sumofall
            
            
                
                    
            
                    
                    

        ############################################################
        # TODO: implement "train"
        ############################################################
        
        
    def predict(self, features: List[List[float]]) -> List[int]:
        
        return Boosting.predict(self, features)

    
class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return
    
    def train(self, features: List[List[float]], labels: List[int]):
        clfs=list(self.clfs)
        pi=[1/2.0 for i in range(len(features))]
        N=len(features)
        z_t=[]
        weights=[]
        ft=[]
        for i in range(self.T):
            for j in range(len(features)):
                weights.append(float(float(pi[j])*float((1-pi[j]))))
                gg=((float((labels[j])+1)/2)-float(pi[j]))
                z_t.append(gg/weights[-1])
           
                
                
                
            eachh=[]
            for k in range(len(clfs)):
                ss=0.0
                for j in range(len(features)):
                    
                    preclf=clfs[k].predict(features[j])
                    
                    ss+=float(weights[j])*((float(z_t[j])-float(preclf))**2)
                eachh.append(ss)
            h_t=np.argmin(np.array(eachh))
            self.clfs_picked.append(clfs[h_t])
            self.betas.append(1/2.0)
            
            ft=[[0.0 for pp in range(self.T)] for po in range(len(features))]
            if i!=0:
                for d in range(N):
                    ft[d][i]+= float(float(ft[d][i-1])+float(clfs[h_t].predict(features[d])))
                
            else:
                for d in range(N):
                    ft[d][i]+=float(clfs[h_t].predict(features[d]))
                
                
            for h in range(len(pi)):
                pi[h]=1.0/(float(1.0)+np.exp(-1.0*ft[h][i]))
                
                
                    
                
                
            

                    
               
                
            
        
                
            
            
                
                
                
            
            
        
        ############################################################
        # TODO: implement "train"
        ############################################################
        
        
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
