import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    
    def __init__(self):    
        self.clf_name = "DecisionTree"
      
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return
        
    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if (node.splittable) and (len(node.features[0])>=1):
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the dim of feature to be splitted

        self.feature_uniq_split = None # the feature to be splitted


    def split(self):

        def conditional_entropy(branches: List[List[int]]) -> float:
            
            ss=0
            s=np.sum(branches,axis=0)
            f=[]
            su=sum(sum(np.array(branches)))
            
            for i in range(len(branches[0])):
                for j in range(len(branches)):
                    g=branches[j][i]/float(s[i])
                    if g==0:
                        ss+=0
                        
                    else:
                        ss+=-(g)*np.log2(g)
                f.append(ss)
                ss=0
            jam=0
            for i in range(len(f)):
                jam+=f[i]*(s[i]/float(su))
            return jam
            
                
                    
                    
            
                
                
#            for i in branches:
#                g=sum(i)/float(s)
#                ss+=-g*np.log2(g)
            
                
            
            
                
            '''
            branches: C x B array, 
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of 
            '''
            ########################################################
            # TODO: compute the conditional entropy
            ########################################################
            
        
        for idx_dim in range(len(self.features[0])):
            
            
#            print(self.features)
#            print(self.features[0])
#            print(idx_dim,'22222')
#            print(conditional_entropy([[2,0,4],[0,4,2]]))
#            print(conditional_entropy([[1,2],[2,3]]))
#            print(conditional_entropy([[2,1],[3,2]]))
#            print(conditional_entropy([[1,2,1],[1,2,1]]),'5555')
            le=np.unique(self.features)
            op=[0 for j in range(len(np.unique(self.features)))]
            converting=[op]*len(le)
            
                ############################################################
                # TODO: compare each split using conditional entropy
                #       find the
                ############################################################
                #print(conditional_entropy([[2,0,4],[0,4,2]]))
            score = []
            uniq=[]
            for i in range(len(self.features)):
                for j in range(len(self.features[0])):
                    if self.features[i][j] not in uniq:
                        uniq.append(self.features[i][j])
   
            for i in range(len(uniq)):
#                print(i,'i')
                for j in range(len(self.features)):
#                    print(j,'j')
                    if uniq[i] == self.features[j][idx_dim]:
#                        print('yes')
                        converting[idx_dim][i] += 1
#                        print(cela,'ce1')
#            print(cela, ' ce')
            score.append(conditional_entropy(converting))
          
        ind = np.argmin(np.array(score))
        self.dim_split=ind
        new_features=[]
        for kk in self.features:
            new_features.append(kk[0:ind]+kk[ind+1:])
        
        
            
            
        uniqfe = []       
        
        for po in self.features:
            var=po[ind]
            if var not in uniqfe:
                uniqfe.append(po[ind])
        self.feature_uniq_split=uniqfe
        newchildren=[]
        newlabel=[]
        for lk in range(len(uniq)):
            newchildren.append([])
            newlabel.append([])
            
        for q in range(len(uniqfe)):
            for w in range(len(self.features)):
                if uniqfe[q] == self.features[w][ind]:
                    NF=[new_features[w]][0]
                    newchildren[q].append(NF)
                    NL=[self.labels[w]][0]
                    newlabel[q].append(NL)
                    
        count=len(newchildren)
        for cp in range(count):
            children = TreeNode(newchildren[cp], newlabel[cp], np.max(newlabel[cp])+1)
            self.children.append(children)
                

        ############################################################
        # TODO: compare each split using conditional entropy
        #       find the 
        ############################################################




        ############################################################
        # TODO: split the node, add child nodes
        ############################################################




        # split the child nodes
        for child in self.children:
            
        
            if (child.splittable)  and (len(child.features[0])>=1):
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
       
       
        
        if (self.splittable) and (len(feature)>=1):
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            feature=feature[:self.dim_split]+feature[self.dim_split+1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max


