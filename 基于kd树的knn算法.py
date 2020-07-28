import numpy as np

class binaryTreeNode():
    
    def __init__(self,data=None,left=None,right=None,split=None):
        self.data = data 
        self.left = left
        self.right = right
        self.split = split
        
    def getdata(self):
        return self.data
    
    def getleft(self):
        return self.left
    
    def getright(self):
        return self.right
    
    def getsplit(self):
        return self.split

    
class KNNClassfier(object):

    def __init__(self, k=1, distance='euc'):
        self.k = k
        self.distance = distance
        self.root = None

    def getroot(self):
        return self.root

    def kd_tree(self,train_X,train_Y):
        '''构造kd树'''        
        if len(train_X)==0:
            return None
        if len(train_X)==1:
            return binaryTreeNode((train_X[0],train_Y[0]))
        index = np.argmax(np.var(train_X,axis=0))
        argsort = np.argsort(train_X[:,index])
        left = self.kd_tree(train_X[argsort[0:len(argsort)//2],:],train_Y[argsort[0:len(argsort)//2]])
        right = self.kd_tree(train_X[argsort[len(argsort)//2+1: ],:],train_Y[argsort[len(argsort)//2+1: ]])
        root = binaryTreeNode((train_X[argsort[len(argsort)//2],:],train_Y[argsort[len(argsort)//2]]),left,right,index)
        return root

    def inOrder(self,root):
        '''中序遍历kd树'''
        if root == None:
            return None
        self.inOrder(root.getleft())
        print(root.getdata())
        self.inOrder(root.getright())

    def search_kd_tree(self,x,knn,root,nodelist):

        while len(knn)==0:
            if root.getleft() == None and root.getright() == None:
                return knn.append(root.getdata())

            if x[root.getsplit()]<root.getdata()[0][root.getsplit()]:
                if root.getleft()!=None:
                    nodelist.append(root.getleft())
                    self.search_kd_tree(x,knn,root.getleft(),nodelist)
                else:
                    nodelist.append(root.getright())
                    self.search_kd_tree(x,knn,root.getright(),nodelist)
            else:
                if root.getright()!=None:
                    nodelist.append(root.getright())
                    self.search_kd_tree(x,knn,root.getright(),nodelist)
                else:
                    nodelist.append(root.getleft())
                    self.search_kd_tree(x,knn,root.getleft(),nodelist)
        
        dis = np.linalg.norm(x-knn[0][0],ord=2)

        while len(nodelist)!=0:
            current = nodelist.pop()            
            # currentdis = np.linalg.norm(x-current.getdata()[0],ord=2)
            if np.linalg.norm(x-current.getdata()[0],ord=2)<dis:
                knn[0] = current.getdata()
            if current.getleft()!=None and np.linalg.norm(x-current.getleft().getdata()[0],ord=2)<dis:
                knn[0] = current.getleft().getdata()
            if current.getright()!=None and np.linalg.norm(x-current.getright().getdata()[0],ord=2)<dis:
                knn[0] = current.getright().getdata()

        return knn

    def fit(self,X,Y):
        '''
        X : array-like [n_samples,shape]
        Y : array-like [n_samples,1]
        '''        
        self.root = self.kd_tree(X,Y)
    def predict(self,X):
        output = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            knn = []
            knn = self.search_kd_tree(X[i,:],knn,self.root,[self.root])
            labels = []
            for j in range(len(knn)):
                labels.append(knn[j][1])
            counts = []
            # print('x:',X[i,:],'knn:',knn)
            for label in labels:
                counts.append(labels.count(label))
            output[i] = labels[np.argmax(counts)]
        return output
    def score(self,X,Y):
        pred = self.predict(X)
        err = 0.0
        for i in range(X.shape[0]):
            if pred[i]!=Y[i]:
                err = err+1
        return 1-float(err/X.shape[0])


if __name__ == '__main__':
    
    from sklearn import datasets
    import time
    
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target

    myknn_start_time = time.time()
    clf = KNNClassfier(k=5)
    clf.fit(x,y)
    print('myknn score:',clf.score(x,y))
    myknn_end_time = time.time()

    from sklearn.neighbors import KNeighborsClassifier
    sklearnknn_start_time = time.time()
    clf_sklearn = KNeighborsClassifier(n_neighbors=5)
    clf_sklearn.fit(x,y)
    print('sklearn score:',clf_sklearn.score(x,y))
    sklearnknn_end_time = time.time()

    print('myknn uses time:',myknn_end_time-myknn_start_time)
    print('sklearn uses time:',sklearnknn_end_time-sklearnknn_start_time)