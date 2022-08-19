#synthetic datasets from "An Efficient Explanation of Individual Classifications using Game Theory"/Sikonja2008

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sikonja_condint(N=1000):
    y= np.random.randint(0,2,N)
    X = np.zeros((N,8),dtype=np.int64)
    X[:,0] = np.logical_xor(y,np.random.random(N)>0.9).astype(int)
    X[:,1] = np.logical_xor(y,np.random.random(N)>0.8).astype(int)
    X[:,2] = np.logical_xor(y,np.random.random(N)>0.7).astype(int)
    X[:,3] = np.logical_xor(y,np.random.random(N)>0.6).astype(int)
    X[:,4] = np.random.randint(0,2,N)
    X[:,5] = np.random.randint(0,2,N)
    X[:,6] = np.random.randint(0,2,N)
    X[:,7] = np.random.randint(0,2,N)
    
    return X,y

def sikonja_xor(N=1000):
    X = np.random.randint(0,2,(N,6),dtype=np.int64)
    y = np.sum(X[:,:3],axis=1)%2
    y = np.logical_xor(y,np.random.random(N)>0.9).astype(int)
    return X,y

def sikonja_cross(N=1000):
    X = np.random.random((5*N,4))#generate more than required and reject
    def check_within(X,width=0.05):
        cond1 = np.logical_and(X[:,0]<0.5-width, np.abs(X[:,1]-0.5)>width)
        cond2 = np.logical_and(X[:,0]>0.5+width, np.abs(X[:,1]-0.5)>width)
        return np.logical_or(cond1,cond2)
    Xinside = check_within(X)
    X = X[np.logical_not(Xinside)]
    
    y = ((X[:,0]-0.5)*(X[:,1]-0.5)>0).astype(int)
    return X,y

def sikonja_group(N=1000):
    X = np.random.random((N, 4))
    y = np.zeros(N, dtype=np.int64)
    
    classes = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    coords = [0.2, 0.5, 0.8]
    idx = 0
    for xx in range(3):
        for yy in range(3):
            Nx = N-8*(N//9) if (xx == 2 and yy == 2) else N//9
            xmean = coords[xx]
            ymean = coords[yy]
            std = 0.05
            X[idx:idx+Nx, 0] = xmean+std*np.random.randn(Nx)
            X[idx:idx+Nx, 1] = ymean+std*np.random.randn(Nx)
            y[idx:idx+Nx] = classes[yy][xx]
            idx += Nx
    permutation = np.arange(len(y))
    np.random.shuffle(permutation)
    return X[permutation], y[permutation]

def sikonja_chess(N=1000):
    X = np.random.random((N,4))
    xbin = (X[:,:2]*4).astype(int)
    y = np.logical_xor(xbin[:,0]%2,xbin[:,1]%2).astype(int)
    return X,y

def sikonja_sphere(N=1000):
    X = np.random.random((N,4))
    y = (np.sum((X[:,:3]-0.5)*(X[:,:3]-0.5),axis=1)<0.5*0.5).astype(int)
    return X,y

def sikonja_disjunct(N=1000):
    X = np.random.randint(0,2,(N,5),dtype=np.int64)
    y = (np.sum(X[:,:3],axis=1)>0).astype(int)
    return X,y

def sikonja_random(N=1000):
    X = np.random.random((N,4))
    y = np.random.randint(0,2,N)
    return X,y

class Sikonja_Synthetic_DataFrame:
    def __init__(self, N=1000, dataset=0):
        if(dataset==0):
            print("Dataset: condint")
            self.ds_name = 'condint'
            X, y = sikonja_condint(2*N)
        elif(dataset==1):
            print("Dataset: xor")
            self.ds_name = 'xor'
            X, y = sikonja_xor(2*N)
        elif(dataset==2):
            print("Dataset: cross")
            self.ds_name = 'cross'
            X, y = sikonja_cross(2*N)
        elif(dataset==3):
            print("Dataset: group")
            self.ds_name = 'group'
            X, y = sikonja_group(2*N)
        elif(dataset==4):
            print("Dataset: chess")
            self.ds_name = 'chess'
            X, y = sikonja_chess(2*N)
        elif(dataset==5):
            print("Dataset: sphere")
            self.ds_name = 'sphere'
            X, y = sikonja_sphere(2*N)
        elif(dataset==6):
            print("Dataset: disjunct")
            self.ds_name = 'disjunct'
            X, y = sikonja_disjunct(2*N)
        elif(dataset==7):
            print("Dataset: random")
            self.ds_name = 'random'
            X, y = sikonja_random(2*N)
        
        self.df_train = pd.DataFrame(data=X[:N], columns=["X"+str(i) for i in range(len(X[0]))])
        self.df_train["Y"]=y[:N]
        self.df_test = pd.DataFrame(data=X[N:],columns = ["X"+str(i) for i in range(len(X[0]))])
        self.df_test["Y"]=y[N:]

        
        self.columns_features = [x for x in self.df_train.columns if x!="Y"]
        self.columns_target = "Y"
        self.column_names = np.array(self.df_train.columns)
                
        self.columns_all = self.df_train.columns
        self.columns_types = self.df_train.dtypes
        self.columns_cont = [x for x in self.df_train.columns if self.df_train[x].dtype==np.float64]
        self.columns_cat = [x for x in self.df_train.columns if self.df_train[x].dtype==np.int64]
        self.columns_cat_unique_vals = [len(self.df_train[c].unique()) for c in self.columns_cat]
        
        self.df=pd.concat([self.df_train,self.df_test])
        
    def get_df(self):
        return self.df
    
    def __len__(self):
        return len(self.df)

    def get_train_df(self):
        return self.df_train
        
    def get_test_df(self):
        return self.df_test
        
    def get_val_df(self):
        return None

    def plot_ds(self, axis1='X0', axis2='X1', test=False):
        df = self.df_test if test is True else self.df_train
        plt.figure(f'{self.ds_name}, {axis1} vs. {axis2}, test = {test}')
        markers = ['x', '.', '*']
        colors = ['r', 'b', 'g']
        for i_class in np.sort(df['Y'].unique()):
            mask = (df['Y'] == i_class)
            plt.scatter(df['X0'][mask], df['X1'][mask], color=colors[i_class], marker=markers[i_class], label=i_class)
        plt.legend()
