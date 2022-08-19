#synthetic datasets from  arXiv:1907.06698

import numpy as np
import pandas as pd

def parr_synthetic0(N=2000):
    X=np.random.uniform(0,10,(N,3))
    y=X[:,0]*X[:,0]+X[:,0]*X[:,1]+X[:,2]+5*X[:,0]*np.sin(3*X[:,1])+10
    return X,y

def parr_synthetic1(N=2000,sigma=1):
    X=np.random.uniform(-2,2,(N,2))
    eps = np.random.normal(size=N)*sigma
    y=X[:,0]*X[:,0]+X[:,1]+eps+10
    return X,y

def parr_synthetic2():
    #AZ = 90, CA = 70, CO = 40, NV = 80, WA = 60
    base=np.array([90,70,40,80,60])
    names = ["AZ","CA","CO","NV","WA"]
    X = np.array([[[state,day] for state in range(len(base))] for day in range(365)]).flatten().reshape(len(base)*365,2)
    y = base[X[:,0]] + 10 * np.sin(2*math.pi/365*X[:,1]+math.pi)+4*np.random.normal(size=365*len(base))
    return X,y

def parr_synthetic3(N=2000):
    sex = np.random.randint(0,2,N) #0=male 1=female
    pregnant = sex * np.random.randint(0,2,N)
    heightm = 5*12+8+np.random.uniform(-7,8,N)
    heightf = 5*12+5+np.random.uniform(-4.5,5,N)
    height = sex*heightf + (1-sex)*heightm
    education = np.array([12,10])[sex]+np.random.uniform(0,8,N)
    X = np.stack([sex,pregnant,height,education],axis=1)
    y = 120 + 10*(height- np.amin(height))+ 40 * pregnant - 1.5*education
    return X,y

class Parr_Synthetic_DataFrame:
    def __init__(self,N=2000, dataset=0,sigma=1):
        if(dataset==0):
            X,y = parr_synthetic0(2*N) 
        elif(dataset==1):
            X,y = parr_synthetic1(2*N,sigma=sigma) 
        elif(dataset==2):
            X,y = parr_synthetic2() 
        elif(dataset==3):
            X,y = parr_synthetic3(2*N) 
        
        self.df_train = pd.DataFrame(data=X[:N],columns = ["X"+str(i) for i in range(len(X[0]))])
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


