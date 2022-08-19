import numpy as np
import pandas as pd
from pathlib import Path

class UCI_Adult_DataFrame(object):
    def __init__(self,path="./pred_diff/datasets/UCI_Datasets/Adult-Dataset/"):
        self.df_train = pd.read_csv(Path(path)/"adult.data",header=None)
        self.df_test=pd.read_csv(Path(path)/"adult.test",header=None)
        with open(Path(path)/"column_names.txt") as fp: 
            column_names = np.array([r.rstrip() for r in fp.readlines()])
        self.df_train.columns = column_names
        self.df_test.columns = column_names
        #preprocess
        self.column_categories={}
        for col in self.df_train.columns:
            if(self.df_train[col].dtype==object):
                self.df_train[col]=self.df_train[col].apply(lambda x:x.strip())
                self.df_test[col]=self.df_test[col].apply(lambda x:x.strip())
                
                self.column_categories[col]=np.array(self.df_train[col].unique())
                stoi = {s:i for i,s in enumerate(self.column_categories[col])}
                self.df_train[col]=self.df_train[col].apply(lambda x:stoi[x])
                self.df_test[col]=self.df_test[col].apply(lambda x:stoi[x[:-1] if col=="income_gt_50k" else x])
            else:#conver all numerical cols to float
                self.df_train[col]=self.df_train[col].astype(float)
                self.df_test[col]=self.df_test[col].astype(float)
        self.df_train = self.df_train.drop(["education-num"],axis=1)
        self.df_test = self.df_test.drop(["education-num"],axis=1)
        

        
        self.columns_features = [x for x in self.df_train.columns if x!="income_gt_50k"]
        self.columns_target = "income_gt_50k"
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
        

class UCI_Bike_DataFrame(object):
    def __init__(self,path="./pred_diff/datasets/UCI_Datasets/Bike-Sharing-Dataset/",daily=True):
        self.df = pd.read_csv(Path(path)/("day.csv" if daily else "hour.csv"))
        self.df.dteday = pd.to_datetime(self.df.dteday)
        self.df["day"]=self.df.dteday.apply(lambda x:x.day)
        self.df= self.df.drop(["dteday","instant"],axis=1)
        self.df = self.df.drop(["casual","registered"],axis=1)#sum gives cnt
                 
        self.columns_features = [x for x in self.df.columns if x!="cnt"]
        self.columns_target = "cnt"
        self.column_names = np.array(self.df.columns)
        
        self.columns_all = self.df.columns
        self.columns_types = self.df.dtypes
        self.columns_cont = [x for x in self.df.columns if self.df[x].dtype==np.float64]
        self.columns_cat = [x for x in self.df.columns if self.df[x].dtype==np.int64]
        self.columns_cat_unique_vals = [len(self.df[c].unique()) for c in self.columns_cat]
        
        np.random.seed(0)
        ids = np.random.permutation(range(len(self.df)))
        train_ids = ids[:int(0.8*len(self.df))]
        test_ids = ids[int(0.8*len(self.df)):]
        self.df_train = self.df.iloc[train_ids].copy()
        self.df_test = self.df.iloc[test_ids].copy()
        
        
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

class UCI_DataFrame(object):
    '''collects dataframe and further metadata from Gal's uncertainty repository'''
    def __init__(self,data_directory="bostonHousing", epochs_multiplier=500, num_hidden_layers=2, method="nn"):
        self.config = _prepare_config(data_directory, epochs_multiplier, num_hidden_layers, method)
        self.df = pd.DataFrame(data = np.loadtxt(self.config["_DATA_FILE"]))
        
        #set categorical columns based on cardinality
        cardinality = [len(np.unique(self.df[c])) for c in self.df.columns]
        for card, col in zip(cardinality,self.df.columns):
            if(card<=10):
                self.df[col]=self.df[col].astype(int) 
        
        with open(self.config["_COLUMN_NAMES_FILE"]) as fp: 
            self.column_names = np.array([r.rstrip() for r in fp.readlines()])
        self.df.columns = self.column_names
        
        self.columns_features = self.column_names[np.loadtxt(self.config["_INDEX_FEATURES_FILE"]).astype(int)]
        self.columns_target = self.column_names[np.loadtxt(self.config["_INDEX_TARGET_FILE"]).astype(int)]
        
        
        self.columns_all = self.df.columns
        self.columns_types = self.df.dtypes
        self.columns_cont = [x for x in self.df.columns if self.df[x].dtype==np.float64]
        self.columns_cat = [x for x in self.df.columns if self.df[x].dtype==np.int64]
        self.columns_cat_unique_vals = [len(self.df[c].unique()) for c in self.columns_cat]
        
        self.n_splits = np.loadtxt(self.config["_N_SPLITS_FILE"])
        
        #more hyperparameter ranges from Gal's paper
        self.n_hidden = np.loadtxt(self.config["_HIDDEN_UNITS_FILE"]).tolist()
        self.n_epochs = np.loadtxt(self.config["_EPOCHS_FILE"]).tolist()
        self.dropout_rates = np.loadtxt(self.config["_DROPOUT_RATES_FILE"]).tolist()
        self.tau_values = np.loadtxt(self.config["_TAU_VALUES_FILE"]).tolist()
        
    def get_df(self):
        return self.df
    
    def __len__(self):
        return len(self.df)

    def get_train_df(self, split=1):
        assert(split>0 and split <= self.n_splits)
        idx, _, _ = _get_train_val_test_indices(self.config["_DATA_DIRECTORY_PATH"],split)
        return self.df.iloc[idx].copy()
    
    def get_val_df(self, split=1):
        assert(split>0 and split <= self.n_splits)
        _, idx, _ = _get_train_val_test_indices(self.config["_DATA_DIRECTORY_PATH"],split)
        return self.df.iloc[idx].copy()
    
    def get_test_df(self, split=1):
        assert(split>0 and split <= self.n_splits)
        _, _, idx = _get_train_val_test_indices(self.config["_DATA_DIRECTORY_PATH"],split)
        return self.df.iloc[idx].copy()
    

    
def _prepare_config(data_directory="bostonHousing", epochs_multiplier=500, num_hidden_layers=2, method="nn"):
    config={}
    config["data_directory"] = data_directory
    config["epochs_multiplier"] = epochs_multiplier
    config["num_hidden_layers"] = num_hidden_layers

    config["method"] = method

    config["_RESULTS_VALIDATION_LL"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/validation_ll_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"
    config["_RESULTS_VALIDATION_RMSE"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/validation_rmse_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"
    config["_RESULTS_VALIDATION_MC_RMSE"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/validation_MC_rmse_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"

    config["_RESULTS_TEST_LL"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/test_ll_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"
    config["_RESULTS_TEST_TAU"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/test_tau_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"
    config["_RESULTS_TEST_RMSE"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/test_rmse_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"
    config["_RESULTS_TEST_MC_RMSE"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/test_MC_rmse_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"
    config["_RESULTS_TEST_LOG"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/results/log_"+config["method"]+"_" + str(config["epochs_multiplier"]) + "_xepochs_" + str(config["num_hidden_layers"]) + "_hidden_layers.txt"

    config["_DATA_DIRECTORY_PATH"] = "./pred_diff/datasets/DropoutUncertaintyExps-master/UCI_Datasets/" + config["data_directory"] + "/data/"
    config["_DROPOUT_RATES_FILE"] = config["_DATA_DIRECTORY_PATH"] + "dropout_rates.txt"
    config["_TAU_VALUES_FILE"] = config["_DATA_DIRECTORY_PATH"] + "tau_values.txt"
    config["_DATA_FILE"] = config["_DATA_DIRECTORY_PATH"] + "data.txt"
    config["_HIDDEN_UNITS_FILE"] = config["_DATA_DIRECTORY_PATH"] + "n_hidden.txt"
    config["_EPOCHS_FILE"] = config["_DATA_DIRECTORY_PATH"] + "n_epochs.txt"
    config["_INDEX_FEATURES_FILE"] = config["_DATA_DIRECTORY_PATH"] + "index_features.txt"
    config["_INDEX_TARGET_FILE"] = config["_DATA_DIRECTORY_PATH"] + "index_target.txt"
    config["_N_SPLITS_FILE"] = config["_DATA_DIRECTORY_PATH"] + "n_splits.txt"
    config["_COLUMN_NAMES_FILE"] = config["_DATA_DIRECTORY_PATH"] + "column_names.txt"
    
    return config

def _get_index_train_test_path(datadir, split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return datadir + "index_train_" + str(split_num) + ".txt"
    else:
        return datadir + "index_test_" + str(split_num) + ".txt" 
    
def _get_train_val_test_indices(datadir, split, train_val_pct=0.8):
    '''returns train, val, test splits for a given split (generally from 1 to 20)'''
    index_trainval = np.loadtxt(_get_index_train_test_path(datadir,split, train=True)).astype(int)
    index_test = np.loadtxt(_get_index_train_test_path(datadir,split, train=False)).astype(int)
    num_training_examples = int(train_val_pct * len(index_trainval))
    index_train = index_trainval[:num_training_examples]
    index_val = index_trainval[num_training_examples:]
    return index_train, index_val, index_test
