import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.regression import rmse

class DataframeDataset(torch.utils.data.Dataset):
    """Dataset pandas dataframes
    """

    def __init__(self, df, columns_cont, columns_target, columns_cat=[]):

        self.df = df
        self.columns_cont = np.array([c for c in columns_cont if not(c in columns_target)])
        self.columns_cat = np.array([c for c in columns_cat if not(c in columns_target)])
        self.columns_target = columns_target
        

    def __getitem__(self, index):
        if(len(self.columns_cat)==0):
            return np.array(self.df.iloc[index][self.columns_cont]).astype(np.float32), np.array(self.df.iloc[index][self.columns_target]).astype(np.float32)
        else:
            return np.array(self.df.iloc[index][self.columns_cont]).astype(np.float32), np.array(self.df.iloc[index][self.columns_cat]).astype(np.int64), np.array(self.df.iloc[index][self.columns_target]).astype(np.float32)

    def __len__(self):
        return len(self.df)
    
    

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""
    #https://forums.pytorchlightning.ai/t/how-to-access-the-logged-results-such-as-losses/155/5
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.logged_metrics)

class DNNRegressor(pl.LightningModule):

    def __init__(self, n_hidden=[120,120], tau = 1.0, dropout = 0.05, N=10, std_y_train=1.,lr=1e-2,T=10000):
        super().__init__()
        
        self.std_y_train = std_y_train
        
        self.lr = lr
        self.tau = tau
        self.T = T
        #allow sequentially increasing dropout
        if(not(isinstance(dropout,list))):
            dropout = [dropout]*(len(n_hidden)+1)
            
        #AdamW
        lengthscale = 1e-2
        self.reg = lengthscale**2 * (1 - dropout[-1]) / (2. * N * tau)

        layers = []
        layers.append(nn.Dropout(dropout[0]))

        layers.append(nn.Linear(N,n_hidden[0]))
        
        for i in range(len(n_hidden)):
            layers.append(nn.Dropout(dropout[i-1]))
            layers.append(nn.Linear(n_hidden[i],n_hidden[i+1] if i<len(n_hidden)-1 else 1))
            if(i<len(n_hidden)):
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        
    def enable_dropout(self, enable=True):
        for module in self.layers.modules():
            if 'Dropout' in type(module).__name__:
                if(enable):
                    module.train()
                else:
                    module.eval()
    
            
    def forward(self, x):
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.mean(torch.pow(y_hat-y,2))
        self.log('train_loss', loss)
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
                
        self.enable_dropout(False)
        single_pred = self(x).squeeze()
        loss = torch.mean(torch.pow(single_pred-y,2))
        
        val_rmse_single = rmse(single_pred,y)*self.std_y_train

        self.enable_dropout(True)
        multi_pred = torch.stack([self(x).squeeze() for _ in range(self.T)])
        mc_pred = torch.mean(multi_pred,dim=0)
        val_rmse_multi = rmse_fn(mc_pred,y)*self.std_y_train
        ll_loss_fn = LogLikelihood(self.tau,self.T)
        ll = ll_loss_fn(multi_pred, y, self.tau, self.T)

        self.log_dict({'val_rmse': val_rmse_single, 'val_rmse_multi': val_rmse_multi, 'val_ll':ll, 'val_loss': loss})
        return loss
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def LogLikelihood(x, y, tau=0.1, T=10000):
        return torch.mean(torch.logsumexp(-0.5 * self.tau * torch.pow(y[None] - x,2), dim=0) - np.log(self.T) - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
    