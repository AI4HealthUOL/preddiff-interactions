#https://colab.research.google.com/drive/1-_LKx4HwAxl5M6xPJmqAAu444LTDQoa3
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from torchvision.datasets import MNIST
# from torchvision import transforms

import pytorch_lightning as pl

import torch.utils.data 
import numpy as np
import pandas as pd

from pytorch_lightning.core.decorators import auto_move_data

# digit = np.array(data_digits.iloc[image_id]).reshape(img_params.n_pixel, img_params.n_pixel)


class FlatMNISTModel(pl.LightningModule):
    def __init__(self, classes=10, n_hidden=[1000, 500], dropout=0.5, lr=1e-3, wd=1e-3, bs_test=1024, T=1, conv_encoder=False):

        super().__init__()
        self.save_hyperparameters()
        
        
        layers = []
        
        if(conv_encoder):
            layers.append(nn.Conv2d(1, 32, 3, stride=1, padding=1,bias=False))
            layers.append(nn.BatchNorm2d(32))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Flatten())
        
        #self.model = torch.nn.Sequential(*layers)
        dos = [dropout/2] * (len(n_hidden)-1) + [dropout]
                    
        for i in range(len(n_hidden)):
            layers.append(torch.nn.Linear((7*7*64 if conv_encoder else 28 * 28) if i == 0 else n_hidden[i-1], n_hidden[i]))
            layers.append(torch.nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dos[i]))

        layers.append(torch.nn.Linear(n_hidden[-1], classes))

        self.model = torch.nn.Sequential(*layers)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.T = T

    @auto_move_data
    def forward(self, x):
        if(self.hparams.conv_encoder):
            return self.model(x.view(x.size(0), 1, 28, 28))
        else:
            return self.model(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_accuracy(torch.argmax(y_hat, dim=-1), y)

        self.log('val_acc', self.val_accuracy, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
    
    def predict_proba(self, x):
        test_dataset = TensorDataset(torch.from_numpy(np.array(x)).float())

        test_loader = DataLoader(test_dataset, batch_size=self.hparams.bs_test, drop_last=False)
        
        res = []
        self.eval()
        with torch.no_grad():
            for data in test_loader:
                preds = F.softmax(self.forward(data[0])/self.T, dim=-1)
                #print(preds[0],preds.shape)
                res.append(preds.cpu().numpy())
        return np.concatenate(res, axis=0)

    def return_accurary(self, data, target):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        if isinstance(target, pd.DataFrame):
            target = target.to_numpy()
        y_hat = self.forward(torch.tensor(data))
        target = torch.tensor(target)
        acc = self.val_accuracy(torch.argmax(y_hat, dim=-1), torch.squeeze(target))
        return acc


class ConvMNISTModel(pl.LightningModule):
    def __init__(self, classes=10, n_hidden=[1000, 500], dropout=0.5, lr=1e-3, wd=1e-3, bs_test=1024, T=1):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        dos = [dropout / 2] * (len(n_hidden) - 1) + [dropout]

        # for i in range(len(n_hidden)):
        #     layers.append(torch.nn.Linear(28 * 28 if i == 0 else n_hidden[i - 1], n_hidden[i]))
        #     layers.append(torch.nn.ReLU(inplace=True))
        #     if dropout > 0:
        #         layers.append(torch.nn.Dropout(dos[i]))
        # layers.append(torch.nn.Linear(n_hidden[-1], classes))
        # self.model = torch.nn.Sequential(*layers)

        self.n_pixel = 28
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 10, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 10, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 1, 1),
            # torch.nn.Linear(25, classes)
        )
        self.linear = torch.nn.Linear(25, classes)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.T = T

    @auto_move_data
    def forward(self, x: torch.Tensor):
        n_batch = x.shape[0]
        in_conv = x.view((n_batch, 1, self.n_pixel, self.n_pixel))
        out_conv = self.conv(in_conv)
        out = self.linear(out_conv.view((n_batch, -1)))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_accuracy(torch.argmax(y_hat, dim=-1), y)

        self.log('val_acc', self.val_accuracy, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def predict_proba(self, x):
        test_dataset = TensorDataset(torch.from_numpy(np.array(x)).float())

        test_loader = DataLoader(test_dataset, batch_size=self.hparams.bs_test, drop_last=False)

        res = []
        self.eval()
        with torch.no_grad():
            for data in test_loader:
                preds = F.softmax(self.forward(data[0]) / self.T, dim=-1)
                # print(preds[0],preds.shape)
                res.append(preds.cpu().numpy())
        return np.concatenate(res, axis=0)

    def return_accurary(self, data, target):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        if isinstance(target, pd.DataFrame):
            target = target.to_numpy()
        y_hat = self.forward(torch.tensor(data))
        target = torch.tensor(target)
        acc = self.val_accuracy(torch.argmax(y_hat, dim=-1), torch.squeeze(target))
        return acc
