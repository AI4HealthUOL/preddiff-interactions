from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .imputer_base import ImputerBase, ImputerBaseTabular
from .tabular_vae import TabularVAE


class VAEImputer(ImputerBase):
    '''
    imputes using a VAE and pseudo Gibbs sampling
    '''
    
    def __init__(self, train_data: np.ndarray, **kwargs):
        super().__init__()
        kwargs["bs"] = kwargs["bs"] if "bs" in kwargs.keys() else 64
        kwargs["bs"] = 64
        kwargs["bs_eval"] = kwargs["bs_eval"] if "bs_eval" in kwargs.keys() else 128
        kwargs["n_hidden"] = kwargs["n_hidden"] if "n_hidden" in kwargs.keys() else [1000,250]
        kwargs["lr"] = kwargs["lr"] if "lr" in kwargs.keys() else 1e-3
        kwargs["epochs"] = kwargs["epochs"] if "epochs" in kwargs.keys() else 20
        kwargs["gibbs_iterations"] = kwargs["gibbs_iterations"] if "gibbs_iterations" in kwargs.keys() else 10
        kwargs["gpus"] = kwargs["gpus"] if "gpus" in kwargs.keys() else 0
        kwargs["n_jobs"] = kwargs["n_jobs"] if "n_jobs" in kwargs.keys() else 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device = {self.device}')
        if self.device == torch.device(type='cpu'):
            print('overwrite kwargs[gpu]')
            kwargs["gpus"] = 0

        self.train_data = train_data.copy()
        self.clip_min = self.train_data.min()
        self.clip_max = self.train_data.max()
        self.kwargs = kwargs
        self.imputer_name = 'VAEImputer'
        self.imputer = None


        self.train_ds = TensorDataset(torch.from_numpy(np.array(self.train_data).reshape(len(self.train_data),-1)).float(), torch.zeros(len(self.train_data)))
        self.train_dl = DataLoader(self.train_ds, batch_size=self.kwargs["bs"], drop_last=False, num_workers=self.kwargs["n_jobs"])


    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        # def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        # n_samples = np_test.shape[0]
        # test_data.shape = (n_samples, *mask_impute.shape)
        # mask_impute: boolean array with shape of one test sample
        #
        # return
        # imputations = np.zeros((n_imputations, n_samples, *mask_impute.shape), dtype=test_data.dtype)      # shape: n_imputation, n_samples, *mask

        output_shape = tuple([n_imputations]+list(test_data.shape))


        test_ds = TensorDataset(torch.from_numpy(test_data.reshape(len(test_data),-1)).float(), torch.zeros(len(test_data)))
        test_dl = DataLoader(test_ds, batch_size=self.kwargs["bs_eval"], drop_last=False, num_workers=self.kwargs["n_jobs"])

        #fit the VAE
        # if(self.imputer is None or retrain is True):
        if self.imputer is None:        # always retrain
            print('Training TabularVAE')
            self.imputer = TabularVAE(n_input=28**2, learning_rate=self.kwargs["lr"])
            trainer = Trainer(gpus=self.kwargs["gpus"], progress_bar_refresh_rate=20, max_epochs=self.kwargs["epochs"])#,checkpoint_callback=False)
            trainer.fit(self.imputer, self.train_dl, test_dl)
            self.imputer.to(self.device)
        
        with torch.no_grad():
            icn = np.where(mask_impute == 1)[0]

            final_output = []
            for batch_id, (xinput, _) in enumerate(test_dl):
                output = []
                for imp in range(n_imputations):
                    x = xinput.clone()
                    x = x.to(self.device)
                    for i in range(self.kwargs["gibbs_iterations"]):
                        if icn.size == 0:       # nothing to impute
                            # print('Nothing to impute')
                            break
                        if i == 0:
                            maximum = max(self.clip_max, test_data[:, icn].max())
                            minimum = min(self.clip_min, test_data[:, icn].min())
                            x[:, icn].uniform_(minimum, maximum)
                        else:
                            x[:, icn] = xhat[:, icn]

                        xhat = self.imputer(x).detach()
                    output.append(x.detach().cpu())
                final_output.append(np.stack(output))#nimputations, bs, -1
        imputations = np.stack(final_output)
        imputations_reshape = imputations.reshape(output_shape)
        # final_output = torch.cat(final_output,dim=1)
        # final_output = final_output.numpy()
        # return
        # imputations = np.zeros((n_imputations, n_samples, *mask_impute.shape), dtype=test_data.dtype)      # shape: n_imputation, n_samples, *mask
        return imputations_reshape, None
