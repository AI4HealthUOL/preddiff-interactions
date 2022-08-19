import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

# https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/0.2.0/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
# https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
# https://github.com/williamFalcon/vae_demo/blob/master/vae.py


class TabularVAE(pl.LightningModule):
    """
    Standard VAE for tabular data
    """

    def __init__(self, n_input, n_hidden=[500, 256],  kl_coeff = 0.1, lr = 1e-4, **kwargs):
        """
        Args:
            n_input: input dimension
            n_hidden: list of hidden dimensions (excluding latent dim)
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """

        super(TabularVAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        binary = True
        
        encoder_layers =[]
        decoder_layers = []        
        for i in range(len(n_hidden)-1):
            encoder_layers.append(nn.Linear(n_input if i==0 else n_hidden[i],2*n_hidden[i+1] if i==len(n_hidden)-2 else n_hidden[i+1]))
            decoder_layers.append(nn.Linear(n_hidden[len(n_hidden)-1-i],n_input if i==len(n_hidden)-2 else n_hidden[len(n_hidden)-2-i]))
            if(i!=len(n_hidden)-2):
                encoder_layers.append(nn.ReLU(inplace=True))
                decoder_layers.append(nn.ReLU(inplace=True))
            elif(binary):
               decoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x.view(x.size(0),-1))
        mu = x[:,:self.hparams.n_hidden[-1]]
        log_var = x[:,self.hparams.n_hidden[-1]:]
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = x[:,:self.hparams.n_hidden[-1]]
        log_var = x[:,self.hparams.n_hidden[-1]:]
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0),-1)
        z, x_hat, p, q = self._run_step(x)

        #if(self.hparams.binary):
        #    recon_loss = F.binary_cross_entropy_with_logits(x_hat, x.view(-1, 784), reduction='mean')
        #else:                           
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.hparams.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

