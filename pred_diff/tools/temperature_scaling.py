import pytorch_lightning as pl
import torch
import torch.nn as nn

# loosely based on https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61


class TemperatureScalingCalibration(pl.LightningModule):
    def __init__(self, model, lr = 1e-2, **kwargs):
        """
        Args:
            n_input: input dimension
            n_hidden: list of hidden dimensions (excluding latent dim)
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """

        super(TemperatureScalingCalibration, self).__init__()

        self.lr = lr
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        self.model.eval()
        inputs, labels = batch
        logits = self.model(inputs)
        
        loss = self.criterion(torch.div(logits, self.temperature), labels)
        #self.log("loss",loss)
        #print(batch_idx,loss, self.temperature)
        return loss

    def configure_optimizers(self):
        return torch.optim.LBFGS([self.temperature], lr=self.lr, max_iter=10000, line_search_fn='strong_wolfe')
