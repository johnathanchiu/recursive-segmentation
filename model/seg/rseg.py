import pytorch_lightning as pl
import torch.nn as nn
import torch


class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.layer = nn.Linear(10, 1)  # Example layer

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
