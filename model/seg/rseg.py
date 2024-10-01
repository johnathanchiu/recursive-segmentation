import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch


class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=1)
        self.out_layer = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.attention(x)
        return self.out_layer(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
