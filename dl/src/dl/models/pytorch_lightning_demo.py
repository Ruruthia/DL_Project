import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import WandbLogger

GPUS = 1
NUM_WORKERS = 12

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)


dataset = MNIST(f'{os.getcwd()}/data/01_raw', download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=1024)

model = LitModel()

trainer = Trainer(logger=WandbLogger(), gpus=GPUS)
trainer.fit(model=model, train_dataloaders=train_loader)
