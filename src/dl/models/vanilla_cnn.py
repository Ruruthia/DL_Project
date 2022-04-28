import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD


class CNNLit(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.step_size = config["step_size"]
        self.gamma = config["gamma"]
        self.save_hyperparameters()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(40000, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 14),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        outputs = self.linear_layers(x)
        return outputs

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out, target)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        _, preds = torch.max(outputs, 1)
        correct_preds = torch.sum(preds == y.data)
        return {"correct": correct_preds, "loss": loss, "total": len(y.data)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        _, preds = torch.max(outputs, 1)
        correct_preds = torch.sum(preds == y.data)
        return {"correct": correct_preds, "loss": loss, "total": len(y.data)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        _, preds = torch.max(outputs, 1)
        correct_preds = torch.sum(preds == y.data)
        return {"correct": correct_preds, "loss": loss, "total": len(y.data)}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log('train_accuracy', correct / total)
        self.log('train_loss', avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log('val_accuracy', correct / total)
        self.log('val_loss', avg_loss)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        self.log('test_accuracy', correct / total)
        self.log('test_loss', avg_loss)