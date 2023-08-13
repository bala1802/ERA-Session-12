from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        '''PrepLayer'''
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        '''Layer-1'''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        '''Max Pool'''
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        '''ResBlock-1'''
        self.resBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        '''Layer-2'''
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        '''Max Pool'''
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        '''Layer-3'''
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        '''Max Pool'''
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=2)
        '''ResBlock-2'''
        self.resBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        '''Average Pool'''
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        '''FC Layer'''
        self.fc = nn.Linear(512, self.num_classes)
    
    def forward(self, x):
        x = self.prepLayer(x)
        x = self.layer1(x)
        x = self.maxpool1(x)

        residualBlock1 = self.resBlock1(x)
        x = x + residualBlock1

        x = self.layer2(x)
        x = self.maxpool2(x)

        x = self.layer3(x)
        x = self.maxpool3(x)
        residualBlock2 = self.resBlock2(x)
        x = x + residualBlock2

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        x = F.softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "scores": scores, "y": y}

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)