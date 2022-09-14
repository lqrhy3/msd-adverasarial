from typing import Collection

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
import monai
import pytorch_lightning as pl

from src.utils.utils import object_from_dict


class UNetModule(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super(UNetModule, self).__init__()

        self.cfg = cfg

        self.model = self._configure_model()
        self.criterion = self._configure_criterion()
        self.batch = None

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        if self.batch is None:
            self.batch = batch
        else:
            batch = self.batch
        image, label = batch['image'], batch['label']
        idx = torch.argmax(torch.sum(label[0], dim=(0, 1, 2)))
        output = self.forward(image)
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(131)
        ax1.set_title('gt_label')
        ax1.imshow(label[0, 0, :, :, idx].detach())
        ax2 = fig.add_subplot(132)
        ax2.set_title('image')
        ax2.imshow(image[0, 0, :, :, idx].detach(), cmap='Greys_r')
        ax3 = fig.add_subplot(133)
        ax3.set_title('pred_label')
        ax3.imshow(torch.sigmoid(output)[0, 0, :, :, idx].detach())
        plt.show()

        loss = self.criterion(output, label)
        return loss

    def configure_optimizers(self):
        cfg_optimizer = self.cfg['optimizer']
        optimizer = object_from_dict(cfg_optimizer,
                                     params=filter(lambda x: x.requires_grad, self.model.parameters()))
        return optimizer

    def _configure_criterion(self):
        cfg_loss = self.cfg['loss']
        criterion = object_from_dict(cfg_loss)

        return criterion

    def _configure_model(self):
        cfg_model = self.cfg['model']
        model = object_from_dict(cfg_model)

        return model
