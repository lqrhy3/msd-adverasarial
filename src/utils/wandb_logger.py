import os
import shutil
from typing import Optional

import numpy as np
import wandb
from torch import nn
from omegaconf import OmegaConf
from monai.visualize.utils import blend_images
from monai.transforms import AsDiscrete


class WandBLogger:
    def __init__(
            self,
            cfg: dict,
            model: nn.Module,
    ):
        self.cfg = cfg
        self.run = wandb.init(
            project='msd-adversarial',
            name=self.cfg['run_name'],
            config=self.cfg,
            resume='allow',

        )

        wandb.watch(model, log='all', log_freq=100, log_graph=True)
        wandb.save(os.path.join(cfg['artefacts_dir'], 'train_config.yaml'), policy='now')

    def log_scalar(self, name: str, value: float, commit: Optional[bool] = None):
        self.run.log({name: value}, commit=commit)

    def log_image(self):
        pass

    def log_slices(self, name: str, inputs, outputs, labels):
        inputs = inputs.numpy().squeeze()
        outputs = AsDiscrete(argmax=True)(outputs.numpy().squeeze())
        labels = labels.numpy().squeeze()

        blended_tgt = blend_images(inputs, labels)
        blended_pred = blend_images(inputs, outputs)
        blended_image = np.concatenate([blended_tgt, blended_pred], axis=1)

        blended_slices = []
        num_slices = 100
        slice_idx_start = blended_image.shape[-1] // 2 - num_slices // 2
        slice_idx_end = slice_idx_start + num_slices + 1
        for slice_idx in range(slice_idx_start, slice_idx_end):
            slice = blended_image[..., slice_idx]
            blended_slices.append(wandb.Image(slice, caption=f'Slice: {slice_idx}'))

        wandb.log({name: blended_slices})

    def finish(self):
        self.run.finish()
