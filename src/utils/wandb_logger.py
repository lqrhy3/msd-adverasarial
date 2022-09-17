import os
import shutil
from typing import Optional

import wandb
from torch import nn
from omegaconf import OmegaConf


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

    def save_config(self):
        config_name = 'train_config.yaml'
        path_to_save_config = os.path.join(self.run.dir, config_name)
        OmegaConf.save(self.config, path_to_save_config)

    def finish(self):
        self.run.finish()
