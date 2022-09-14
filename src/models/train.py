import os
from typing import Dict
import torch.nn
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl

from src.models.unet_module import UNetModule
from src.data.msd_data_module import MSDData
from src.utils.utils import object_from_dict, plot_sample


def create_transform(cfg_transform: Dict):
    cfg_transform = OmegaConf.to_container(cfg_transform, resolve=True)

    transform = []
    for cfg_transform_fn in cfg_transform['transform_fns']:
        transform.append(object_from_dict(cfg_transform_fn))

    _Compose = object_from_dict(cfg_transform['compose_fn'])
    if _Compose is not None:
        transform = _Compose(transform)

    return transform


def main():
    project_root = os.environ['PROJECT_ROOT']
    cfg = OmegaConf.load(os.path.join(project_root, 'src/configs/train.yaml'))

    train_transform = create_transform(cfg['transform']['train'])
    val_transform = None #create_transform(cfg['transform']['val'])

    datamodule = MSDData(
        root_dir=os.path.join(project_root, 'data/raw'),
        task='Task06_Lung',
        val_frac=0.1,
        batch_size=2,
        num_workers=0,
        train_transform=train_transform,
        val_transform=val_transform
    )

    model = UNetModule(cfg)

    trainer = pl.Trainer(limit_train_batches=1)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
