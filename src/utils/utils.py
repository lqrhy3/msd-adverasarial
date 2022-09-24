import os
import pydoc
from typing import Dict, Optional

from omegaconf import DictConfig
import torch
from torch.optim import Optimizer
from monai.visualize import blend_images
import matplotlib.pyplot as plt


def object_from_dict(d, parent=None, ignore_keys=None, **default_kwargs):
    assert isinstance(d, (dict, DictConfig)) and 'type' in d
    kwargs = d.copy()
    kwargs = dict(kwargs)
    object_type = kwargs.pop('type')

    if object_type is None:
        return None

    if ignore_keys:
        for key in ignore_keys:
            kwargs.pop(key, None)

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    # support nested constructions
    for key, value in kwargs.items():
        if isinstance(value, (dict, DictConfig)) and 'type' in value:
            value = object_from_dict(value)
            kwargs[key] = value

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)


def create_transform(cfg_transform: Dict):
    transform = []
    for cfg_transform_fn in cfg_transform['transform_fns']:
        transform.append(object_from_dict(cfg_transform_fn))

    _Compose = object_from_dict(cfg_transform['compose_fn'])
    if _Compose is not None:
        transform = _Compose.__class__(transform)

    return transform


def save_checkpoint(
        model: torch.nn.Module,
        epoch: int,
        artefacts_dir: str,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
):
    state_dict = model.state_dict()
    save_dict = {'epoch': epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    snapshot_path = os.path.join(artefacts_dir, 'snapshots')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    filename = os.path.join(snapshot_path, f'best_metric_checkpoint.pt')
    torch.save(save_dict, filename)


def load_checkpoint(model, path_to_checkpoint, device, optimizer = None):
        state_dict = torch.load(path_to_checkpoint, map_location=device)
        model.load_state_dict(state_dict['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(state_dict['optimizer'])