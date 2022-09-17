import pydoc
from omegaconf import DictConfig
import torch
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
