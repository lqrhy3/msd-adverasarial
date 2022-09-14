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


def plot_sample(sample, slice_idxs, show=True):
    ret = blend_images(image=sample["image"][0], label=sample["label"][0], alpha=0.5, cmap="hsv", rescale_arrays=False)
    for slice_idx in slice_idxs:
        plt.figure("blend image and label", (12, 4))
        plt.subplot(1, 3, 1)
        plt.title(f"image slice {slice_idx}")
        plt.imshow(sample["image"][0, 0, :, :, slice_idx], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label slice {slice_idx}")
        plt.imshow(sample["label"][0, 0, :, :, slice_idx])
        plt.subplot(1, 3, 3)
        plt.title(f"blend slice {slice_idx}")
        # switch the channel dim to the last dim
        plt.imshow(torch.moveaxis(ret[:, :, :, slice_idx], 0, -1))
        if show:
            plt.show()
