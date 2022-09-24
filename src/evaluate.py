import glob
import logging
import math
import os
import shutil
import sys
import time
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
from datetime import datetime
from dotenv import load_dotenv

import cv2
import torch
import typer
from torch.optim import SGD
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    decollate_batch,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,

)
from monai.utils import set_determinism

from src.utils.utils import create_transform
from src.utils.wandb_logger import WandBLogger
import wandb


def run(cfg):
    data_dir = os.path.join(cfg['data_root'], cfg['task'])

    images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz"))
    )
    labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz"))
    )
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]

    val_split = cfg['val_split']
    num_val_samples = int(val_split * len(data_dicts))
    val_files = data_dicts[-num_val_samples:]

    set_track_meta(True)
    val_transforms = create_transform(cfg['transform']['val'])

    val_cache_rate = cfg['val_cache_rate']
    val_num_workers = cfg['val_num_workers']

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=val_cache_rate,
        num_workers=val_num_workers,
        copy_cache=False
    )

    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    device = torch.device(cfg['device'])
    num_classes = cfg['num_classes']

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        kernel_size=3,
        up_kernel_size=3,
        act=Act.PRELU,
        dropout=0.2,
        bias=True,
    )

    state_dict = torch.load(cfg['path_to_checkpoint'], map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)

    # avoid the computation of meta information in random transforms
    if val_cache_rate == 1:
        set_track_meta(False)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    do_log_to_wandb = cfg['do_log_to_wandb']
    if do_log_to_wandb:
        wandb_logger = WandBLogger(cfg, model, save_config=False)
        columns = ['эfilename', 'image', 'ground_truth', 'prediction']
        table = wandb.Table(columns=columns)
    else:
        wandb_logger = None
        table = None

    do_log_locally = cfg['do_log_locally']

    model.eval()
    with torch.no_grad():
        val_loader_iterator = iter(val_loader)

        # for i in range(len(val_loader)):
        for i in range(2):#len(val_loader)):
            val_data = next(val_loader_iterator)
            val_inputs, val_labels = (
                val_data['image'].to(device),
                val_data['label'].to(device),
            )

            roi_size = cfg['val_roi_size']
            sw_batch_size = cfg['sw_batch_size']

            # set AMP for MONAI validation
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model
                )

            val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels_post = [post_label(i) for i in decollate_batch(val_labels)]

            sample_metric = dice_metric(y_pred=val_outputs_post, y=val_labels_post)
            image_name = os.path.split(val_loader.dataset.data[i]['image'])[-1].split('.')[0]
            logging.info(f'Filename: {image_name} | Metric value: {round(sample_metric.item(), 5)}')
            if do_log_to_wandb or do_log_locally:
                num_slices = 20
                slice_idx_start = val_inputs.shape[-1] // 2 - num_slices // 2
                slice_idx_end = slice_idx_start + num_slices + 1

                for slice_idx in range(slice_idx_start, slice_idx_end):
                    image = val_inputs[0, 0, :, :, slice_idx]
                    label = val_labels[0, 0, :, :, slice_idx]
                    prediction = torch.argmax(
                        val_outputs, dim=1).detach().cpu()[0, :, :, slice_idx].to(torch.float32)

                    if do_log_to_wandb:
                        table.add_data(image_name, wandb.Image(image), wandb.Image(label), wandb.Image(prediction))

                    if do_log_locally:
                        os.makedirs(os.path.join(cfg['prediction_dir'], image_name), exist_ok=True)

                        np.save(
                            os.path.join(cfg['prediction_dir'], image_name, 'prediction.npy'),
                            np.asarray([x.cpu().numpy() for x in val_outputs_post])
                        )

                        image = torch.clamp(image.cpu() * 255, 0, 255).to(torch.uint8).numpy()
                        label = torch.clamp(label.cpu() * 255, 0, 255).to(torch.uint8).numpy()
                        prediction = torch.clamp(prediction.cpu() * 255, 0, 255).to(torch.uint8).numpy()

                        cv2.imwrite(
                            os.path.join(cfg['prediction_dir'], image_name, f'image_{str(slice_idx).zfill(3)}.png'),
                            image
                        )
                        cv2.imwrite(
                            os.path.join(cfg['prediction_dir'], image_name, f'label{str(slice_idx).zfill(3)}.png'),
                            label
                        )
                        cv2.imwrite(
                            os.path.join(
                                cfg['prediction_dir'], image_name, f'prediction_{str(slice_idx).zfill(3)}.png'
                            ),
                            prediction
                        )

        metric = dice_metric.aggregate().item()
        logging.info('-' * 15)
        logging.info(f'Mean metric value: {metric}')

    if do_log_to_wandb:
        wandb_logger.log({'val_predictions': table})
        wandb_logger.finish()


def main(config_name: str = typer.Option('eval_task09.yaml', metavar='--config-name')):
    load_dotenv()

    cfg_pth = os.path.join(os.environ['PROJECT_ROOT'], 'src', 'configs', config_name)
    cfg = OmegaConf.load(cfg_pth)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg['data_root'] = os.path.expanduser(cfg['data_root'])
    cfg['path_to_checkpoint'] = os.path.expanduser(cfg['path_to_checkpoint'])

    do_log_locally = cfg['do_log_locally']
    if do_log_locally:
        snapshot_dir = os.path.join(*os.path.split(cfg['path_to_checkpoint'])[:-1])
        prediction_dir = os.path.join(snapshot_dir, 'predictions')

        if os.path.exists(prediction_dir):
            print(f'Prediction dir for checkpoint {cfg["path_to_checkpoint"]} already exists. '
                  f'Do you want to erase it? [y/N]')
            to_erase = input().lower()
            if to_erase in ['y', 'yes']:
                shutil.rmtree(prediction_dir)
            else:

                now = datetime.now().strftime("%b%d_%H-%M-%S")
                prediction_dir = '_'.join([prediction_dir, now])

        os.makedirs(prediction_dir)
        cfg['prediction_dir'] = prediction_dir

    handlers = [logging.StreamHandler(sys.stdout)]
    if do_log_locally:
        handlers.append(logging.FileHandler(os.path.join(cfg['prediction_dir'], 'metrics.log')))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(message)s"
    )

    if do_log_locally:
        logging.info(f'Predictions will be saved to {cfg["prediction_dir"]} directory.')

    set_determinism(seed=42)
    run(cfg)


if __name__ == '__main__':
    typer.run(main)
