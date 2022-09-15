import logging
import os
import random
import shutil
import sys
import tempfile
from glob import glob
from typing import Dict
from omegaconf import OmegaConf
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
import typer
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.apps.datasets import DecathlonDataset
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    Spacingd
)
from monai.visualize import plot_2d_or_3d_image

from src.utils.utils import object_from_dict


def create_transform(cfg_transform: Dict):
    cfg_transform = OmegaConf.to_container(cfg_transform, resolve=True)

    transform = []
    for cfg_transform_fn in cfg_transform['transform_fns']:
        transform.append(object_from_dict(cfg_transform_fn))

    _Compose = object_from_dict(cfg_transform['compose_fn'])
    if _Compose is not None:
        transform = _Compose.__class__(transform)

    return transform


def run(cfg):
    monai.config.print_config()

    # define transforms for image and segmentation
    train_transforms = create_transform(cfg['transform']['train'])
    val_transforms = create_transform(cfg['transform']['val'])

    # create a training data loader
    train_ds = DecathlonDataset(
        cfg['data_dir'],
        task='Task06_Lung',
        section='training',
        transform=train_transforms,
        download=False,
        val_frac=0.1,
        cache_num=cfg['cache_num'])

    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=cfg['train_num_workers'],
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = DecathlonDataset(
        cfg['data_dir'],
        task='Task06_Lung',
        section='validation',
        transform=val_transforms,
        download=False,
        val_frac=0.1,
        cache_num=cfg['cache_num'])

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=cfg['val_num_workers'],
        collate_fn=list_data_collate
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = monai.losses.DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)

    # start a typical PyTorch training
    val_interval = cfg['val_interval']
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(log_dir=os.path.join(cfg['artefacts_dir'], 'tb'))

    for epoch in range(cfg['num_epochs']):
        logging.info('-' * 10)
        logging.info(f'epoch {epoch + 1}/{cfg["num_epochs"]}')
        model.train()
        epoch_loss = 0
        step = 0
        # for batch_data in train_loader:
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            logging.info(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                    roi_size = (192, 192, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    if epoch > 10:
                        torch.save(model.state_dict(), os.path.join(cfg['artefacts_dir'], 'snapshots', f'chk_{epoch}.pth'))
                        logging.info("saved new best metric model")
                logging.info(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    logging.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


def read_config(config_name: str):
    pth = os.path.join(os.environ['PROJECT_ROOT'], 'src', 'configs', config_name)
    cfg = OmegaConf.load(pth)
    cfg['data_dir'] = os.path.expanduser(cfg['data_dir'])
    return cfg


def main(config_name: str = typer.Option('train.yaml', metavar='--config_name')):
    load_dotenv()

    cfg_pth = os.path.join(os.environ['PROJECT_ROOT'], 'src', 'configs', config_name)
    cfg = OmegaConf.load(cfg_pth)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg['artefacts_dir'], 'logs', 'figures.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    artf_pth = os.path.join(os.environ['PROJECT_ROOT'], 'artefacts', cfg['run_name'])
    if os.path.exists(artf_pth):
        print(f'Run with name "{cfg["run_name"]}" already exists. Do you want to erase it? [yn]')
        to_erase = input()
        if to_erase == 'y':
            shutil.rmtree(artf_pth)
        else:
            artf_pth = '_'.join([artf_pth, str(random.randint(0, 1000))])

    logging.info(f'Run artefacts will be saved to {artf_pth}')
    os.makedirs(os.path.join(artf_pth, 'logs'))
    os.makedirs(os.path.join(artf_pth, 'snapshots'))
    os.makedirs(os.path.join(artf_pth, 'tb'))

    cfg['data_dir'] = os.path.expanduser(cfg['data_dir'])
    cfg['artefacts_dir'] = artf_pth
    OmegaConf.save(cfg, os.path.join(artf_pth, 'train_config.yaml'))

    run(cfg)


if __name__ == "__main__":
    typer.run(main)
