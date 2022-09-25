import glob
import logging
import math
import os
import shutil
import sys
import time
from typing import Dict, Optional

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from datetime import datetime
from dotenv import load_dotenv

import torch
from torch.optim import Optimizer
import typer
from torch.optim import SGD
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    decollate_batch,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
)
from monai.utils import set_determinism

from src.utils.utils import create_transform, load_checkpoint
from src.utils.cross_entropy_loss import CrossEntropyLossAdv
from src.utils.wandb_logger import WandBLogger


def get_adversarial_input(inputs, xi):
    inputs_adv = []
    for input_ in inputs:
        min_, max_ = input_.min(), input_.max()
        inputs_adv.append(
            torch.clamp(input_ + xi, min_, max_).unsqueeze(0)
        )

    return torch.cat(inputs_adv, dim=0)


def get_adversarial_label(labels, strategy: str = 'zeros'):
    if strategy == 'zeros':
        labels_adv = torch.zeros_like(labels, device=labels.device)
    else:
        raise ValueError

    return labels_adv


def run(cfg):
    data_dir = os.path.join(cfg['data_root'], cfg['task'])

    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz"))
    )
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz"))
    )
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    num_val_samples = int(cfg['val_split'] * len(data_dicts))
    train_files, val_files = data_dicts[:-num_val_samples], data_dicts[-num_val_samples:]

    set_track_meta(True)
    train_transforms = create_transform(cfg['transform']['train'])
    val_transforms = create_transform(cfg['transform']['val'])

    train_cache_rate, val_cache_rate = cfg['train_cache_rate'], cfg['val_cache_rate']
    train_num_workers, val_num_workers = cfg['train_num_workers'], cfg['val_num_workers']

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=train_cache_rate,
        num_workers=train_num_workers,
        copy_cache=False,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=val_cache_rate,
        num_workers=val_num_workers,
        copy_cache=False
    )

    batch_size = cfg['batch_size']
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

    loss_function = CrossEntropyLossAdv()

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
    ).to(device)

    path_to_checkpoint = cfg['path_to_checkpoint']
    load_checkpoint(model, path_to_checkpoint, device)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    num_epochs = cfg['num_epochs']
    val_interval = cfg['val_interval']

    # wandb_logger = WandBLogger(cfg, model, save_config=True)

    artefacts_dir = cfg['artefacts_dir']
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    xi_norm = cfg['xi_norm']
    alpha = cfg['alpha']
    xi = torch.zeros((1, 96, 96, 96), dtype=torch.float32, device=device)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{num_epochs}")

        model.eval()
        epoch_loss = 0

        inputs_grad = 0.
        for step, batch_data in enumerate(train_loader, 1):
            step_start = time.time()
            inputs, labels = (
                batch_data['image'].to(device),
                batch_data['label'].to(device),
            )

            inputs.requires_grad = True
            inputs_adv = get_adversarial_input(inputs, xi)
            labels_adv = get_adversarial_label(labels)
            outputs = model(inputs_adv)

            loss = loss_function(outputs, labels_adv)
            model.zero_grad()
            loss.backward()

            inputs_grad += inputs.grad.data

            epoch_loss += loss.item()
            epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
            logging.info(
                f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            )
            # wandb_logger.log_scalar('train/loss', loss.item())

        inputs_grad = inputs_grad.sum(0)
        xi = torch.clamp(xi - alpha * torch.sign(inputs_grad), -xi_norm, xi_norm)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        # wandb_logger.log_scalar('train/epoch_loss', epoch_loss)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, val_data in enumerate(val_loader, 1):
                    val_inputs, val_labels = (
                        val_data['image'].to(device),
                        val_data['label'].to(device),
                    )

                    val_inputs_adv = get_adversarial_input(val_inputs, xi)
                    val_labels_adv = get_adversarial_label(val_labels)

                    roi_size = cfg['val_roi_size']
                    sw_batch_size = cfg['sw_batch_size']

                    val_outputs = sliding_window_inference(
                        val_inputs_adv, roi_size, sw_batch_size, model
                    )

                    val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_post = [post_label(i) for i in decollate_batch(val_labels_adv)]

                    dice_metric(y_pred=val_outputs_post, y=val_labels_post)
                    # if i < 5:
                    #     wandb_logger.log_slices('Val_images', val_inputs, val_outputs, val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(xi, os.path.join(artefacts_dir, 'best_perturbation.pt'))
                    # save_checkpoint(model, epoch, artefacts_dir, optimizer)
                    logging.info('saved new best perturbation')
                logging.info(
                    f'current epoch: {epoch + 1} current'
                    f' mean dice: {metric:.4f}'
                    f' best mean dice: {best_metric:.4f}'
                    f' at epoch: {best_metric_epoch}'
                )
                # wandb_logger.log_scalar('val/mean_dice', metric)

        logging.info(
            f'time consuming of epoch {epoch + 1} is:'
            f' {(time.time() - epoch_start):.4f}'
        )
    # wandb_logger.finish()


def main(config_name: str = typer.Option('attack_task09.yaml', metavar='--config-name')):
    load_dotenv()

    cfg_pth = os.path.join(os.environ['PROJECT_ROOT'], 'src', 'configs', config_name)
    cfg = OmegaConf.load(cfg_pth)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    artefacts_dir = os.path.join(os.environ['PROJECT_ROOT'], 'artefacts', cfg['run_name'])
    if os.path.exists(artefacts_dir):
        print(f'Attack with name "{cfg["run_name"]}" already exists. Do you want to erase it? [y/N]')
        to_erase = input().lower()
        if to_erase in ['y', 'yes']:
            shutil.rmtree(artefacts_dir)
        else:

            now = datetime.now().strftime("%b%d_%H-%M-%S")
            artefacts_dir = '_'.join([artefacts_dir, now])

    os.makedirs(os.path.join(artefacts_dir, 'logs'))
    os.makedirs(os.path.join(artefacts_dir, 'snapshots'))

    cfg['data_root'] = os.path.expanduser(cfg['data_root'])
    cfg['artefacts_dir'] = artefacts_dir
    OmegaConf.save(cfg, os.path.join(artefacts_dir, 'attack_config.yaml'))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg['artefacts_dir'], 'logs', 'figures.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f'Run artefacts will be saved to {cfg["artefacts_dir"]} directory.')

    set_determinism(seed=42)
    run(cfg)

if __name__ == '__main__':
    typer.run(main)
