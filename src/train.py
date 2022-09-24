import glob
import logging
import math
import os
import shutil
import sys
import time
from typing import Dict, Optional

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
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,

)
from monai.utils import set_determinism

from src.utils.utils import create_transform, save_checkpoint
from src.utils.poly_lr_scheduler import PolynomialLRDecay
from src.utils.wandb_logger import WandBLogger


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

    loss_function = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=True,
        smooth_nr=0.00001,
        smooth_dr=0.00001,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )

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

    # avoid the computation of meta information in random transforms
    if train_cache_rate == 1. and val_cache_rate == 1:
        set_track_meta(False)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    num_epochs = cfg['num_epochs']
    val_interval = cfg['val_interval']
    lr = cfg['lr']

    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=0.95,
        weight_decay=0.00004,
    )

    scheduler = PolynomialLRDecay(
        optimizer,
        max_decay_steps=num_epochs,
        power=0.9
    )
    scaler = torch.cuda.amp.GradScaler()

    artefacts_dir = cfg['artefacts_dir']
    wandb_logger = WandBLogger(cfg, model, save_config=True)

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{num_epochs}")

        model.train()
        epoch_loss = 0
        train_loader_iterator = iter(train_loader)

        # using step instead of iterate through train_loader directly to track data loading time
        # steps are 1-indexed for printing and calculation purposes
        for step in range(1, len(train_loader) + 1):
            step_start = time.time()

            batch_data = next(train_loader_iterator)
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )

            optimizer.zero_grad()
            # set AMP for MONAI training
            # profiling: forward
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
            logging.info(
                f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            )
            wandb_logger.log_scalar('train/loss', loss.item())

        scheduler.step()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        wandb_logger.log_scalar('train/epoch_loss', epoch_loss)
        wandb_logger.log_scalar('train/lr', scheduler.get_lr()[0])

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loader_iterator = iter(val_loader)

                for i in range(len(val_loader)):
                    val_data = next(val_loader_iterator)
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
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

                    dice_metric(y_pred=val_outputs_post, y=val_labels_post)
                    if i < 5:
                        wandb_logger.log_slices('Val_images', val_inputs, val_outputs, val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1

                    save_checkpoint(model, epoch, artefacts_dir, optimizer)
                    logging.info('saved new best metric model')
                logging.info(
                    f'current epoch: {epoch + 1} current'
                    f' mean dice: {metric:.4f}'
                    f' best mean dice: {best_metric:.4f}'
                    f' at epoch: {best_metric_epoch}'
                )
                wandb_logger.log_scalar('val/mean_dice', metric)

        logging.info(
            f'time consuming of epoch {epoch + 1} is:'
            f' {(time.time() - epoch_start):.4f}'
        )
    wandb_logger.finish()


def main(config_name: str = typer.Option('train_task09.yaml', metavar='--config-name')):
    load_dotenv()

    cfg_pth = os.path.join(os.environ['PROJECT_ROOT'], 'src', 'configs', config_name)
    cfg = OmegaConf.load(cfg_pth)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    artefacts_dir = os.path.join(os.environ['PROJECT_ROOT'], 'artefacts', cfg['run_name'])
    if os.path.exists(artefacts_dir):
        print(f'Run with name "{cfg["run_name"]}" already exists. Do you want to erase it? [y/N]')
        to_erase = input().lower()
        if to_erase in ['y', 'yes']:
            shutil.rmtree(artefacts_dir)
        else:

            now = datetime.now().strftime("%b%d_%H-%M-%S")
            artefacts_dir = '_'.join([artefacts_dir, now])

    os.makedirs(os.path.join(artefacts_dir, 'logs'))
    os.makedirs(os.path.join(artefacts_dir, 'snapshots'))
    os.makedirs(os.path.join(artefacts_dir, 'tb'))

    cfg['data_root'] = os.path.expanduser(cfg['data_root'])
    cfg['artefacts_dir'] = artefacts_dir
    OmegaConf.save(cfg, os.path.join(artefacts_dir, 'train_config.yaml'))

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
