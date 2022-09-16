
import glob
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import time

import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from monai.apps import download_and_extract
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
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    FgBgToIndicesd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ScaleIntensityd,
    Lambdad
)
from monai.utils import set_determinism


# resource https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_Brain.tar
data_root = '/root/data/Task01_Brain'
outputs_dir = os.path.join(os.environ['PROJECT_ROOT'], 'artefacts', 'fast_run_brain_0')
if os.path.exists(outputs_dir):
    print(f'Run with name "fast_run_brain_0" already exists. Do you want to erase it? [y/N]')
    to_erase = input().lower()
    if to_erase in ['y', 'yes']:
        shutil.rmtree(outputs_dir)
    else:
        artf_pth = '_'.join([outputs_dir, str(random.randint(0, 1000))])

os.makedirs(os.path.join(outputs_dir, 'logs'))
os.makedirs(os.path.join(outputs_dir, 'snapshots'))
os.makedirs(os.path.join(outputs_dir, 'tb'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(outputs_dir, 'logs', 'figures.log')),
        logging.StreamHandler(sys.stdout)
    ]
)


max_epochs = 600


train_images = sorted(
    glob.glob(os.path.join(data_root, "imagesTr", "*.nii.gz"))
)
train_labels = sorted(
    glob.glob(os.path.join(data_root, "labelsTr", "*.nii.gz"))
)
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]


def transformations(device='cuda:0'):
    train_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=['image', 'label'], func=lambda x: x[0]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1., 1., 1.),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityd(
            keys=["image"],
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # pre-compute foreground and background indexes
        # and cache them to accelerate training
        FgBgToIndicesd(
            keys="label",
            fg_postfix="_fg",
            bg_postfix="_bg",
            image_key="image",
        ),
        # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
        EnsureTyped(
            keys=["image", "label"], device=device, track_meta=False
        ),
        # randomly crop out patch samples from big
        # image based on pos / neg ratio
        # the image centers of negative samples
        # must be in valid image area
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=4,
            fg_indices_key="label_fg",
            bg_indices_key="label_bg",
        )
    ]

    val_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=['image', 'label'], func=lambda x: x[0]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1., 1., 1.),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityd(
            keys=["image"],
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False)

    ]

    return Compose(train_transforms), Compose(val_transforms)


def train_process():
    learning_rate = 2e-4
    val_interval = 10  # do validation for every epoch
    set_track_meta(True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise RuntimeError('this tutorial is intended for GPU, but no CUDA device is available')

    train_trans, val_trans = transformations(device=device)
    # set CacheDataset, ThreadDataLoader and DiceCE loss for MONAI fast training
    # as `RandCropByPosNegLabeld` crops from the cached content and `deepcopy`
    # the crop area instead of modifying the cached value, we can set `copy_cache=False`
    # to avoid unnecessary deepcopy of cached content in `CacheDataset`
    train_ds = CacheDataset(
        data=train_files,
        transform=train_trans,
        cache_rate=1.0,
        num_workers=8,
        copy_cache=False,
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_trans, cache_rate=1.0, num_workers=5, copy_cache=False
    )
    # disable multi-workers because `ThreadDataLoader` works with multi-threads
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

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
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
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
    set_track_meta(False)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=4)])
    post_label = Compose([AsDiscrete(to_onehot=4)])

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # SGD prefer to much bigger learning rate
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate * 1000,
        momentum=0.9,
        weight_decay=0.00004,
    )
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir=os.path.join(outputs_dir, 'tb'))
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    epoch_times = []
    total_start = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{max_epochs}")

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
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loader_iterator = iter(val_loader)

                for _ in range(len(val_loader)):
                    val_data = next(val_loader_iterator)
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )

                    roi_size = (160, 160, 160)
                    sw_batch_size = 4

                    # set AMP for MONAI validation
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, model
                        )

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(
                        time.time() - total_start
                    )
                    torch.save(model.state_dict(), os.path.join(outputs_dir, 'snapshots', 'best_metric_model.pt'))
                    logging.info("saved new best metric model")
                logging.info(
                    f"current epoch: {epoch + 1} current"
                    f" mean dice: {metric:.4f}"
                    f" best mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)

        logging.info(
            f"time consuming of epoch {epoch + 1} is:"
            f" {(time.time() - epoch_start):.4f}"
        )
        epoch_times.append(time.time() - epoch_start)

    total_time = time.time() - total_start
    logging.info(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
        f" total time: {total_time:.4f}"
    )
    return (
        max_epochs,
        epoch_loss_values,
        metric_values,
        epoch_times,
        best_metrics_epochs_and_time,
        total_time,
    )


set_determinism(seed=0)
monai_start = time.time()
(
    epoch_num,
    m_epoch_loss_values,
    m_metric_values,
    m_epoch_times,
    m_best,
    m_train_time,
) = train_process()
m_total_time = time.time() - monai_start
logging.info(
    f"total time of {epoch_num} epochs with MONAI fast training: {m_train_time:.4f},"
    f" time of preparing cache: {(m_total_time - m_train_time):.4f}"
)