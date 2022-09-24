import os

import numpy as np
from tqdm import tqdm
from monai.apps.datasets import DecathlonDataset
from monai.data import DatasetSummary
from monai.transforms import (
    Compose,
    LoadImaged,
)


def main(root_dir: str, task: str, strategy: str):
    train_dataset = DecathlonDataset(
        root_dir=root_dir,
        task=task,
        section='training',
        transform=Compose([LoadImaged(keys=['image', 'label'])]),
        cache_rate=0.,
        val_frac=0.
    )

    if strategy == 'monai':
        dataset_summary = DatasetSummary(train_dataset)
        tgt_spacing = dataset_summary.get_target_spacing()
        print(f'Target spacing: {tgt_spacing};')

    elif strategy == 'custom':
        save_dir = os.path.join(*os.path.split(root_dir)[:-1], 'processed', task)

        train_pixdims = np.empty((len(train_dataset), 8), dtype=float)
        for i in tqdm(range(len(train_dataset))):
            sample = train_dataset[i]
            train_pixdims[i] = sample['image'].meta['pixdim']

        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'train_pixdims.npy'), train_pixdims)
        print(f'Target spacing: {np.median(train_pixdims, axis=0)};')

    else:
        raise ValueError


if __name__ == '__main__':
    root_dir = '/home/lqrhy3/PycharmProjects/coursework/data/raw'
    task = 'Task09_Spleen'
    strategy = 'monai'  # custom or monai

    main(root_dir, task, strategy)
