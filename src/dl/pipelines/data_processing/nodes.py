"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

import random
from copy import copy
from typing import Dict
from typing import List

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def split(
        sharks_dataset: ImageFolder,
        test_ratio: float,
        val_ratio: float,
        seed: int,
) -> Dict[str, Subset]:
    random.seed(seed)

    train_indices, val_test_indices, _, val_test_targets = train_test_split(
        range(len(sharks_dataset)),
        sharks_dataset.targets,
        stratify=sharks_dataset.targets,
        test_size=test_ratio + val_ratio,
        random_state=seed
    )

    val_indices, test_indices, _, _ = train_test_split(
        val_test_indices,
        val_test_targets,
        stratify=val_test_targets,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )

    train_sharks_subset = Subset(sharks_dataset, train_indices)
    val_sharks_subset = Subset(sharks_dataset, val_indices)
    val_sharks_subset.dataset = copy(sharks_dataset)
    test_sharks_subset = Subset(sharks_dataset, test_indices)
    test_sharks_subset.dataset = copy(sharks_dataset)

    return {"train": train_sharks_subset, "val": val_sharks_subset, "test": test_sharks_subset}


def process_data(
        sharks_data_subsets: Dict[str, Subset],
        batch_size: int,
) -> List[DataLoader]:

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomRotation(25),  # randomly rotate images by 25 degrees
        transforms.RandomHorizontalFlip(),  # randomly flip images horizontally
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]),

    data_transforms = {
        'train': train_transforms,
        'test': test_transforms,
        'val': test_transforms,
    }

    for data_set in ('train', 'test', 'val'):
        sharks_data_subsets[data_set].dataset.transform = data_transforms[data_set]

    data_loaders = [
        DataLoader(
            sharks_data_subsets[data_set],
            batch_size=batch_size,
            shuffle=True,
        )
        for data_set in ('train', 'test', 'val')
    ]

    return data_loaders
