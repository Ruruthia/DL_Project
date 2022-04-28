"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from typing import Dict, List

from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def process_data(
        sharks_data_subsets: Dict[str, Subset],
        batch_size: int,
) -> List[DataLoader]:

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomRotation(25),  # randomly rotate images by 25 degrees
            transforms.RandomHorizontalFlip(),  # randomly flip images horizontally
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]),
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
