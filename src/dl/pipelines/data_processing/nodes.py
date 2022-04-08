"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder



def process_data(
        source: Union[str, Path],
        batch_size: int,
) -> List[DataLoader]:

    if isinstance(source, str):
        source = Path(source)

    train_transform = transforms.Compose([
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
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    data_transforms = {
        'train': train_transform,
        'test': transform,
        'val': transform
    }

    data_loaders = [
        DataLoader(
            ImageFolder(root=str(source / data_set), transform=data_transforms[data_set]),
            batch_size=batch_size,
            shuffle=True,
        )
        for data_set in ('train', 'test', 'val')
    ]
    return data_loaders
