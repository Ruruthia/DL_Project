"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.17.7
"""
import shutil
import random
from pathlib import Path
from typing import Union, Iterable


def split(
        source: Union[str, Path],
        destination: Union[str, Path],
        test_ratio: float,
        val_ratio: float,
        seed: int,
) -> None:
    random.seed(seed)

    if isinstance(source, str):
        source = Path(source)
    if isinstance(destination, str):
        destination = Path(destination)

    for class_dir in source.iterdir():
        images = [file.name for file in class_dir.iterdir()]
        random.shuffle(images)
        no_test = int(len(images) * test_ratio)
        no_val = int(len(images) * val_ratio)

        copy_files(class_dir, destination / 'test' / class_dir.name, images[:no_test])
        copy_files(class_dir, destination / 'val' / class_dir.name, images[no_test: no_test + no_val])
        copy_files(class_dir, destination / 'train' / class_dir.name, images[no_test + no_val:])


def copy_files(source: Path, destination: Path, files: Iterable[str]):
    destination.mkdir(parents=True)
    for file in files:
        shutil.copyfile(source / file, destination / file)
