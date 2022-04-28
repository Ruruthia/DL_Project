from kedro.io import AbstractDataSet
from torchvision.datasets import ImageFolder


class ImageFolderDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> ImageFolder:
        return ImageFolder(self._filepath)

    def _save(self, data) -> None:
        raise NotImplementedError("Not needed")

    def _describe(self):
        return dict(filepath=self._filepath)
