from kedro.io import AbstractDataSet
from icecream import ic
from torchvision.datasets import ImageFolder


class KaggleDataSet(AbstractDataSet):
    def __init__(self, filepath: str, save_args):
        ic(filepath, save_args)
        self.filepath = filepath
        self.save_args = save_args

    def _load(self) -> ImageFolder:
        # TODO return ImageFolder from self.filepath
        raise NotImplementedError

    def _save(self, data) -> None:
        # TODO save data to self.filepath
        ic(data)
        raise NotImplementedError("TODO")

    def _describe(self):
        return dict(filepath=self.filepath)
