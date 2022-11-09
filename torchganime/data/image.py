import os
from glob import glob

import pytorch_lightning as pl
import torch
import webdataset as wds
from torchvision import transforms

DATASET_SIZE = 10000


def identity(x):
    return x


class ImageData(pl.LightningDataModule):
    def __init__(self, path, batch_size=64, workers=4, **kw):
        super().__init__()
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.training_urls = self.get_urls(path)
        print("training_urls = ", self.training_urls)
        print("len training_urls = ", len(self.training_urls))
        # self.val_urls = os.path.join(bucket, valshards)
        # print("val_urls = ", self.val_urls)
        self.batch_size = batch_size
        self.num_workers = workers
        print("batch_size", self.batch_size, "num_workers", self.num_workers)

    def make_transform(self, mode="train"):
        if mode == "train":
            return transforms.Compose(
                [
                    transforms.Resize(294),
                    transforms.RandomCrop((256, 512)),
                    # transforms.RandomResizedCrop((256, 512)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    transforms.Resize((256, 512)),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )

    def get_urls(self, path):
        if isinstance(path, str):
            path = [path]
        paths = []
        for p in path:
            paths.extend(self._get_urls_from_path(p))
        return paths

    def _get_urls_from_path(self, path):
        if os.path.isdir(path):
            paths = sorted(glob(os.path.join(path, "*")))
        else:
            paths = path
        return paths

    def make_loader(self, urls, mode="train"):

        if mode == "train":
            dataset_size = DATASET_SIZE
            shuffle = 5000
        elif mode == "val":
            dataset_size = 5000
            shuffle = 0

        transform = self.make_transform(mode=mode)

        dataset = (
            wds.WebDataset(urls)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("__key__ __url__ jpg;png;jpeg")
            .map(lambda x: (f"{x[1]}:{x[0]}", x[2]))
            .map_tuple(identity, transform)
            .batched(self.batch_size, partial=False)
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

        loader.length = dataset_size // self.batch_size

        # if mode == "train":
        #     # ensure same number of batches in all clients
        #     loader = loader.ddp_equalize(dataset_size // self.batch_size)
        #     # print("# loader length", len(loader))

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")
