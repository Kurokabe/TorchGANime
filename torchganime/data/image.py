import os
from glob import glob
from typing import List, Union

import pytorch_lightning as pl
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.image_paths = sorted(glob(os.path.join(root, "**/*.png")))
        self.image_paths = random.sample(self.image_paths, len(self.image_paths))
        # self.image_paths = sorted(glob(os.path.join(root, "**/*.png")))
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        return self.transform(image)

    def __len__(self):
        return len(self.image_paths)


class ImageData(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        image_size: Union[int, List[int]] = 256,
        batch_size: int = 8,
        num_workers: int = 16,
    ):
        super().__init__()
        self.path = path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        self.train_dataset = ImageDataset(
            root=os.path.join(path, "train"),
            transform=self.make_transform(mode="train"),
        )
        self.val_dataset = ImageDataset(
            root=os.path.join(path, "val"),
            transform=self.make_transform(mode="val"),
        )

    def make_transform(self, mode="train"):
        if mode == "train":
            try:
                resized_shape = int(self.image_size * 1.2)
            except TypeError:
                resized_shape = tuple([int(x * 1.2) for x in self.image_size])

            return transforms.Compose(
                [
                    transforms.Resize(resized_shape),
                    transforms.RandomCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return dataloader


# DATASET_SIZE = 10000


# def identity(x):
#     return x


# class ImageData(pl.LightningDataModule):
#     def __init__(self, path, batch_size=64, workers=4, **kw):
#         super().__init__()
#         self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         self.training_urls = self.get_urls(path)
#         # TODO pretty print
#         print("training_urls = ", self.training_urls)
#         print("len training_urls = ", len(self.training_urls))
#         # self.val_urls = os.path.join(bucket, valshards)
#         # print("val_urls = ", self.val_urls)
#         self.val_urls = self.training_urls  # TODO replace with real val set
#         self.batch_size = batch_size
#         self.num_workers = workers
#         print("batch_size", self.batch_size, "num_workers", self.num_workers)

#     def make_transform(self, mode="train"):
#         if mode == "train":
#             return transforms.Compose(
#                 [
#                     transforms.Resize(294),
#                     transforms.RandomCrop((256, 512)),
#                     # transforms.RandomResizedCrop((256, 512)),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     self.normalize,
#                 ]
#             )
#         elif mode == "val":
#             return transforms.Compose(
#                 [
#                     transforms.Resize((256, 512)),
#                     transforms.ToTensor(),
#                     self.normalize,
#                 ]
#             )

#     def get_urls(self, path):
#         if isinstance(path, str):
#             path = [path]
#         paths = []
#         for p in path:
#             paths.extend(self._get_urls_from_path(p))
#         return paths

#     def _get_urls_from_path(self, path):
#         if os.path.isdir(path):
#             paths = sorted(glob(os.path.join(path, "*")))
#         else:
#             paths = path
#         return paths

#     def convert_tuple_title(self, input):
#         return f"{input[1]}:{input[0]}", input[2]

#     def make_loader(self, urls, mode="train"):

#         if mode == "train":
#             dataset_size = DATASET_SIZE
#             shuffle = 5000
#         elif mode == "val":
#             dataset_size = 5000
#             shuffle = 0

#         transform = self.make_transform(mode=mode)

#         dataset = (
#             wds.WebDataset(urls)
#             .shuffle(shuffle)
#             .decode("pil")
#             .to_tuple("__key__ __url__ jpg;png;jpeg")
#             .map(self.convert_tuple_title)
#             .map_tuple(identity, transform)
#             .batched(self.batch_size, partial=False)
#             .with_epoch(1000)
#         )

#         loader = wds.WebLoader(
#             dataset,
#             batch_size=None,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )

#         loader.length = dataset_size // self.batch_size

#         # if mode == "train":
#         #     # ensure same number of batches in all clients
#         #     loader = loader.ddp_equalize(dataset_size // self.batch_size)
#         #     # print("# loader length", len(loader))

#         return loader

#     def train_dataloader(self):
#         return self.make_loader(self.training_urls, mode="train")

#     def val_dataloader(self):
#         return self.make_loader(self.val_urls, mode="val")
