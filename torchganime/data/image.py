import os
from glob import glob
from typing import List, Union

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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

