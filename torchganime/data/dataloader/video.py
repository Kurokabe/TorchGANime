from typing import List, Union
import os
import pytorch_lightning as pl

from scenedataset.scenedataset import SceneDataset

# from pytorchvideo import transforms as video_transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import _transforms_video as video_transforms
import torch


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollateVideo:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences. Returns a dictionary with keys:
        target - the padded tensor
        frame_number - the number of frames originally in each sequence (before padding)
        first_frame - the first frame of each sequence
        end_frame - the last frame of each sequence (before padding)
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of tensor

        return dictionary with keys:
            target - the padded tensor
            frame_number - the number of frames originally in each sequence (before padding)
            first_frame - the first frame of each sequence
            end_frame - the last frame of each sequence (before padding)
        """
        # find longest sequence
        frame_number = [b.shape[1] for b in batch]
        max_len = max(frame_number)
        first_frames = [b[:, 0, :, :] for b in batch]
        end_frames = [b[:, -1, :, :] for b in batch]

        # pad according to max_len
        batch = [pad_tensor(b, pad=max_len, dim=1) for b in batch]

        # stack all
        xs = torch.stack(batch, dim=0)
        frame_number = torch.tensor(frame_number)
        first_frames = torch.stack(first_frames, dim=0)
        end_frames = torch.stack(end_frames, dim=0)

        return {
            "frame_number": frame_number,
            "first_frame": first_frames,
            "end_frame": end_frames,
        }, xs

    def __call__(self, batch):
        return self.pad_collate(batch)


class CollateFirstLastFrame:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences. Returns a dictionary with keys:
        target - the padded tensor
        frame_number - the number of frames originally in each sequence (before padding)
        first_frame - the first frame of each sequence
        end_frame - the last frame of each sequence (before padding)
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of tensor

        return dictionary with keys:
            target - the padded tensor
            frame_number - the number of frames originally in each sequence (before padding)
            first_frame - the first frame of each sequence
            end_frame - the last frame of each sequence (before padding)
        """
        # find longest sequence
        frame_number = torch.tensor([b.shape[1] for b in batch])
        # first_frames = [b[:, 0, :, :] for b in batch]
        end_frames = [b[:, -1, :, :] for b in batch]

        sample_frame_number = frame_number - 1
        frames_to_keep = (
            torch.rand(len(sample_frame_number)) * sample_frame_number
        ).type(torch.int) + 1

        current_frames = [
            b[:, frames_to_keep[i] - 1, :, :] for i, b in enumerate(batch)
        ]

        # pad according to max_len
        # batch = [pad_tensor(b, pad=max_len, dim=1) for b in batch]
        batch = [b[:, frames_to_keep[i], :, :] for i, b in enumerate(batch)]

        # stack all
        target = torch.stack(batch, dim=0)
        current_frames = torch.stack(current_frames, dim=0)
        end_frames = torch.stack(end_frames, dim=0)

        return {
            # "frame_number": frame_number,
            "current_frames": current_frames,
            "end_frames": end_frames,
            "remaining_frames": frame_number - frames_to_keep - 1,
            # "frames_to_keep": frames_to_keep,
        }, target

    def __call__(self, batch):
        return self.pad_collate(batch)


class VideoData(pl.LightningDataModule):
    def __init__(
        self,
        train_paths: Union[List[str], str],
        val_paths: Union[List[str], str],
        image_size: Union[int, List[int]] = 256,
        batch_size: int = 8,
        num_workers: int = 16,
    ):
        super().__init__()
        self.prepare_data_per_node = False
        self.train_paths = self._validate_path(train_paths)
        self.val_paths = self._validate_path(val_paths)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = video_transforms.NormalizeVideo(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        self.train_dataset = SceneDataset(
            self.train_paths,
            self.make_transform(mode="train"),
            recursive=True,
            show_progress=True,
            min_max_len=(10, 25),
            detector="content",
            threshold=15,
            min_scene_len=15,
            duplicate_metric="lpips",
            duplicate_threshold=0.01,
        )

        self.val_dataset = SceneDataset(
            self.val_paths,
            self.make_transform(mode="val"),
            recursive=True,
            show_progress=True,
            min_max_len=(10, 25),
            detector="content",
            threshold=15,
            min_scene_len=15,
            duplicate_metric="lpips",
            duplicate_threshold=0.01,
        )

    def make_transform(self, mode="train"):
        if mode == "train":
            try:
                resized_shape = int(self.image_size * 1.2)
            except TypeError:
                resized_shape = tuple([int(x * 1.2) for x in self.image_size])

            # return transforms.Compose(
            #     [
            #         video_transforms.ToTensorVideo(),
            #         transforms.Resize(resized_shape),
            #         video_transforms.RandomCropVideo(self.image_size),
            #         video_transforms.RandomHorizontalFlipVideo(),
            #         self.normalize,
            #     ]
            # )
            return transforms.Compose(
                [
                    video_transforms.ToTensorVideo(),
                    transforms.Resize(self.image_size),
                    self.normalize,
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    video_transforms.ToTensorVideo(),
                    transforms.Resize(self.image_size),
                    self.normalize,
                ]
            )

    def _validate_path(self, path: Union[List[str], str]):
        if isinstance(path, str):
            path = [path]
        # for p in path:
        #     if not os.path.exists(p):
        #         raise ValueError(f"The provided path {p} does not exist")
        return path

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=CollateFirstLastFrame(dim=0),
            # collate_fn=PadCollateVideo(dim=0),
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
            collate_fn=CollateFirstLastFrame(dim=0),
            # collate_fn=PadCollateVideo(dim=0),
        )
        return dataloader
