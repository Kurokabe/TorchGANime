from typing import Optional, Callable, Dict, Literal
import os
import ffmpeg
from typing import List, Union
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from decord import VideoReader, cpu
from glob import glob
from loguru import logger
from scenedetect import SceneManager, open_video
import json
import dataclasses
from dataclasses import dataclass
import decord
import torch

decord.bridge.set_bridge("torch")


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Scene:
    start: int
    end: int
    video_path: str

    def __len__(self):
        return self.end - self.start

    def __repr__(self):
        return f"Scene({self.start}, {self.end}, {self.video_path})"

    def __str__(self):
        return f"Scene({self.start}, {self.end}, {self.video_path})"


class SceneDataset(Dataset):
    scene_list_file = "mapping_video_path_to_scene_file.json"

    def __init__(
        self,
        paths: List[str],
        transform: Optional[Callable] = None,
        recursive: bool = False,
        show_progress: bool = False,
        detector: Literal["content", "threshold", "adaptive"] = "content",
        # TODO: add min/max scene length
        **kwargs,
    ):
        self.show_progress = show_progress
        self.video_paths = self._get_video_paths(paths, recursive)
        self.transform = transform

        self.detector = self.load_detector(detector, **kwargs)

        # This will be used to store inside the user home folder files containing the scene spliting
        self.scene_list_dir = self.get_scene_list_dir(
            detector_params={"detector": detector, **kwargs}
        )

        self.mapping_video_path_to_scene_file: Dict[
            str, str
        ] = self.retrieve_video_to_scene_list(self.scene_list_dir)

        self.scenes = self.retrieve_scenes(self.video_paths)
        self.update_mapping()

    def load_detector(
        self, detector: Literal["content", "threshold", "adaptive"], **kwargs
    ):
        if detector == "content":
            from scenedetect import ContentDetector

            return ContentDetector(**kwargs)
        elif detector == "threshold":
            from scenedetect import ThresholdDetector

            return ThresholdDetector(**kwargs)
        elif detector == "adaptive":
            from scenedetect import AdaptiveDetector

            return AdaptiveDetector(**kwargs)

    def get_scene_list_dir(self, detector_params: Dict):
        scene_list_dir = os.path.join(
            os.path.expanduser("~"),
            ".scene_dataset",
            "scene_list",
            "_".join(str(val) for val in detector_params.values()),
        )
        os.makedirs(scene_list_dir, exist_ok=True)
        return scene_list_dir

    def retrieve_scenes(
        self,
        video_paths: List[str],
    ) -> List[Scene]:
        scenes = []
        for video_path in video_paths:
            scenes.extend(self.retrieve_scenes_from_video(video_path))
        return scenes

    def retrieve_scenes_from_video(
        self,
        video_path: str,
    ) -> List[Scene]:
        if video_path in self.mapping_video_path_to_scene_file:
            scenes = self.load_precomputed_scenes(
                self.mapping_video_path_to_scene_file[video_path]
            )
        else:
            scenes = [
                Scene(
                    start=scene[0].get_frames(),
                    end=scene[1].get_frames(),
                    video_path=video_path,
                )
                for scene in self.detect_scenes(video_path)
            ]
            save_path = os.path.join(
                self.scene_list_dir, f"{video_path.replace('/', '_')}.json"
            )
            self.save_scenes(scenes, save_path)
            self.mapping_video_path_to_scene_file[video_path] = save_path
        return scenes

    def save_scenes(self, scenes: List[Scene], save_path: str):
        with open(save_path, "w") as f:
            json.dump(scenes, f, cls=EnhancedJSONEncoder)

    def update_mapping(self):
        with open(
            os.path.join(self.scene_list_dir, SceneDataset.scene_list_file), "w"
        ) as f:
            json.dump(self.mapping_video_path_to_scene_file, f)

    def load_precomputed_scenes(self, scene_file: str) -> List[Scene]:
        with open(scene_file, "r") as f:
            scenes = json.load(f)

        scenes = [Scene(**scene) for scene in scenes]
        return scenes

    def detect_scenes(
        self,
        video_path: str,
    ):

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(self.detector)
        # Detect all scenes in video from current position to end.
        scene_manager.detect_scenes(video, show_progress=self.show_progress)
        # `get_scene_list` returns a list of start/end timecode pairs
        # for each scene that was found.
        return scene_manager.get_scene_list()

    def retrieve_video_to_scene_list(self, root: str) -> Dict[str, str]:
        file_path = os.path.join(root, SceneDataset.scene_list_file)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                video_to_scene_list = json.load(f)
        else:
            video_to_scene_list = {}
        return video_to_scene_list

    def _get_video_paths(self, paths: List[str], recursive: bool) -> List[str]:
        video_paths = []
        for path in paths:
            if os.path.isfile(path):
                # File
                if self.check_if_video(path):
                    video_paths.append(path)
            else:
                logger.info(f"Finding video files inside {path} ...")
                # Folder
                for file_path in glob(os.path.join(path, "**"), recursive=recursive):
                    if os.path.isfile(file_path):
                        if self.check_if_video(file_path):
                            video_paths.append(file_path)

        video_paths = list(map(os.path.abspath, video_paths))
        logger.info(f"Found {len(video_paths)} videos")
        return video_paths

    def check_if_video(self, path: str) -> bool:
        metadata = self.get_metadata(path)
        return metadata["codec_type"] == "video"

    def get_metadata(self, path: str) -> dict:
        return ffmpeg.probe(path, select_streams="v")["streams"][0]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int):
        scene = self.scenes[idx]
        vr = VideoReader(scene.video_path)
        frames = vr.get_batch(range(scene.start, scene.end))
        frames = frames.to(dtype=torch.float) / 255
        if self.transform:
            frames = self.transform(frames)
        return frames


# class VideoData(pl.LightningDataModule):
#     def __init__(
#         self,
#         train_paths: Union[List[str], str],
#         val_paths: Union[List[str], str],
#         image_size: Union[int, List[int]] = 256,
#         batch_size: int = 8,
#         num_workers: int = 16,
#     ):
#         self.train_paths = self._validate_path(train_paths)
#         self.val_paths = self._validate_path(val_paths)
#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def _validate_path(self, path: Union[List[str], str]):
#         if path.isinstance(str):
#             path = [path]
#         if not os.path.exists(path):
#             raise ValueError(f"The provided path {path} does not exist")

#         return path
