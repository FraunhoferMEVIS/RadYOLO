from __future__ import annotations

import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Sized
from pathlib import Path
from typing import ClassVar, Literal, TypedDict

import cv2
import nibabel as nib
import nibabel.filebasedimages
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset
from tqdm import tqdm

from ultralytics.utils.ops import xyxy2xywh

LOGGER = logging.getLogger(__name__)
# IMG_FORMATS = {"jpg", "jpeg", "png"}


class BaseDataset(Dataset, Sized, ABC):
    img_formats: ClassVar[set[str]]

    def __init__(
        self,
        src_path: Path | list[Path],
        imgsz: tuple[int, int] | tuple[int, int, int],
        augment: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        prefix: str = "",
        fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.src_path = src_path
        self.imgsz = imgsz
        self.augment = augment
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.prefix = prefix
        self.fraction = fraction

        self.img_paths: list[Path] = self.get_img_files(self.src_path)
        self.samples = self.get_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Label:
        return self.load_image_and_label(index)

    def get_img_files(self, src_path: Path | list[Path]) -> list[Path]:
        try:
            f: list[Path] = []  # image files
            for p in src_path if isinstance(src_path, list) else [src_path]:
                if p.is_dir():  # dir
                    f = list(p.rglob("*.*"))
                elif p.is_file():  # file
                    with p.open() as t:
                        t_ = t.read().strip().splitlines()
                        f.extend(p.parent / x.lstrip(os.sep) for x in t_)
                else:
                    msg = f"{self.prefix}{p} does not exist"
                    raise FileNotFoundError(msg)
            im_files = sorted([x for x in f if x.suffix[1:].lower() in self.img_formats])  # pathlib
            assert im_files, f"{self.prefix}No images found in {src_path}."
        except Exception as e:
            msg = f"{self.prefix}Error loading data from {src_path}"
            raise FileNotFoundError(msg) from e
        if self.fraction < 1:
            im_files = im_files[
                : round(len(im_files) * self.fraction)
            ]  # retain a fraction of the dataset
        return im_files

    def load_image_and_label(self, index: int) -> Label:
        label = self.samples[index]
        img, original_shape, resized_shape = self.load_image(label["im_file"])
        return Label(
            im_file=label["im_file"],
            shape=resized_shape,
            classes=label["classes"],
            bboxes=label["bboxes"],
            normalized=label["normalized"],
            bbox_format=label["bbox_format"],
            img=img,
        )

    @abstractmethod
    def get_samples(self) -> list[LazyLabel]: ...

    @abstractmethod
    def load_image(
        self, f: Path
    ) -> tuple[
        torch.Tensor,
        tuple[int, int] | tuple[int, int, int],
        tuple[int, int] | tuple[int, int, int],
    ]: ...

    @staticmethod
    def collate_fn(batch: list[Label]) -> dict:
        new_batch = {}

        with torch.no_grad():
            new_batch["img"] = torch.stack([item["img"] for item in batch])
            new_batch["im_file"] = tuple(item["im_file"] for item in batch)
            new_batch["shape"] = tuple(item["shape"] for item in batch)
            new_batch["cls"] = torch.cat([item["classes"] for item in batch], 0)
            new_batch["bboxes"] = torch.cat([item["bboxes"] for item in batch], 0)
            new_batch["normalized"] = tuple(item["normalized"] for item in batch)
            new_batch["bbox_format"] = tuple(item["bbox_format"] for item in batch)
            new_batch["batch_idx"] = torch.cat(
                [torch.full(item["classes"].size()[:1], idx) for idx, item in enumerate(batch)], 0
            )

        return new_batch


class MIPDataset(BaseDataset):
    img_formats: ClassVar[set[str]] = {"png", "jpg", "jpeg"}

    def load_image(
        self,
        # i: int,
        f: Path,
        # rect_mode: bool = True,
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        # f = self.img_paths[i]
        if not f.is_file():
            raise FileNotFoundError(f"Image Not Found {f}")
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)[..., None]  # BGR
        if im is None:
            raise ValueError
        im_t = torch.as_tensor(im, dtype=torch.float32).movedim(-1, 0).contiguous() / 255.0
        im_t.clamp_(0.0, 1.0)
        im_t_scaled = F.interpolate(im_t[None, ...], size=(256, 448), mode="bilinear")[0]

        h0: int
        w0: int
        h0, w0 = im.shape[:2]  # orig hw
        # if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
        #     raise NotImplementedError
        #     r = self.imgsz / max(h0, w0)  # ratio
        #     if r != 1:  # if sizes are not equal
        #         w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
        #         im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        # elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
        #     raise NotImplementedError
        #     im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # Add to buffer if training with augmentations
        if self.augment:
            raise NotImplementedError
            self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                im,
                (h0, w0),
                im.shape[:2],
            )  # im, hw_original, hw_resized
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer.pop(0)
                # if self.cache != "ram":
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im_t_scaled, (h0, w0), im_t_scaled.shape[1:]

    def get_samples(self) -> list[LazyLabel]:
        # find all images and corresponding labels
        self.label_paths = [
            Path(*("labels" if part == "images" else part for part in path.parts)).with_suffix(
                ".txt"
            )
            for path in self.img_paths
        ]
        samples: list[LazyLabel] = []
        for img_path, label_path in tqdm(
            zip(self.img_paths, self.label_paths, strict=True),
            desc=f"Scanning {self.src_path}...",
            total=len(self.img_paths),
        ):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)[..., None]
            if img is None:
                LOGGER.warning("An error occurred loading image, skipping: %s", img_path)
                continue
            if not label_path.is_file():
                LOGGER.warning("label path does not exist, skipping: %s", label_path)
                continue
            with label_path.open() as f:
                lines = f.read().strip().splitlines()
            if lines:
                string_labels = [line.split() for line in lines if len(line)]
                if any(len(label) not in (0, 7) for label in string_labels):
                    LOGGER.warning("label not in expected 3D-Format, skipping: %s", label_path)
                    continue
                labels = np.array(string_labels, dtype=np.float32)
                labels = labels[:, [0, 1, 2, 4, 5]]  # remove z and depth-axis
            else:
                labels = np.zeros((0, 5))
            samples.append(
                LazyLabel(
                    im_file=img_path,
                    shape=img.shape,
                    classes=torch.as_tensor(labels[:, 0]),
                    bboxes=torch.as_tensor(labels[:, 1:]),
                    normalized=True,
                    bbox_format="xywh",
                )
            )
        # optionally verify all images
        return samples


class NifTIDataset(BaseDataset):
    img_formats: ClassVar[set[str]] = {"nii"}

    def get_samples(self) -> list[LazyLabel]:
        # find all images and corresponding labels
        self.label_paths = [
            Path(*("labels" if part == "images" else part for part in path.parts)).with_suffix(
                ".txt"
            )
            for path in self.img_paths
        ]
        samples: list[LazyLabel] = []
        for img_path, label_path in tqdm(
            zip(self.img_paths, self.label_paths, strict=True),
            desc=f"Scanning {self.src_path}...",
            total=len(self.img_paths),
        ):
            try:
                img = nib.load(img_path)
            except (FileNotFoundError, nibabel.filebasedimages.ImageFileError):
                LOGGER.warning("An error occurred loading image, skipping: %s", img_path)
                continue
            if not label_path.is_file():
                LOGGER.warning("label path does not exist, skipping: %s", label_path)
                continue
            with label_path.open() as f:
                lines = f.read().strip().splitlines()
            if lines:
                string_labels = [line.split() for line in lines if len(line)]
                if any(len(label) not in (0, 7) for label in string_labels):
                    LOGGER.warning("label not in expected 3D-Format, skipping: %s", label_path)
                    continue
                labels = np.array(string_labels, dtype=np.float32)
            else:
                labels = np.zeros((0, 7))
            samples.append(
                LazyLabel(
                    im_file=img_path,
                    shape=img.shape[:-1],
                    classes=torch.as_tensor(labels[:, 0]),
                    bboxes=torch.as_tensor(labels[:, 1:]),
                    normalized=True,
                    bbox_format="xywh",
                )
            )
        # optionally verify all images
        return samples

    def load_image(
        self, f: Path
    ) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        if not f.is_file():
            raise FileNotFoundError(f"Image Not Found {f}")
        im = nib.load(f)
        if not isinstance(im, nibabel.nifti1.Nifti1Image):
            raise TypeError
        im_t = torch.as_tensor(
            im.get_fdata(caching="unchanged", dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).movedim(-1, 0)
        im_t.clamp_(0.0, 1.0)
        im_t_scaled = F.interpolate(im_t[None, ...], self.imgsz, mode="trilinear")[0]

        return im_t_scaled, im.shape[1:], im_t_scaled.shape[1:]


class DummyDataset(Dataset, Sized):
    def __init__(
        self,
        n: int,
        num_classes: int,
        shape: tuple[int, int] | tuple[int, int, int],
        *,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._n: int = n
        self._num_classes: int = num_classes
        self._seed: int = seed
        self._shape: tuple[int, int] | tuple[int, int, int] = shape
        self._rng = np.random.default_rng(self._seed)
        self._data: list[Label] = self._regenerate()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Label:
        return self._data[index]

    def shuffle(self) -> None:
        self._regenerate()

    @staticmethod
    def collate_fn(batch: list[Label]) -> dict:
        new_batch = {}
        # keys = Label.model_fields.keys()
        # values = list(zip(*[tuple(dict(b).values()) for b in batch], strict=True))
        # for i, k in enumerate(keys):
        #     value = values[i]
        #     if k == "img":
        #         value = torch.stack(value, 0)
        #     if k in {"bboxes", "cls"}:
        #         value = torch.cat(value, 0)
        #     new_batch[k] = value
        # new_batch["batch_idx"] = list(new_batch["batch_idx"])
        # for i in range(len(new_batch["batch_idx"])):
        #     new_batch["batch_idx"][i] += i
        # new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

        new_batch["img"] = torch.stack([item["img"] for item in batch]).contiguous()
        new_batch["im_file"] = tuple(item["im_file"] for item in batch)
        new_batch["shape"] = tuple(item["shape"] for item in batch)
        new_batch["cls"] = torch.cat([item["classes"] for item in batch], 0)
        new_batch["bboxes"] = torch.cat([item["bboxes"] for item in batch], 0)
        new_batch["normalized"] = tuple(item["normalized"] for item in batch)
        new_batch["bbox_format"] = tuple(item["bbox_format"] for item in batch)
        new_batch["batch_idx"] = torch.cat(
            [torch.full(item["classes"].size()[:1], idx) for idx, item in enumerate(batch)], 0
        )
        return new_batch

    def _regenerate(self) -> list[Label]:
        labels: list[Label] = []
        for _ in range(self._n):
            num_bboxes = random.randrange(3)  # noqa: S311
            cls: list[int] = []
            bboxes: list[torch.Tensor] = []
            for _ in range(num_bboxes):
                x1, y1 = random.uniform(0, 1), random.uniform(0, 1)  # noqa: S311
                x2, y2 = random.uniform(x1, 1), random.uniform(y1, 1)  # noqa: S311
                bbox = xyxy2xywh(torch.as_tensor((x1, y1, x2, y2), dtype=torch.float32))
                bboxes.append(bbox)
                cls.append(random.randrange(self._num_classes))  # noqa: S311
            labels.append(
                Label(
                    im_file=Path(),
                    img=torch.rand(self._shape, dtype=torch.float32, device="cuda"),
                    shape=self._shape,
                    classes=torch.as_tensor(cls),
                    bboxes=torch.stack(bboxes, 0) if bboxes else torch.tensor(()),
                    normalized=True,
                    bbox_format="xywh",
                )
            )

        return labels


class LazyLabel(TypedDict):
    im_file: Path
    shape: tuple[int, int] | tuple[int, int, int]
    classes: torch.Tensor
    bboxes: torch.Tensor
    normalized: bool
    bbox_format: Literal["xywh"]


class Label(LazyLabel):
    img: torch.Tensor
