from __future__ import annotations

import logging
import os
import random
from collections.abc import Sized
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict, cast

import nibabel as nib
import nibabel.filebasedimages
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset
from tqdm_loggable.auto import tqdm

from yolo11_3d.nn.augmentations import (
    batch_generators_intensity_augmentations,
    enlarge_bounding_boxes,
    random_mirroring,
    random_zoom,
    tensor_cutout,
)
from yolo11_3d.ultralytics.utils.ops import xyxy2xywh, xyzwhd2xyzxyz, xyzxyz2xyzwhd

LOGGER = logging.getLogger(__name__)


class LazyLabel(TypedDict):
    im_file: Path
    shape: tuple[int, int] | tuple[int, int, int]
    classes: torch.Tensor
    bboxes: torch.Tensor
    normalized: bool
    bbox_format: Literal["xywh"]


class Label(LazyLabel):
    img: torch.Tensor
    ori_shape: tuple[int, int] | tuple[int, int, int]
    ratio_pad: (
        tuple[
            tuple[float, float] | tuple[float, float, float], tuple[int, int] | tuple[int, int, int]
        ]
        | None
    )


class Batch(TypedDict):
    img: torch.Tensor
    ori_shape: tuple[tuple[int, int] | tuple[int, int, int], ...]
    ratio_pad: tuple[
        tuple[
            tuple[float, float] | tuple[float, float, float], tuple[int, int] | tuple[int, int, int]
        ]
        | None,
        ...,
    ]
    im_file: tuple[Path, ...]
    shape: tuple[tuple[int, int] | tuple[int, int, int], ...]
    cls: torch.Tensor
    bboxes: torch.Tensor
    normalized: tuple[bool, ...]
    bbox_format: tuple[Literal["xywh"], ...]
    batch_idx: torch.Tensor


class NIfTIDataset(Dataset[Label], Sized):
    img_formats: ClassVar[set[str]] = {"nii"}

    def __init__(
        self,
        src_path: Path | list[Path],
        imgsz: tuple[int, int] | tuple[int, int, int],
        augment: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        min_stride: int = 8,
        pad: float = 0.5,
        prefix: str = "",
        fraction: float = 1.0,
        normalization: Literal["none", "zScore", "255_to_01"] = "255_to_01",
        **hyp: Any,
    ) -> None:
        super().__init__()
        self.src_path = src_path
        self.imgsz = imgsz
        self.augment = augment
        self.batch_size = batch_size
        self.stride = stride
        self.min_stride = min_stride
        self.pad = pad
        self.prefix = prefix
        self.fraction = fraction
        self.normalization = normalization
        self.hyp = hyp

        self.img_paths: list[Path] = self.get_img_files(self.src_path)
        self.samples = self.get_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Label:
        with torch.no_grad():
            return self.augmentation(self.load_image_and_label(index))

    def augmentation(self, label: Label) -> Label:
        if not self.augment:
            return label

        scale_tensor = torch.as_tensor(self.imgsz)[[2, 1, 0, 2, 1, 0]]

        img = label["img"]
        classes = label["classes"]

        if label["bbox_format"] == "xywh":
            bboxes = xyzwhd2xyzxyz(label["bboxes"])
        elif label["bbox_format"] == "xyxy":
            bboxes = label["bboxes"]
        else:
            msg = f"bbox format {label['bbox_format']} is not supported."
            raise NotImplementedError(msg)
        bboxes = bboxes * scale_tensor

        labels = torch.cat((classes, bboxes), dim=1)

        img, labels = random_zoom(
            img,
            labels,
            self.hyp["max_zoom"],
            self.hyp["min_zoom"],
            self.hyp["prob_zoom"],
            self.hyp["out_zoom_padding"],
        )

        img, labels = tensor_cutout(img, labels, self.hyp["cutout_params"], self.hyp["prob_cutout"])
        nl = len(labels)
        if nl:
            labels[:, 1:] = xyzxyz2xyzwhd(labels[:, 1:])

        img, labels = enlarge_bounding_boxes(img, labels, self.imgsz, self.min_stride)
        
        if nl:
            labels[:, 1:] = labels[:, 1:] / scale_tensor

        img, labels = random_mirroring(
            img, labels, self.hyp["mirror_axes"], self.hyp["prob_mirroring"]
        )

        if self.hyp["do_batchgenerators_intensity_augmentation"]:
            img = batch_generators_intensity_augmentations(
                img, self.hyp["do_intensity_augmentations_per_channel"]
            )

        label["img"] = img
        label["classes"] = labels[:, 0:1]
        if label["bbox_format"] == "xywh":
            label["bboxes"] = labels[:, 1:]
        elif label["bbox_format"] == "xyxy":
            label["bboxes"] = xyzwhd2xyzxyz(labels[:, 1:])

        return label

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
        if len(original_shape) == 2 and len(resized_shape) == 2:
            ratio = (resized_shape[0] / original_shape[0], resized_shape[1] / original_shape[1])
            pad = (0, 0)
        elif len(original_shape) == 3 and len(resized_shape) == 3:
            ratio = (
                resized_shape[0] / original_shape[0],
                resized_shape[1] / original_shape[1],
                resized_shape[2] / original_shape[2],
            )
            pad = (0, 0, 0)
        else:
            raise ValueError
        return Label(
            im_file=label["im_file"],
            shape=resized_shape,
            ori_shape=original_shape,
            ratio_pad=(ratio, pad),
            classes=label["classes"],
            bboxes=label["bboxes"],
            normalized=label["normalized"],
            bbox_format=label["bbox_format"],
            img=img,
        )

    def get_samples(self) -> list[LazyLabel]:
        # find all images and corresponding labels
        self.label_paths = [
            Path(*("labels" if part == "volumes" or part == "images" else part for part in path.parts)).with_suffix(
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
                img = nib.loadsave.load(img_path)
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
            image_shape = img.shape
            spatial_shape = image_shape if len(image_shape) == 3 else image_shape[:-1]
            samples.append(
                LazyLabel(
                    im_file=img_path,
                    shape=spatial_shape,
                    classes=torch.from_numpy(labels[:, 0:1]),
                    bboxes=torch.from_numpy(labels[:, 1:]),
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
            msg = f"Image Not Found {f}"
            raise FileNotFoundError(msg)
        nifti_image = nib.loadsave.load(f)
        if not isinstance(nifti_image, nibabel.nifti1.Nifti1Image):
            raise TypeError
        # image = nifti_image.get_fdata(caching="unchanged", dtype=np.float32)
        image = np.asanyarray(nifti_image.dataobj, dtype=np.float32)
        im_t = torch.from_numpy(image)
        im_t = im_t.permute(*reversed(range(im_t.ndim)))
        if len(im_t.shape) == 3:
            im_t = im_t.unsqueeze(0)
        im_t_scaled = cast(
            "torch.Tensor", F.interpolate(im_t.unsqueeze(0), self.imgsz, mode="trilinear")
        ).squeeze_(0)

        if self.normalization == "255_to_01":
            im_t_scaled = im_t_scaled.div_(255)
            im_t_scaled = im_t_scaled.clamp_(0.0, 1.0)
        elif self.normalization == "zScore":
            im_t_scaled = (im_t_scaled - im_t_scaled.mean()) / im_t_scaled.std()
        elif self.normalization == "none":
            pass
        else:
            raise NotImplementedError(f"Unknown normalization scheme {self.normalization}")

        return im_t_scaled, im_t.shape[1:], im_t_scaled.shape[1:]

    @staticmethod
    def collate_fn(batch: list[Label]) -> Batch:
        with torch.no_grad():
            return {
                "img": torch.stack([item["img"] for item in batch]),
                "ori_shape": tuple(item["ori_shape"] for item in batch),
                "ratio_pad": tuple(item["ratio_pad"] for item in batch),
                "im_file": tuple(item["im_file"] for item in batch),
                "shape": tuple(item["shape"] for item in batch),
                "cls": torch.cat([item["classes"] for item in batch], 0),
                "bboxes": torch.cat([item["bboxes"] for item in batch], 0),
                "normalized": tuple(item["normalized"] for item in batch),
                "bbox_format": tuple(item["bbox_format"] for item in batch),
                "batch_idx": torch.cat(
                    [torch.full(item["classes"].size()[:1], idx) for idx, item in enumerate(batch)],
                    0,
                ),
            }


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
                    ori_shape=self._shape,
                    ratio_pad=((1.0,) * len(self._shape), (0,) * len(self._shape)),
                    shape=self._shape,
                    classes=torch.as_tensor(cls),
                    bboxes=torch.stack(bboxes, 0) if bboxes else torch.tensor(()),
                    normalized=True,
                    bbox_format="xywh",
                )
            )

        return labels
