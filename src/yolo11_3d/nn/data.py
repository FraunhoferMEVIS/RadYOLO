from __future__ import annotations

import json
import logging
import os
from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from yolo11_3d.nn.augmentations import (
    batch_generators_intensity_augmentations,
    enlarge_bounding_boxes,
    random_mirroring,
    random_zoom,
    tensor_cutout,
)
from yolo11_3d.ultralytics.utils.ops import xyzwhd2xyzxyz, xyzxyz2xyzwhd

LOGGER = logging.getLogger(__name__)


class Sample(TypedDict):
    image: torch.Tensor
    shape: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    ratio_pad: (
        tuple[tuple[float, float, float], tuple[int, int, int]]
        | None
    )
    classes: torch.Tensor
    bboxes: torch.Tensor
    masks: list[torch.Tensor | None]

class Batch(TypedDict):
    img: torch.Tensor
    ori_shape: tuple[tuple[int, int, int], ...]
    ratio_pad: tuple[
        tuple[
            tuple[float, float, float], tuple[int, int, int]
        ]
        | None,
        ...,
    ]
    shape: tuple[tuple[int, int, int], ...]
    cls: torch.Tensor
    bboxes: torch.Tensor
    batch_idx: torch.Tensor
    masks: tuple[list[torch.Tensor | None]]


class NIfTIDatasetWithMasks(Dataset[Sample], Sized):
    def __init__(
        self,
        src_path: Path,
        imgsz: tuple[int, int, int],
        augment: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        min_stride: int = 8,
        pad: float = 0.5,
        fold: int = 0,
        group: Literal["train", "val", "test"] = "train",
        normalization: Literal["none", "zScore", "255_to_01"] = "255_to_01",
        load_masks: bool = True,
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
        self.fold = fold
        self.group = group
        self.normalization = normalization
        self.hyp = hyp
        self.load_masks = load_masks
        splits_file = os.path.join(src_path, "data_splits.json")
        with open(splits_file, "r") as file:
            splits = json.load(file)
        if fold >= len(splits):
            raise Exception(f"There are only {len(splits)} folds defined, but fold index {fold} was selected.")
        self.cases: list[str] = splits[fold][group]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Sample:
        with torch.no_grad():
            return self.augmentation(self.load_image_label_and_mask(index))

    def augmentation(self, sample: Sample) -> Sample:
        if not self.augment:
            return sample

        scale_tensor = torch.as_tensor(self.imgsz)[[2, 1, 0, 2, 1, 0]]

        image = sample["image"]
        classes = sample["classes"]
        bboxes = xyzwhd2xyzxyz(sample["bboxes"])
        masks = sample["masks"]
        bboxes = bboxes * scale_tensor

        labels = torch.cat((classes, bboxes), dim=1)

        image, labels, masks = random_zoom(
            image,
            labels,
            masks,
            self.hyp["max_zoom"],
            self.hyp["min_zoom"],
            self.hyp["prob_zoom"],
            self.hyp["out_zoom_padding"],
        )

        image, labels, masks = tensor_cutout(image, labels, masks, self.hyp["cutout_params"], self.hyp["prob_cutout"])
        nl = len(labels)
        if nl:
            labels[:, 1:] = xyzxyz2xyzwhd(labels[:, 1:])

        image, labels, masks = enlarge_bounding_boxes(image, labels, masks, self.imgsz, self.min_stride)
        
        if nl:
            labels[:, 1:] = labels[:, 1:] / scale_tensor

        image, labels, masks = random_mirroring(
            image, labels, masks, self.hyp["mirror_axes"], self.hyp["prob_mirroring"]
        )

        if self.hyp["do_batchgenerators_intensity_augmentation"]:
            image = batch_generators_intensity_augmentations(
                image, self.hyp["do_intensity_augmentations_per_channel"]
            )

        sample["image"] = image
        sample["classes"] = labels[:, 0:1]
        sample["bboxes"] = labels[:, 1:]
        sample["masks"] = masks

        return sample

    def load_image_label_and_mask(self, index: int) -> Sample:
        case_name = self.cases[index]

        image = self.load_image(case_name)
        image, original_shape, resized_shape = self.resample_and_normalize(image)
        ratio = (
            resized_shape[0] / original_shape[0],
            resized_shape[1] / original_shape[1],
            resized_shape[2] / original_shape[2],
        )
        pad = (0, 0, 0)
        classes, bboxes, masks = self.load_labels_and_masks(case_name)

        return Sample(
            image=image,
            shape=resized_shape,
            original_shape=original_shape,
            ratio_pad=(ratio, pad),
            classes=classes,
            bboxes=bboxes,
            masks=masks
        )

    def load_image(self, case_name: str) -> torch.Tensor:
        image_path = os.path.join(self.src_path, "images", f"{case_name}.nii.gz")
        if not os.path.exists(image_path):
            raise IOError(f"Image path does not exists: {image_path}. Is your dataset format correct?")
        nifti_image = nib.loadsave.load(image_path)
        if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
            raise TypeError
        image = np.asanyarray(nifti_image.dataobj, dtype=np.float32)
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(*reversed(range(image_tensor.ndim)))
        return image_tensor

    def resample_and_normalize(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image_resampled = cast(
            "torch.Tensor", F.interpolate(image.unsqueeze(0), self.imgsz, mode="trilinear")
        ).squeeze_(0)

        if self.normalization == "255_to_01":
            image_resampled = image_resampled.div_(255)
            image_resampled = image_resampled.clamp_(0.0, 1.0)
        elif self.normalization == "zScore":
            image_resampled = (image_resampled - image_resampled.mean()) / image_resampled.std()
        elif self.normalization == "none":
            pass
        else:
            raise NotImplementedError(f"Unknown normalization scheme {self.normalization}")

        return image_resampled, image.shape[1:], image_resampled.shape[1:]
    
    def load_labels_and_masks(self, case_name: str) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
        label_path = os.path.join(self.src_path, "labels", f"{case_name}.txt")
        with open(label_path, "r") as file:
            lines = file.read().strip().splitlines()
        if lines:
            string_labels = [line.split() for line in lines if len(line)]
            if any(len(label) not in (0, 7) for label in string_labels):
                raise Exception(f"Labels in wrong fomat in file {label_path}")
            labels = np.array(string_labels, dtype=np.float32)
        else:
            labels = np.zeros((0, 7))
        classes = torch.from_numpy(labels[:, 0:1])
        bboxes = torch.from_numpy(labels[:, 1:])

        if self.load_masks:
            mask_path = os.path.join(self.src_path, "masks", f"{case_name}.nii.gz")
            if not os.path.exists(mask_path):
                mask_tensor_resampled = None
            else:
                nifti_image = nib.loadsave.load(mask_path)
                if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
                    raise TypeError
                mask_numpy = np.asanyarray(nifti_image.dataobj, dtype=np.uint8)
                mask_tensor = torch.from_numpy(mask_numpy)
                mask_tensor = mask_tensor.permute(*reversed(range(mask_tensor.ndim)))
                mask_tensor = mask_tensor.unsqueeze(0)
                if len(mask_tensor.shape) == 4:
                    mask_tensor = mask_tensor.unsqueeze(0)
                mask_tensor_resampled = cast(
                    "torch.Tensor", F.interpolate(mask_tensor, self.imgsz, mode="nearest-exact")
                ).squeeze_(0)

        return classes, bboxes, mask_tensor_resampled

    @staticmethod
    def collate_fn(batch: list[Sample]) -> Batch:
        with torch.no_grad():
            return {
                "img": torch.stack([sample["image"] for sample in batch]),
                "ori_shape": tuple(sample["original_shape"] for sample in batch),
                "ratio_pad": tuple(sample["ratio_pad"] for sample in batch),
                "shape": tuple(sample["shape"] for sample in batch),
                "cls": torch.cat([sample["classes"] for sample in batch], 0),
                "bboxes": torch.cat([sample["bboxes"] for sample in batch], 0),
                "batch_idx": torch.cat(
                    [torch.full(sample["classes"].size()[:1], idx) for idx, sample in enumerate(batch)],
                    0,
                ),
                "masks": tuple(sample["masks"] for sample in batch),
            }


def build_dataloader(
    dataset: NIfTIDatasetWithMasks,
    batch_size: int,
    workers: int,
    shuffle: bool = True,
    device: torch.device | str = "cpu",
) -> DataLoader:
    """Return an DataLoader for training or validation set."""
    batch_size = min(batch_size, len(dataset))
    number_workers = min(os.cpu_count() or 1, workers)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_workers,
        pin_memory=(device.type if isinstance(device, torch.device) else device) != "cpu",
        collate_fn=getattr(dataset, "collate_fn", None),
        persistent_workers=True
    )
