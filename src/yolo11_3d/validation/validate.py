from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm_loggable.auto import tqdm

from yolo11_3d.utils import ops
from yolo11_3d.validation.metrics import (
    ConfusionMatrix,
    DetMetrics,
    box_iou,
)
from yolo11_3d.validation.plotting import output_to_target, plot_images

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

    from yolo11_3d.config.config import DatasetConfig, DefaultConfig
    from yolo11_3d.data.data import Batch
    from yolo11_3d.models.model import HardcodedModel
    from yolo11_3d.training.train import Trainer
    from yolo11_3d.utils.util import ModelEMA

LOGGER = logging.getLogger(__name__)
LOGGER_YOLO = logging.getLogger("yolo." + __name__)


class Validator:
    def __init__(
        self,
        dataloader: DataLoader,
        save_dir: Path,
        config: DefaultConfig,
        names: dict[int, str],
        save_predictions: bool = False,
    ) -> None:
        self.config = config
        self.dataloader = dataloader
        self.data: DatasetConfig | None = None
        self.device: torch.device | None = None
        self.training: bool = True
        self.names: dict[int, str] = names
        self.seen: int = 0
        self.stats: dict[str, list[torch.Tensor]] = {
            "tp": [],
            "fp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": [],
            "target_img": [],
        }
        self.nc: int = len(self.names)
        if self.config.conf is None:
            self.config.conf = 0.001
        self.confusion_matrix: ConfusionMatrix = ConfusionMatrix(self.nc, self.config.conf)
        self.predictions_for_saving: dict[str, list[dict[str, Any]]] = {}
        self.save_predictions = save_predictions

        self.save_dir: Path = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.nt_per_class: npt.NDArray | None = None
        self.nt_per_image: npt.NDArray | None = None
        self.metrics = DetMetrics(
            save_dir=self.save_dir,
            names=self.names,
            plot=self.config.plots,
        )
        self.iouv: torch.Tensor = torch.linspace(0.1, 0.95, 18)  # IoU vector for mAP@0.1:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        self.dice_sum = 0.0
        self.dice_count = 0

    def __call__(self, trainer: Trainer | None, ema: ModelEMA) -> dict[str, float]:
        """Execute validation process, run inference on dataloader and compute metrics."""
        model = cast("HardcodedModel", ema.ema)
        self.training = trainer is not None
        if self.training and trainer is not None:
            self.device = trainer.device
            validation_loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        else:
            self.device = model.device
        model.eval()

        bar = tqdm(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics()
        self.predictions_for_saving = {}
        for batch_i, raw_batch in enumerate(bar):
            batch: Batch = self.preprocess(raw_batch)

            with torch.autocast(self.device.type), torch.no_grad():
                raw_preds: (
                    list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
                ) = model(batch["img"])
                if isinstance(raw_preds, torch.Tensor):
                    raise TypeError(raw_preds)

                if self.training:
                    validation_loss += cast(
                        "tuple[torch.Tensor, torch.Tensor]", model.loss(batch, raw_preds)
                    )[1]

            # Postprocess and metrics
            pred_class_box, pred_mask_coeff, pred_prototype_masks = raw_preds
            preds = self.postprocess(pred_class_box, pred_mask_coeff)

            self.update_metrics(preds, batch, pred_prototype_masks)
            if self.config.plots and batch_i < 3:
                val_sample_folder = self.save_dir / "validation_samples"
                if not val_sample_folder.exists():
                    val_sample_folder.mkdir()
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

        stats = self.get_stats()
        self.metrics.confusion_matrix = self.confusion_matrix
        self.print_results()
        if self.training:
            model.float()
            results = {
                **stats,
                **trainer.label_loss_items(
                    validation_loss.cpu() / len(self.dataloader), prefix="val"
                ),
            }
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats

        if self.save_predictions and self.predictions_for_saving:
            with self.save_dir.joinpath("predictions.json").open("w") as f:
                LOGGER.info("Saving %s...", f.name)
                json.dump(self.predictions_for_saving, f, indent=4)  # flatten and save
        if self.config.plots or self.save_predictions:
            LOGGER.info("Results saved to %s", self.save_dir)
        return stats

    def init_metrics(self) -> None:
        """Initialize evaluation metrics for YOLO."""
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.config.conf)
        self.seen = 0
        self.predictions_for_saving = {}
        self.stats = {
            "tp": [],
            "fp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": [],
            "target_img": [],
        }
        self.dice_sum = 0.0
        self.dice_count = 0

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP10",
            "mAP10-95)",
        )

    def preprocess(self, batch: Batch) -> Batch:
        """Preprocesses batch of images for YOLO validation."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["masks"] = tuple(
            masks.to(self.device, non_blocking=True) if masks is not None else None
            for masks in batch["masks"]
        )

        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch

    def postprocess(self, prediction, mask_coefficients) -> list[torch.Tensor]:
        """Apply Non-maximum suppression to prediction outputs and handle mask coefficients."""
        # Unpack detection outputs (inference_out, loss_out)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        # If model returns separate boxes, class probs, and objectness, fold for NMS
        if isinstance(prediction, tuple) and len(prediction) == 3:
            dbox, cls_prob, obj_prob = prediction
            # Multiply class probabilities with objectness for NMS
            combined = torch.cat((dbox, cls_prob * obj_prob), 1)
            prediction = combined

        # Combine boundig box and class prediction into one tensor
        if isinstance(prediction, tuple) and len(prediction) == 2:
            dbox, cls_prob = prediction
            combined = torch.cat((dbox, cls_prob), 1)
            prediction = combined

        # Attach mask coefficients to predictions if available
        if isinstance(mask_coefficients, torch.Tensor):
            if (
                prediction.shape[0] != mask_coefficients.shape[0]
                or prediction.shape[-1] != mask_coefficients.shape[-1]
            ):
                LOGGER.warning(
                    f"Mask coefficients shape {prediction.shape} mismatch with predictions"
                    f" shape {mask_coefficients.shape}. Ignoring mask coefficients for NMS."
                )
            else:
                prediction = torch.cat((prediction, mask_coefficients), dim=1)

        return ops.non_max_suppression(
            prediction,
            self.config.conf,
            self.config.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.config.agnostic_nms,
            max_det=self.config.max_det,
            nc=self.nc,
        )

    def _prepare_batch(self, si: int, batch: Batch) -> dict[str, Any]:
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        classes = batch["cls"][idx].squeeze(-1)
        bboxes = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(classes):
            bboxes = (
                ops.xyzwhd2xyzxyz(bboxes)
                * torch.tensor(imgsz, device=self.device)[[2, 1, 0, 2, 1, 0]]
            )  # target boxes
            ops.scale_boxes(imgsz, bboxes, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {
            "cls": classes,
            "bboxes": bboxes,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
        }

    def _prepare_pred(self, pred: torch.Tensor, pbatch: dict[str, Any]) -> torch.Tensor:
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :6], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(
        self, preds: list[torch.Tensor], batch: Batch, prototype_masks: torch.Tensor | None
    ) -> None:
        """Metrics."""
        for sample_index, pred_with_scores_and_coefficients in enumerate(preds):
            self.seen += 1
            num_predictions = len(pred_with_scores_and_coefficients)
            stat = {
                "conf": torch.zeros(0, device=self.device),
                "pred_cls": torch.zeros(0, device=self.device),
                "tp": torch.zeros(num_predictions, self.niou, dtype=torch.bool, device=self.device),
                "fp": torch.zeros(num_predictions, self.niou, dtype=torch.bool, device=self.device),
            }
            prepared_batch = self._prepare_batch(sample_index, batch)
            classes, bboxes = prepared_batch.pop("cls"), prepared_batch.pop("bboxes")
            num_labels = len(classes)
            stat["target_cls"] = classes
            # Count presence per class per-image; ignore negatives already filtered
            stat["target_img"] = classes.unique()
            if num_predictions == 0:
                if num_labels:
                    for k in self.stats:
                        self.stats[k].append(stat[k])
                    if self.config.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, gt_bboxes=bboxes, gt_cls=classes
                        )
                continue

            # Predictions
            predn = self._prepare_pred(pred_with_scores_and_coefficients, prepared_batch)
            stat["conf"] = predn[:, 6]
            stat["pred_cls"] = predn[:, 7]

            # Evaluate
            if num_labels:
                stat["tp"], stat["fp"] = self._process_batch(predn, bboxes, classes)
                # Compute Dice for matched instances if masks are available
                if prototype_masks is not None:
                    # Use possibly filtered masks aligned to classes/bboxes
                    gt_masks_si = batch["masks"][sample_index]
                    self._update_dice_for_image(
                        predn,
                        pred_with_scores_and_coefficients,
                        bboxes,
                        classes,
                        prototype_masks[sample_index],
                        gt_masks_si if gt_masks_si is not None else [],
                    )
            if self.config.plots:
                self.confusion_matrix.process_batch(predn, bboxes, classes)
            for k in self.stats:
                self.stats[k].append(stat[k])

            # Save
            if self.save_predictions:
                prediction_list = self._prediction_to_list(predn)
                case_name = batch["case_names"][sample_index]
                self.predictions_for_saving[case_name] = prediction_list

    def _prediction_to_list(self, prediction: torch.Tensor) -> list[dict[str, Any]]:
        prediction_list = []
        for prediction_i in prediction:
            box_i = prediction_i[:6].tolist()
            box_i = [round(coordinate, 2) for coordinate in box_i]
            confidence_i = round(prediction_i[6].item(), 6)
            class_i = int(prediction_i[7].item())
            # mask_coefficients = prediction_i[8:]
            prediction_list.append({"box": box_i, "confidence": confidence_i, "class": class_i})
        return prediction_list

    def _process_batch(
        self, detections: torch.Tensor, gt_bboxes: torch.Tensor, gt_cls: torch.Tensor
    ) -> torch.Tensor:
        """Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 8) representing detections
            where each detection is (x1, y1, z1, x2, y2, z2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 6) representing ground-truth bounding box
            coordinates. Each bounding box is of the format: (x1, y1, z1, x2, y2, z2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): True positive prediction matrix of shape (N, 18) for 18 IoU levels.
            (torch.Tensor): False positive prediction matrix of shape (N, 18) for 18 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead,
            it provides an intermediate representation used for evaluating predictions against
            ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :6])
        true_positives, false_positives = self.match_predictions(detections[:, 7], gt_cls, iou)
        return true_positives, false_positives

    def _get_matches(
        self,
        detections: torch.Tensor,
        gt_bboxes: torch.Tensor,
        iou_thres: float,
        gt_classes: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return matched indices (gt_index, det_index) using IoU threshold and class equality."""
        if gt_bboxes.shape[0] == 0 or detections.shape[0] == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
        iou = box_iou(gt_bboxes, detections[:, :6])
        # zero-out pairs with different classes
        class_equal = gt_classes[:, None] == detections[:, 7]
        iou = iou * class_equal
        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        if matches.shape[0] == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
        m0, m1 = matches[:, 0].astype(int), matches[:, 1].astype(int)
        return m0, m1

    def _update_dice_for_image(
        self,
        predn: torch.Tensor,
        pred_with_coeffs: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        prototypes: torch.Tensor,
        gt_masks_list: list[torch.Tensor | None],
    ) -> None:
        """Compute and accumulate Dice for matched instances in a single image."""
        # Require mask coefficients appended after class column
        nm = pred_with_coeffs.shape[1] - 8
        if len(gt_masks_list) == 0:
            return
        predn_f = predn.to(device=prototypes.device, dtype=prototypes.dtype)
        gt_bboxes_f = gt_bboxes.to(device=prototypes.device, dtype=prototypes.dtype)
        # Match detections to gt by IoU threshold from confusion matrix
        gt_idx, det_idx = self._get_matches(
            predn_f, gt_bboxes_f, self.confusion_matrix.iou_thres, gt_cls
        )
        if gt_idx.size == 0:
            return
        # Prepare shapes (D, W, H)
        proto_dhw = prototypes.shape[1:]
        for g_i, d_i in zip(gt_idx, det_idx, strict=True):
            gt_mask = gt_masks_list[g_i]
            if gt_mask is None:
                continue
            gt_mask_downsampled = F.interpolate(
                gt_mask[None, None, :].to(device=prototypes.device, dtype=prototypes.dtype),
                proto_dhw,
                mode="nearest-exact",
            )[0, 0]
            # Reconstruct predicted mask logits from coefficients and prototypes
            coeff = pred_with_coeffs[d_i, 8 : 8 + nm].to(
                device=prototypes.device, dtype=prototypes.dtype
            )
            # prototypes shape: (n_proto, H, W, D) -> logits (H, W, D), match training loss
            pred_logits = torch.einsum("n,nhwd->hwd", coeff, prototypes)
            pred_prob = pred_logits.sigmoid()
            pred_mask = pred_prob >= 0.5
            if not gt_mask_downsampled.any():
                continue
            intersection = (pred_mask * gt_mask_downsampled).sum().item()
            denom = (pred_mask.sum() + gt_mask_downsampled.sum()).item()
            dice = (2.0 * intersection) / (denom + 1e-7)
            self.dice_sum += float(dice)
            self.dice_count += 1

    def match_predictions(
        self,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        iou: torch.Tensor,
    ) -> torch.Tensor:
        """Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(P,).
            true_classes (torch.Tensor): Target class indices of shape(T,).
            iou (torch.Tensor): An PxT tensor containing the pairwise IoU values for prediction and
                ground truth bounding boxes

        Returns:
            (torch.Tensor): True positive tensor of shape(P, 18) for 18 IoU thresholds.
            (torch.Tensor): False positive tensor of shape(P, 18) for 18 IoU thresholds.
        """
        # shape(P, 18), P: predictions, 18: IoU thresholds
        true_positives_np = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        true_positives_missing_np = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(
            bool
        )
        # shape(T, P) T: true class, P: predicted
        correct_class = true_classes[:, None] == pred_classes
        missing_class = torch.logical_and(
            true_classes[:, None] != pred_classes, true_classes[:, None] == -1
        )  # calculate false positives for class -1 (missing class label)
        iou_tp = iou * correct_class  # shape(T, P) zero out the wrong classes
        iou_missing_class = iou * missing_class  # shape(T, P) zero out the non-missing classes
        iou_tp_np = iou_tp.cpu().numpy()  # shape(T, P)
        iou_missing_np = iou_missing_class.cpu().numpy()  # shape(T, P)
        for i, iou_threshold in enumerate(self.iouv.cpu().tolist()):
            matches_tp = np.nonzero(
                iou_tp_np >= iou_threshold
            )  # IoU >= threshold and classes match
            matches_missing = np.nonzero(
                iou_missing_np >= iou_threshold
            )  # IoU >= threshold, classes do not match and true class -1
            matches_tp = np.array(matches_tp).T  # shape(N, 2) N: number of true positive matches
            matches_missing = np.array(
                matches_missing
            ).T  # shape(M, 2) M: number of missing class matches
            if matches_tp.shape[0] > 0:
                sorted_matches_tp = matches_tp[
                    iou_tp_np[matches_tp[:, 0], matches_tp[:, 1]].argsort()[::-1]
                ]  # sort matches by IoU
                unique_matches_tp = sorted_matches_tp[
                    np.unique(sorted_matches_tp[:, 1], return_index=True)[1]
                ]  # select unique match with highest IoU for each true class
                unique_matches_tp = unique_matches_tp[
                    np.unique(unique_matches_tp[:, 0], return_index=True)[1]
                ]  # select unique match with highest IoU for each predicted class
                true_positives_np[unique_matches_tp[:, 1].astype(int), i] = True
            if matches_missing.shape[0] > 0:
                sorted_matches_missing = matches_missing[
                    iou_missing_np[matches_missing[:, 0], matches_missing[:, 1]].argsort()[::-1]
                ]  # sort matches by IoU
                unique_matches_missing = sorted_matches_missing[
                    np.unique(sorted_matches_missing[:, 1], return_index=True)[1]
                ]  # select unique match with highest IoU for each true class
                unique_matches_missing = unique_matches_missing[
                    np.unique(unique_matches_missing[:, 0], return_index=True)[1]
                ]  # select unique match with highest IoU for each predicted class
                true_positives_missing_np[unique_matches_missing[:, 1].astype(int), i] = True
        true_positives = torch.tensor(
            true_positives_np, dtype=torch.bool, device=pred_classes.device
        )
        true_positives_missing = torch.tensor(
            true_positives_missing_np, dtype=torch.bool, device=pred_classes.device
        )
        false_positives = torch.logical_and(
            torch.logical_not(true_positives), torch.logical_not(true_positives_missing)
        )
        return true_positives, false_positives

    def get_stats(self) -> dict[str, float]:
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        # Robust against negative classes (e.g., ignore label -1)
        target_cls_np = stats.get("target_cls", np.array([], dtype=np.int64))
        target_img_np = stats.get("target_img", np.array([], dtype=np.int64))
        if target_cls_np.size:
            pos = target_cls_np >= 0
            self.nt_per_class = np.bincount(target_cls_np[pos].astype(int), minlength=self.nc)
        else:
            self.nt_per_class = np.zeros(self.nc, dtype=np.int64)
        if target_img_np.size:
            posi = target_img_np >= 0
            self.nt_per_image = np.bincount(target_img_np[posi].astype(int), minlength=self.nc)
        else:
            self.nt_per_image = np.zeros(self.nc, dtype=np.int64)
        stats.pop("target_img", None)
        if stats and stats["tp"].any():
            self.metrics.process(**stats)
        # Add Dice metric
        dice_avg = float(self.dice_sum / self.dice_count) if self.dice_count > 0 else 0.0
        return {**self.metrics.results_dict, "metrics/dice(M)": dice_avg}

    def print_results(self) -> None:
        """Prints training/validation set metrics per class."""
        LOGGER.warning(self.get_desc())
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.warning(pf, "all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                "WARNING ⚠️ no labels found in dataset, can not compute metrics without labels"
            )

        # Print results per class
        if self.config.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.warning(
                    pf,
                    self.names[c],
                    self.nt_per_image[c],
                    self.nt_per_class[c],
                    *self.metrics.class_result(i),
                )

        if self.config.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    names=self.names.values(),
                    normalize=normalize,
                )
        # Print Dice metric summary
        dice_avg = float(self.dice_sum / self.dice_count) if self.dice_count > 0 else 0.0
        LOGGER.warning("%22s%11.3g", "Mask Dice(M)", dice_avg)

    def plot_val_samples(self, batch: Batch, ni: int) -> None:
        """Plot validation image samples."""
        plot_images(
            project_nifti_images(batch["img"], -3),
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"][:, [0, 1, 3, 4]],
            fname=str(self.save_dir / "validation_samples" / f"val_batch{ni}_xy_labels.jpg"),
            names=self.names,
        )
        plot_images(
            project_nifti_images(batch["img"], dim=-2),
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"][:, [0, 2, 3, 5]],
            fname=str(self.save_dir / "validation_samples" / f"val_batch{ni}_xz_labels.jpg"),
            names=self.names,
        )

    def plot_predictions(self, batch: Batch, preds: list[torch.Tensor], ni: int) -> None:
        """Plot predictions samples."""
        plot_images(
            project_nifti_images(batch["img"], -3),
            *output_to_target(preds, max_det=self.config.max_det, axis=2),
            fname=str(self.save_dir / "validation_samples" / f"val_batch{ni}_xy_pred.jpg"),
            names=self.names,
        )
        plot_images(
            project_nifti_images(batch["img"], -2),
            *output_to_target(preds, max_det=self.config.max_det, axis=1),
            fname=str(self.save_dir / "validation_samples" / f"val_batch{ni}_xz_pred.jpg"),
            names=self.names,
        )


def project_nifti_images(
    image: torch.Tensor, dim: int, upper_quantile: float = 0.995
) -> torch.Tensor:
    if image.shape[1] > 1:
        time_diff_img = (image[:, [1]] - image[:, [0]]).amax(dim)
    else:
        time_diff_img = image.amax(dim)
    percentiles = (
        time_diff_img.reshape((time_diff_img.shape[0], -1))
        .to(torch.float)
        .quantile(upper_quantile, 1, keepdim=True)
        .to(time_diff_img.dtype)
    )
    return time_diff_img.div(percentiles[..., None, None]).clamp(0, 1)
