from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from tqdm_loggable.auto import tqdm

from yolo11_3d.ultralytics.utils import ops
from yolo11_3d.ultralytics.utils.metrics import (
    ConfusionMatrix,
    DetMetrics,
    box_iou,
)
from yolo11_3d.ultralytics.utils.plotting import output_to_target, plot_images

if TYPE_CHECKING:
    from yolo11_3d.ultralytics.data.build import InfiniteDataLoader

    from .config import DatasetConfig, DefaultConfig
    from .data import Batch
    from .model import HardcodedModel
    from .train import Trainer
    from .util import ModelEMA

LOGGER = logging.getLogger(__name__)
LOGGER_YOLO = logging.getLogger("yolo." + __name__)


class Validator():
    def __init__(
        self,
        dataloader: InfiniteDataLoader,
        save_dir: Path,
        args: DefaultConfig,
        names: dict[int, str],
    ) -> None:
        self.args = args
        self.dataloader = dataloader
        self.stride = None  # TODO(muellergroh): determine type annotation
        self.data: DatasetConfig | None = None
        self.device: torch.device | None = None
        self.batch_i: int | None = None
        self.training: bool = True
        self.names: dict[int, str] = names
        self.seen: int = 0
        self.stats: dict[str, list[torch.Tensor]] = {
            "tp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": [],
            "target_img": [],
        }
        self.nc: int = len(self.names)
        if self.args.conf is None:
            self.args.conf = 0.001
        self.confusion_matrix: ConfusionMatrix = ConfusionMatrix(self.nc, self.args.conf)
        self.jdict: list[dict[str, Any]] = []

        self.save_dir: Path = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plots: dict[Path, Any] = {}

        self.nt_per_class: npt.NDArray | None = None
        self.nt_per_image: npt.NDArray | None = None
        self.class_map: list[int] = list(range(1, len(self.names)))
        self.args.task = "detect"
        self.metrics = DetMetrics(
            save_dir=self.save_dir,
            names=self.names,
            plot=self.args.plots,
        )
        self.iouv: torch.Tensor = torch.linspace(0.1, 0.95, 10)  # IoU vector for mAP@0.1:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        self.dice_sum = 0.0
        self.dice_count = 0
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def __call__(
        self, trainer: Trainer | None, ema: ModelEMA, amp: bool = False
    ) -> dict[str, float]:
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training and trainer is not None:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and amp
            model = cast("HardcodedModel", ema.ema)
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            model.eval()
        else:
            model = cast("HardcodedModel", ema.ema)
            self.device = model.device  # update device
            if self.args.task == "classify":
                raise NotImplementedError
            if self.device.type in {"cpu", "mps"}:
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )
            model.eval()


        bar = tqdm(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(model)
        self.jdict = []  # empty before each val
        for batch_i, raw_batch in enumerate(bar):
            self.batch_i = batch_i
            # Preprocess
            batch: Batch = self.preprocess(raw_batch)

            # Inference
            raw_preds: (
                list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
            ) = model(batch["img"])
            if isinstance(raw_preds, torch.Tensor):
                raise TypeError(raw_preds)

            # Loss
            if self.training:
                self.loss += cast(
                    "tuple[torch.Tensor, torch.Tensor]", model.loss(batch, raw_preds)
                )[1]

            # Postprocess and metrics
            pred_class_box, pred_mask_coeff, pred_prototype_masks = raw_preds
            preds = self.postprocess(pred_class_box, pred_mask_coeff)

            self.update_metrics(preds, batch, pred_prototype_masks)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

        stats = self.get_stats()
        self.finalize_metrics()
        self.print_results()
        if self.training:
            model.float()
            results = {
                **stats,
                **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val"),
            }
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats

        if self.args.save_json and self.jdict:
            with self.save_dir.joinpath("predictions.json").open("w") as f:
                LOGGER.info("Saving %s...", f.name)
                json.dump(self.jdict, f)  # flatten and save
        if self.args.plots or self.args.save_json:
            LOGGER.info("Results saved to %s", self.save_dir)
        return stats

    def init_metrics(self, model) -> None:
        """Initialize evaluation metrics for YOLO."""
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
        # reset Dice accumulators
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
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()  # / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor(
                (width, height, width, height), device=self.device
            )
            self.lb = [
                torch.cat(
                    [batch["classes"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]],
                    dim=-1,
                )
                for i in range(nb)
            ]

        return batch

    def postprocess(
        self,
        prediction,
        mask_coefficients
    ) -> list[torch.Tensor]:
        """Apply Non-maximum suppression to prediction outputs, concatenating mask coefficients if present."""

        # Unpack detection outputs (inference_out, loss_out)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        # Attach mask coefficients to predictions if available
        if isinstance(mask_coefficients, torch.Tensor):
            if prediction.shape[0] != mask_coefficients.shape[0] or prediction.shape[-1] != mask_coefficients.shape[-1]:
                LOGGER.warning("Mask coefficients shape mismatch with predictions; ignoring mask coefficients for NMS.")
            else:
                prediction = torch.cat((prediction, mask_coefficients), dim=1)

        return ops.non_max_suppression(
            prediction,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
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
            # TODO(muellergroh): check for 3D
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
        # TODO(muellergroh): check for 3D
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :6], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(
        self, preds: list[torch.Tensor], batch: Batch, prototype_masks: torch.Tensor | None
    ) -> None:
        """Metrics."""
        for si, pred_with_scores_and_coefficients in enumerate(preds):
            self.seen += 1
            num_predictions = len(pred_with_scores_and_coefficients)
            stat = {
                "conf": torch.zeros(0, device=self.device),
                "pred_cls": torch.zeros(0, device=self.device),
                "tp": torch.zeros(num_predictions, self.niou, dtype=torch.bool, device=self.device),
            }
            prepared_batch = self._prepare_batch(si, batch)
            classes, bboxes = prepared_batch.pop("cls"), prepared_batch.pop("bboxes")
            orignal_shape = prepared_batch["ori_shape"]
            num_labels = len(classes)
            stat["target_cls"] = classes
            stat["target_img"] = (
                classes.unique()
            )  # TODO(muellergroh): this seems wrong, since classes don't contain a reference to the image
            if num_predictions == 0:
                if num_labels:
                    for k in self.stats:
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, gt_bboxes=bboxes, gt_cls=classes
                        )
                continue

            # Predictions
            if self.args.single_cls:
                pred_with_scores_and_coefficients[:, 7] = 0
            predn = self._prepare_pred(pred_with_scores_and_coefficients, prepared_batch)
            stat["conf"] = predn[:, 6]
            stat["pred_cls"] = predn[:, 7]

            # Evaluate
            if num_labels:
                stat["tp"] = self._process_batch(predn, bboxes, classes)
                # Compute Dice for matched instances if masks are available
                if prototype_masks is not None:
                    self._update_dice_for_image(
                        predn,
                        pred_with_scores_and_coefficients,
                        bboxes,
                        classes,
                        prototype_masks[si],
                        batch["masks"][si],
                        orignal_shape
                    )
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bboxes, classes)
            for k in self.stats:
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                raise NotImplementedError
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                raise NotImplementedError
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    prepared_batch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def _process_batch(
        self, detections: torch.Tensor, gt_bboxes: torch.Tensor, gt_cls: torch.Tensor
    ) -> torch.Tensor:
        """Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :6])
        return self.match_predictions(detections[:, 7], gt_cls, iou)

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
        orignal_shape
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
        # Prepare shapes (H, W, D)
        proto_hwd = prototypes.shape[1:]
        for g_i, d_i in zip(gt_idx, det_idx, strict=True):
            gt_mask = gt_masks_list[g_i]
            if gt_mask is None:
                continue
            # Downsample GT mask to prototype resolution
            gt_mask_d = F.interpolate(
                gt_mask[None, None, :].to(device=prototypes.device, dtype=prototypes.dtype), proto_hwd, mode="nearest-exact"
            )[0, 0]
            # Reconstruct predicted mask logits from coefficients and prototypes
            coeff = pred_with_coeffs[d_i, 8 : 8 + nm].to(device=prototypes.device, dtype=prototypes.dtype)
            # prototypes shape: (n_proto, H, W, D) -> logits (H, W, D), match training loss
            pred_logits = torch.einsum("n,nhwd->hwd", coeff, prototypes)
            pred_prob = pred_logits.sigmoid()
            # Compute cropping box in prototype grid (approx. stride 4 like in loss)
            gt_box = gt_bboxes_f[g_i]
            gt_box[[0, 3]] = gt_box[[0, 3]] / orignal_shape[2] * proto_hwd[2]
            gt_box[[1, 4]] = gt_box[[1, 4]] / orignal_shape[1] * proto_hwd[1]
            gt_box[[2, 5]] = gt_box[[2, 5]] / orignal_shape[0] * proto_hwd[0]
            pr_box = predn_f[d_i, :6]
            pr_box[[0, 3]] = pr_box[[0, 3]] / orignal_shape[2] * proto_hwd[2]
            pr_box[[1, 4]] = pr_box[[1, 4]] / orignal_shape[1] * proto_hwd[1]
            pr_box[[2, 5]] = pr_box[[2, 5]] / orignal_shape[0] * proto_hwd[0]
            # [x1,y1,z1,x2,y2,z2]
            xyz1 = torch.minimum(gt_box[:3], pr_box[:3])
            xyz2 = torch.maximum(gt_box[3:], pr_box[3:])
            xyz1 = torch.floor(xyz1).clamp_min(0)
            xyz2 = torch.ceil(xyz2)
            xyz2[0] = xyz2[0].clamp(max=proto_hwd[0])
            xyz2[1] = xyz2[1].clamp(max=proto_hwd[1])
            xyz2[2] = xyz2[2].clamp(max=proto_hwd[2])
            x0, y0, z0 = xyz1.to(torch.int)
            x1, y1, z1 = xyz2.to(torch.int)
            if (x1 <= x0) or (y1 <= y0) or (z1 <= z0):
                continue
            p_region = pred_prob[x0:x1, y0:y1, z0:z1]
            g_region = gt_mask_d[x0:x1, y0:y1, z0:z1].to(dtype=p_region.dtype)
            if not g_region.any():
                continue
            # Soft Dice
            intersection = (p_region * g_region).sum().item()
            denom = (p_region.sum() + g_region.sum()).item()
            dice = (2.0 * intersection) / (denom + 1e-7)
            self.dice_sum += float(dice)
            self.dice_count += 1

    def match_predictions(
        self,
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        iou: torch.Tensor,
        use_scipy: Literal[False] = False,
    ) -> torch.Tensor:
        """Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou_np = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                raise NotImplementedError
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou_np * (iou_np >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(
                        cost_matrix, maximize=True
                    )
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou_np >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def get_stats(self) -> dict[str, float]:
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if stats and stats["tp"].any():
            self.metrics.process(**stats)
        # Add Dice metric
        dice_avg = float(self.dice_sum / self.dice_count) if self.dice_count > 0 else 0.0
        return {**self.metrics.results_dict, "metrics/dice(M)": dice_avg}

    def finalize_metrics(self, *args, **kwargs) -> None:
        """Set final values for metrics speed and confusion matrix."""
        # self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
    
    def print_results(self) -> None:
        """Prints training/validation set metrics per class."""
        LOGGER.warning(self.get_desc())
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.warning(pf, "all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                "WARNING ⚠️ no labels found in %s set, can not compute metrics without labels",
                self.args.task,
            )

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.warning(
                    pf,
                    self.names[c],
                    self.nt_per_image[c],
                    self.nt_per_class[c],
                    *self.metrics.class_result(i),
                )

        if self.args.plots:
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
            fname=str(self.save_dir / f"val_batch{ni}_xy_labels.jpg"),
            names=self.names,
        )
        plot_images(
            project_nifti_images(batch["img"], dim=-2),
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"][:, [0, 2, 3, 5]],
            fname=str(self.save_dir / f"val_batch{ni}_xz_labels.jpg"),
            names=self.names,
        )

    def plot_predictions(self, batch: Batch, preds: list[torch.Tensor], ni: int) -> None:
        plot_images(
            project_nifti_images(batch["img"], -3),
            *output_to_target(preds, max_det=self.args.max_det, axis=2),
            fname=str(self.save_dir / f"val_batch{ni}_xy_pred.jpg"),
            names=self.names,
        )
        plot_images(
            project_nifti_images(batch["img"], -2),
            *output_to_target(preds, max_det=self.args.max_det, axis=1),
            fname=str(self.save_dir / f"val_batch{ni}_xz_pred.jpg"),
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
