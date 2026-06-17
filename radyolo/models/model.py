from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import torch
from torch import nn

from radyolo.config.config import (
    DefaultConfig,
    ImageLabelHeadFeature,
    ModelCheckpoint,
    PrototypeLayer,
    load_model_checkpoint,
)
from radyolo.data.image_labels import (
    ImageLabelTaskSpec,
    load_image_label_task_specs,
    resolve_survival_task_specs,
)
from radyolo.data.tabular import load_tabular_feature_names
from radyolo.models.layers import (
    C2PSA,
    SPPF,
    C3k2,
    Concat,
    Conv,
    Detect,
    ModelOutput,
    PrototypeMasksFullResolution,
    PrototypeMasksFullResolutionV2,
    PrototypeMasksHalfResolution,
    PrototypeMasksHalfResolutionV2,
    PrototypeMasksHighResolution,
)
from radyolo.utils.util import intersect_dicts, make_divisible, select_device, unwrap_model


@dataclass(frozen=True)
class LoadedCheckpointModel:
    checkpoint: ModelCheckpoint
    train_config: DefaultConfig
    model: HardcodedModel
    device: torch.device


def get_model_min_stride(model: nn.Module | None, default: int = 8) -> int:
    if model is not None:
        model = unwrap_model(model)
    stride = getattr(model, "stride", None)
    if isinstance(stride, torch.Tensor):
        return int(stride.min().item())
    return default


class GlobalImageLabelHead(nn.Module):
    def __init__(
        self,
        feature_channels: dict[str, int],
        task_specs: dict[str, ImageLabelTaskSpec],
        selected_feature_maps: list[ImageLabelHeadFeature],
        projection_channels: int,
        mlp_hidden_layers: int,
        mlp_hidden_channels: int,
        dropout: float,
        tabular_encoder: TabularTransformerEncoder | TabularTransformerEncoderV2 | None = None,
    ) -> None:
        super().__init__()
        if not task_specs:
            msg = "Global image-label head requires at least one declared image label task"
            raise ValueError(msg)
        if not selected_feature_maps:
            msg = "Global image-label head requires at least one selected feature map"
            raise ValueError(msg)

        self.selected_feature_maps = tuple(selected_feature_maps)
        self.projectors = nn.ModuleDict(
            {
                feature_name: Conv(feature_channels[feature_name], projection_channels, 1)
                for feature_name in self.selected_feature_maps
            }
        )
        pooled_feature_width = len(self.selected_feature_maps) * projection_channels * 2
        self.tabular_encoder = tabular_encoder
        mlp_input_width = pooled_feature_width
        if self.tabular_encoder is not None:
            mlp_input_width += self.tabular_encoder.output_width

        mlp_layers: list[nn.Module] = []
        for _ in range(mlp_hidden_layers):
            mlp_layers.extend(
                [
                    nn.Linear(mlp_input_width, mlp_hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            mlp_input_width = mlp_hidden_channels
        self.shared_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.task_heads = nn.ModuleDict()
        for task_name, task_spec in task_specs.items():
            if task_spec.type == "multiclass":
                if task_spec.classes is None:
                    msg = f"Multiclass task '{task_name}' requires declared class names"
                    raise ValueError(msg)
                output_width = len(task_spec.classes)
            elif task_spec.type == "survival":
                if not task_spec.survival_bin_thresholds:
                    msg = f"Survival task '{task_name}' requires resolved bin thresholds"
                    raise ValueError(msg)
                output_width = len(task_spec.survival_bin_thresholds)
            else:
                msg = f"Unsupported image-label task type '{task_spec.type}' for '{task_name}'"
                raise ValueError(msg)
            self.task_heads[task_name] = nn.Linear(mlp_input_width, output_width)

    def _resolve_tabular_inputs(
        self,
        batch_size: int,
        device: torch.device,
        tabular_values: torch.Tensor | None,
        tabular_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tabular_encoder is None:
            msg = "Tabular inputs can only be resolved when tabular_encoder is configured"
            raise RuntimeError(msg)
        num_features = self.tabular_encoder.num_features
        default_values = torch.zeros(
            (batch_size, num_features),
            dtype=torch.float32,
            device=device,
        )
        default_mask = torch.zeros(
            (batch_size, num_features),
            dtype=torch.bool,
            device=device,
        )
        if tabular_values is None or tabular_mask is None:
            return default_values, default_mask

        if tabular_values.shape != default_values.shape:
            msg = (
                "tabular_values has incompatible shape. Expected "
                f"{tuple(default_values.shape)}, got {tuple(tabular_values.shape)}."
            )
            raise ValueError(msg)
        if tabular_mask.shape != default_mask.shape:
            msg = (
                "tabular_mask has incompatible shape. Expected "
                f"{tuple(default_mask.shape)}, got {tuple(tabular_mask.shape)}."
            )
            raise ValueError(msg)
        return (
            tabular_values.to(device=device, dtype=torch.float32),
            tabular_mask.to(device=device, dtype=torch.bool),
        )

    def forward(
        self,
        feature_maps: dict[str, torch.Tensor],
        tabular_values: torch.Tensor | None = None,
        tabular_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled_features: list[torch.Tensor] = []
        for feature_name in self.selected_feature_maps:
            projected = self.projectors[feature_name](feature_maps[feature_name])
            pooled_features.append(self.avg_pool(projected).flatten(1))
            pooled_features.append(self.max_pool(projected).flatten(1))
        shared_feature_inputs = [torch.cat(pooled_features, dim=1)]
        if self.tabular_encoder is not None:
            feature_device = pooled_features[0].device
            resolved_values, resolved_mask = self._resolve_tabular_inputs(
                batch_size=pooled_features[0].shape[0],
                device=feature_device,
                tabular_values=tabular_values,
                tabular_mask=tabular_mask,
            )
            shared_feature_inputs.append(self.tabular_encoder(resolved_values, resolved_mask))
        shared_features = self.shared_mlp(torch.cat(shared_feature_inputs, dim=1))
        return {
            task_name: task_head(shared_features)
            for task_name, task_head in self.task_heads.items()
        }


class TabularTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_features < 1:
            msg = "Tabular transformer encoder requires at least one feature"
            raise ValueError(msg)
        if d_model % num_heads != 0:
            msg = (
                "Tabular transformer encoder requires d_model divisible by num_heads, got "
                f"d_model={d_model}, num_heads={num_heads}."
            )
            raise ValueError(msg)
        self.num_features = num_features
        self.output_width = d_model
        self.feature_value_projection = nn.Linear(1, d_model)
        self.feature_embeddings = nn.Parameter(torch.empty(num_features, d_model))
        self.mask_tokens = nn.Parameter(torch.empty(num_features, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.feature_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_tokens, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.num_features:
            msg = (
                "Tabular values must have shape (batch_size, num_features) with "
                f"num_features={self.num_features}, got {tuple(values.shape)}."
            )
            raise ValueError(msg)
        if observed_mask.shape != values.shape:
            msg = (
                "Tabular observed_mask must match values shape, got "
                f"values={tuple(values.shape)} and mask={tuple(observed_mask.shape)}."
            )
            raise ValueError(msg)

        projected_values = self.feature_value_projection(values.unsqueeze(-1))
        expanded_feature_embeddings = self.feature_embeddings.unsqueeze(0).expand(
            values.shape[0],
            -1,
            -1,
        )
        expanded_mask_tokens = self.mask_tokens.unsqueeze(0).expand(values.shape[0], -1, -1)
        observed_mask_expanded = observed_mask.unsqueeze(-1)
        feature_tokens = torch.where(observed_mask_expanded, projected_values, expanded_mask_tokens)
        feature_tokens = feature_tokens + expanded_feature_embeddings
        encoded_tokens = self.transformer(feature_tokens)
        return self.output_norm(encoded_tokens.mean(dim=1))


class TabularTransformerEncoderV2(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_channels: int,
        dropout: float,
        per_feature_value_projection: bool = False,
        shared_mask_token: bool = False,
    ) -> None:
        super().__init__()
        if num_features < 1:
            msg = "Tabular transformer encoder requires at least one feature"
            raise ValueError(msg)
        if d_model % num_heads != 0:
            msg = (
                "Tabular transformer encoder requires d_model divisible by num_heads, got "
                f"d_model={d_model}, num_heads={num_heads}."
            )
            raise ValueError(msg)
        self.num_features = num_features
        self.output_width = d_model
        self.per_feature_value_projection = per_feature_value_projection
        if self.per_feature_value_projection:
            self.feature_value_projections = nn.ModuleList(
                nn.Linear(1, d_model) for _ in range(num_features)
            )
            self.feature_value_projection = None
        else:
            self.feature_value_projection = nn.Linear(1, d_model)
            self.feature_value_projections = None
        self.feature_embeddings = nn.Parameter(torch.empty(num_features, d_model))
        self.shared_mask_token = shared_mask_token
        if self.shared_mask_token:
            self.mask_token = nn.Parameter(torch.empty(1, d_model))
            self.mask_tokens = None
        else:
            self.mask_tokens = nn.Parameter(torch.empty(num_features, d_model))
            self.mask_token = None
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.feature_embeddings, mean=0.0, std=0.02)
        if self.mask_tokens is not None:
            nn.init.normal_(self.mask_tokens, mean=0.0, std=0.02)
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.num_features:
            msg = (
                "Tabular values must have shape (batch_size, num_features) with "
                f"num_features={self.num_features}, got {tuple(values.shape)}."
            )
            raise ValueError(msg)
        if observed_mask.shape != values.shape:
            msg = (
                "Tabular observed_mask must match values shape, got "
                f"values={tuple(values.shape)} and mask={tuple(observed_mask.shape)}."
            )
            raise ValueError(msg)

        if self.feature_value_projections is not None:
            projected_values = torch.stack(
                [
                    projection(values[:, feature_index : feature_index + 1])
                    for feature_index, projection in enumerate(self.feature_value_projections)
                ],
                dim=1,
            )
        else:
            if self.feature_value_projection is None:
                msg = "feature_value_projection is not initialized"
                raise RuntimeError(msg)
            projected_values = self.feature_value_projection(values.unsqueeze(-1))
        expanded_feature_embeddings = self.feature_embeddings.unsqueeze(0).expand(
            values.shape[0],
            -1,
            -1,
        )
        if self.mask_tokens is not None:
            expanded_mask_tokens = self.mask_tokens.unsqueeze(0).expand(values.shape[0], -1, -1)
        else:
            if self.mask_token is None:
                msg = "mask_token is not initialized"
                raise RuntimeError(msg)
            expanded_mask_tokens = self.mask_token.unsqueeze(0).expand(
                values.shape[0],
                self.num_features,
                -1,
            )
        observed_mask_expanded = observed_mask.unsqueeze(-1)
        feature_tokens = torch.where(observed_mask_expanded, projected_values, expanded_mask_tokens)
        feature_tokens = feature_tokens + expanded_feature_embeddings
        cls_tokens = self.cls_token.expand(values.shape[0], -1, -1)
        feature_tokens = torch.cat([cls_tokens, feature_tokens], dim=1)
        encoded_tokens = self.transformer(feature_tokens)
        return self.output_norm(encoded_tokens[:, 0])


def _resolve_image_label_tasks(
    train_config: DefaultConfig,
    image_label_tasks: dict[str, ImageLabelTaskSpec] | None = None,
) -> dict[str, ImageLabelTaskSpec]:
    if image_label_tasks is not None:
        return dict(image_label_tasks)

    task_specs = load_image_label_task_specs(Path(train_config.data))
    task_specs = resolve_survival_task_specs(
        Path(train_config.data),
        task_specs,
        num_bins=train_config.image_label_survival_num_bins,
    )
    if train_config.enable_image_label_head and not task_specs:
        msg = (
            "enable_image_label_head=True requires dataset.yaml to declare image_label_tasks"
        )
        raise ValueError(msg)
    return task_specs


def _resolve_tabular_feature_names(train_config: DefaultConfig) -> tuple[str, ...]:
    feature_names = load_tabular_feature_names(Path(train_config.data))
    if train_config.enable_tabular_image_label_fusion and not feature_names:
        msg = (
            "enable_tabular_image_label_fusion=True requires dataset.yaml to declare "
            "tabular_feature_names"
        )
        raise ValueError(msg)
    return feature_names


class HardcodedModel(nn.Module):
    MODEL_SCALES: ClassVar[dict[str, tuple[float, float, int]]] = {
        "n": (0.50, 0.25, 256),
        "s": (0.50, 0.50, 512),
        "m": (0.50, 1.00, 512),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.50, 768),
    }

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        scale: Literal["n", "s", "m", "l", "x"],
        dfl_reg_max: int,
        bbox_parametrization: Literal["ltrb", "center_extents"],
        bbox_center_offset_limit: float,
        enable_output_masks: bool = True,
        num_prototype_masks: int = 8,
        prototype_layer: PrototypeLayer = "higher_resolution",
        separate_objectness_head: bool = False,
        include_stride4_output: bool = False,
        image_label_tasks: dict[str, ImageLabelTaskSpec] | None = None,
        enable_image_label_head: bool = False,
        image_label_head_feature_maps: list[ImageLabelHeadFeature] | None = None,
        image_label_head_projection_channels: int = 128,
        image_label_head_mlp_hidden_layers: int = 1,
        image_label_head_mlp_hidden_channels: int = 256,
        image_label_head_dropout: float = 0.1,
        enable_tabular_image_label_fusion: bool = False,
        tabular_transformer_variant: Literal["v1", "v2"] = "v1",
        tabular_feature_names: tuple[str, ...] = (),
        tabular_transformer_d_model: int = 64,
        tabular_transformer_num_heads: int = 4,
        tabular_transformer_num_layers: int = 2,
        tabular_transformer_feedforward_channels: int = 256,
        tabular_transformer_dropout: float = 0.1,
        tabular_transformer_v2_per_feature_value_projection: bool = False,
        tabular_transformer_v2_shared_mask_token: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.bbox_parametrization = bbox_parametrization
        self.bbox_center_offset_limit = bbox_center_offset_limit
        self.image_label_task_specs = dict(image_label_tasks or {})
        self.image_label_head_feature_maps = tuple(
            image_label_head_feature_maps or ["o10", "o16", "o19", "o22"]
        )
        self.tabular_feature_names = tuple(tabular_feature_names)

        depth_multiplier, width_multiplier, max_channels = HardcodedModel.MODEL_SCALES[scale]

        def resolve_channel_size(x: int) -> int:
            return make_divisible(min(x * width_multiplier, max_channels), 8)

        def num_repeats_(n: int) -> int:
            return max(round(n * depth_multiplier), 1)

        self.enable_output_masks = enable_output_masks
        self.num_prototype_masks = num_prototype_masks
        self.prototype_layer_mode = prototype_layer
        self.separate_objectness_head = separate_objectness_head
        self.include_stride4_output = include_stride4_output
        self.enable_image_label_head = enable_image_label_head and bool(self.image_label_task_specs)
        self.enable_tabular_image_label_fusion = (
            enable_tabular_image_label_fusion
            and self.enable_image_label_head
            and len(self.tabular_feature_names) > 0
        )

        c3k2_use_c3k = scale in ("m", "l", "x")

        # BACKBONE

        out_channels = resolve_channel_size(64)
        self.l0 = Conv(input_channels, out_channels, 3, 2)
        output_channels: list[int] = [out_channels]

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(128)
        self.l1 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(256)
        self.l2 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k, 0.25)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(256)
        self.l3 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l4 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k, 0.25)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(512)
        self.l5 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l6 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(1024)
        self.l7 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(1024)
        self.l8 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(1024)
        self.l9 = SPPF(in_channels, out_channels, 5)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(1024)
        self.l10 = C2PSA(in_channels, out_channels, num_repeats)
        output_channels.append(out_channels)

        # HEAD

        out_channels = output_channels[-1]
        self.l11 = nn.Upsample(None, 2, "nearest")
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 6])
        self.l12 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l13 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        out_channels = output_channels[-1]
        self.l14 = nn.Upsample(None, 2, "nearest")
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 4])
        self.l15 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(256)
        self.l16 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        # Optional P2 (stride-4) output branch: upsample P3 and fuse with stride-4 backbone (l2)
        # Channel sizes used across pyramid
        c_p2 = resolve_channel_size(128)
        c_p3 = resolve_channel_size(256)
        c_p4 = resolve_channel_size(512)
        c_p5 = resolve_channel_size(1024)
        if self.include_stride4_output:
            # These layers operate on feature maps produced in forward; we define them here
            self.p2_upsample = nn.Upsample(None, 2, "nearest")
            # Concat of upsampled P3 (c_p3) with stride-4 backbone feature (from l2, c_p3)
            self.p2_concat = Concat(1)
            self.p2_c3k2 = C3k2(c_p3 + c_p3, c_p2, num_repeats_(2), c3k2_use_c3k)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(256)
        self.l17 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 13])
        self.l18 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l19 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(512)
        self.l20 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 10])
        self.l21 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(1024)
        self.l22 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        if enable_image_label_head and not self.image_label_task_specs:
            msg = "Global image-label head was enabled but no image label tasks were provided"
            raise ValueError(msg)

        if self.enable_image_label_head:
            tabular_encoder = None
            if self.enable_tabular_image_label_fusion:
                if tabular_transformer_variant == "v2":
                    tabular_encoder = TabularTransformerEncoderV2(
                        num_features=len(self.tabular_feature_names),
                        d_model=tabular_transformer_d_model,
                        num_heads=tabular_transformer_num_heads,
                        num_layers=tabular_transformer_num_layers,
                        feedforward_channels=tabular_transformer_feedforward_channels,
                        dropout=tabular_transformer_dropout,
                        per_feature_value_projection=(
                            tabular_transformer_v2_per_feature_value_projection
                        ),
                        shared_mask_token=tabular_transformer_v2_shared_mask_token,
                    )
                else:
                    tabular_encoder = TabularTransformerEncoder(
                        num_features=len(self.tabular_feature_names),
                        d_model=tabular_transformer_d_model,
                        num_heads=tabular_transformer_num_heads,
                        num_layers=tabular_transformer_num_layers,
                        feedforward_channels=tabular_transformer_feedforward_channels,
                        dropout=tabular_transformer_dropout,
                    )
            self.image_label_head = GlobalImageLabelHead(
                feature_channels={
                    "o10": output_channels[10],
                    "o16": output_channels[16],
                    "o19": output_channels[19],
                    "o22": output_channels[22],
                },
                task_specs=self.image_label_task_specs,
                selected_feature_maps=list(self.image_label_head_feature_maps),
                projection_channels=image_label_head_projection_channels,
                mlp_hidden_layers=image_label_head_mlp_hidden_layers,
                mlp_hidden_channels=image_label_head_mlp_hidden_channels,
                dropout=image_label_head_dropout,
                tabular_encoder=tabular_encoder,
            )

        detect_chs: list[int] = (
            [c_p2, c_p3, c_p4, c_p5] if self.include_stride4_output else [c_p3, c_p4, c_p5]
        )

        if self.enable_output_masks:
            if self.prototype_layer_mode == "full_resolution":
                self.prototype_layer = PrototypeMasksFullResolution(
                    input_channels,
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            elif self.prototype_layer_mode == "full_resolution_v2":
                self.prototype_layer = PrototypeMasksFullResolutionV2(
                    input_channels,
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            elif self.prototype_layer_mode == "half_resolution":
                self.prototype_layer = PrototypeMasksHalfResolution(
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            elif self.prototype_layer_mode == "half_resolution_v2":
                self.prototype_layer = PrototypeMasksHalfResolutionV2(
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            else:
                self.prototype_layer = PrototypeMasksHighResolution(
                    output_channels[0], output_channels[2], output_channels[16], num_prototype_masks
                )

            c4 = max(output_channels[16] // 4, num_prototype_masks)
            # Build coefficient heads dynamically for 3 or 4 detection levels
            self.mask_coefficient_layer = nn.ModuleList(
                nn.Sequential(
                    Conv(ch, c4, 3),
                    Conv(c4, c4, 3),
                    nn.Conv3d(c4, num_prototype_masks, 1),
                )
                for ch in detect_chs
            )

        self.l23 = Detect(
            num_classes,
            detect_chs,
            dfl_reg_max=dfl_reg_max,
            separate_objectness_head=self.separate_objectness_head,
            bbox_parametrization=self.bbox_parametrization,
            center_offset_limit=self.bbox_center_offset_limit,
        )
        s = 256
        sample_output = self.forward(torch.zeros(1, input_channels, s, s, s))
        self.l23.stride = torch.tensor([s / x.shape[-2] for x in sample_output.raw_predictions])
        self.stride = self.l23.stride
        self.l23.bias_init()

    def forward(
        self,
        x_input: torch.Tensor,
        *,
        tabular_values: torch.Tensor | None = None,
        tabular_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        # BACKBONE
        x = o0 = self.l0(x_input)
        x = self.l1(x)
        x = o2 = self.l2(x)
        x = self.l3(x)
        x = o4 = self.l4(x)
        x = self.l5(x)
        x = o6 = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = o10 = self.l10(x)

        # HEAD
        x = self.l11(x)
        x = self.l12([x, o6])
        x = o13 = self.l13(x)

        x = self.l14(x)
        x = self.l15([x, o4])
        x = o16 = self.l16(x)

        # Optional P2 (stride-4) feature from upsampled P3 fused with stride-4 backbone feature
        if self.include_stride4_output:
            p2 = self.p2_c3k2(self.p2_concat([self.p2_upsample(o16), o2]))

        x = self.l17(x)
        x = self.l18([x, o13])
        x = o19 = self.l19(x)

        x = self.l20(x)
        x = self.l21([x, o10])
        x = o22 = self.l22(x)

        if self.include_stride4_output:
            output = self.l23([p2, o16, o19, o22])
        else:
            output = self.l23([o16, o19, o22])

        if self.enable_output_masks:
            if self.prototype_layer_mode in {"full_resolution", "full_resolution_v2"}:
                output.prototype_masks = self.prototype_layer(x_input, o0, o2, o16, o19)
            elif self.prototype_layer_mode in {"half_resolution", "half_resolution_v2"}:
                output.prototype_masks = self.prototype_layer(o0, o2, o16, o19)
            else:
                output.prototype_masks = self.prototype_layer(o0, o2, o16)
            if output.prototype_masks is None:
                msg = "Prototype masks are not available"
                raise RuntimeError(msg)
            batch_size = output.prototype_masks.shape[0]
            if self.include_stride4_output:
                output.mask_coefficients = torch.cat(
                    [
                        self.mask_coefficient_layer[0](p2).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[1](o16).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[2](o19).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[3](o22).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                    ],
                    2,
                )
            else:
                output.mask_coefficients = torch.cat(
                    [
                        self.mask_coefficient_layer[0](o16).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[1](o19).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[2](o22).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                    ],
                    2,
                )
        if self.enable_image_label_head:
            output.image_label_outputs = self.image_label_head(
                {"o10": o10, "o16": o16, "o19": o19, "o22": o22},
                tabular_values=tabular_values,
                tabular_mask=tabular_mask,
            )
        return output

    def load(self, state_dict: dict) -> None:
        state_dict_intersected = intersect_dicts(state_dict, self.state_dict())
        self.load_state_dict(state_dict_intersected, strict=False)


def build_model_from_config(
    train_config: DefaultConfig,
    *,
    num_classes: int | None = None,
    state_dict: dict | None = None,
    image_label_tasks: dict[str, ImageLabelTaskSpec] | None = None,
) -> HardcodedModel:
    resolved_image_label_tasks = _resolve_image_label_tasks(train_config, image_label_tasks)
    resolved_tabular_feature_names = _resolve_tabular_feature_names(train_config)
    model = HardcodedModel(
        input_channels=train_config.input_channels,
        num_classes=num_classes or train_config.num_classes,
        scale=train_config.model_scale,
        dfl_reg_max=train_config.dfl_reg_max,
        bbox_parametrization=train_config.bbox_parametrization,
        bbox_center_offset_limit=train_config.bbox_center_offset_limit,
        enable_output_masks=train_config.segmentation_output,
        num_prototype_masks=train_config.prototype_masks,
        prototype_layer=train_config.prototype_layer,
        separate_objectness_head=train_config.separate_objectness_head,
        include_stride4_output=train_config.include_stride4_output,
        image_label_tasks=resolved_image_label_tasks,
        enable_image_label_head=train_config.enable_image_label_head,
        image_label_head_feature_maps=train_config.image_label_head_feature_maps,
        image_label_head_projection_channels=train_config.image_label_head_projection_channels,
        image_label_head_mlp_hidden_layers=train_config.image_label_head_mlp_hidden_layers,
        image_label_head_mlp_hidden_channels=train_config.image_label_head_mlp_hidden_channels,
        image_label_head_dropout=train_config.image_label_head_dropout,
        enable_tabular_image_label_fusion=train_config.enable_tabular_image_label_fusion,
        tabular_transformer_variant=train_config.tabular_transformer_variant,
        tabular_feature_names=resolved_tabular_feature_names,
        tabular_transformer_d_model=train_config.tabular_transformer_d_model,
        tabular_transformer_num_heads=train_config.tabular_transformer_num_heads,
        tabular_transformer_num_layers=train_config.tabular_transformer_num_layers,
        tabular_transformer_feedforward_channels=train_config.tabular_transformer_feedforward_channels,
        tabular_transformer_dropout=train_config.tabular_transformer_dropout,
        tabular_transformer_v2_per_feature_value_projection=(
            train_config.tabular_transformer_v2_per_feature_value_projection
        ),
        tabular_transformer_v2_shared_mask_token=train_config.tabular_transformer_v2_shared_mask_token,
    )
    if state_dict is not None:
        model.load(state_dict)
    return model


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device_override: str | None = None,
) -> LoadedCheckpointModel:
    checkpoint = load_model_checkpoint(checkpoint_path)
    train_config = checkpoint["train_config"]
    device = select_device(device_override or getattr(train_config, "device", ""))
    model = build_model_from_config(
        train_config,
        state_dict=checkpoint["model_state_dict"],
    )
    model.eval().to(device)
    model.device = device  # type: ignore[attr-defined]
    return LoadedCheckpointModel(
        checkpoint=checkpoint,
        train_config=train_config,
        model=model,
        device=device,
    )
