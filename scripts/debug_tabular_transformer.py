from __future__ import annotations

# isort: skip_file

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from radyolo.models.model import TabularTransformerEncoder, TabularTransformerEncoderV2
from radyolo.training.optimizer import build_optimizer


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToyDatasetSpec:
    num_continuous: int = 5
    num_categorical: int = 3
    num_binary: int = 3

    @property
    def num_features(self) -> int:
        return self.num_continuous + self.num_categorical + self.num_binary


class SurvivalTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
    ) -> None:
        self.values = values
        self.observed_mask = observed_mask
        self.times = times
        self.events = events

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.values[index],
            self.observed_mask[index],
            self.times[index],
            self.events[index],
        )


class TabularSurvivalModel(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        encoder_version: str,
        num_features: int,
        num_bins: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_cls = {
            "v1": TabularTransformerEncoder,
            "v2": TabularTransformerEncoderV2,
        }.get(encoder_version)
        if encoder_cls is None:
            msg = f"Unsupported tabular encoder version '{encoder_version}'"
            raise ValueError(msg)
        self.encoder = encoder_cls(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, num_bins)

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values, observed_mask)
        return self.head(encoded)


def _load_yaml_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _coerce_int(raw: object, fallback: int) -> int:
    if isinstance(raw, bool):
        return fallback
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return fallback


def _coerce_float(raw: object, fallback: float) -> float:
    if isinstance(raw, bool):
        return fallback
    if isinstance(raw, (int, float)):
        return float(raw)
    return fallback


def _make_toy_survival_data(
    num_samples: int,
    spec: ToyDatasetSpec,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    continuous = torch.randn((num_samples, spec.num_continuous), generator=generator)
    continuous[:, 0] *= 20
    continuous[:, 1] *= 40

    cat0 = torch.randint(0, 4, (num_samples,), generator=generator)
    cat1 = torch.randint(0, 3, (num_samples,), generator=generator)
    cat2 = torch.randint(0, 5, (num_samples,), generator=generator)
    categorical = torch.stack(
        [
            (cat0.float() - 1.5) / 1.5,
            (cat1.float() - 1.0) / 1.0,
            (cat2.float() - 2.0) / 2.0,
        ],
        dim=1,
    )

    binary = torch.randint(0, 2, (num_samples, spec.num_binary), generator=generator).float()

    values = torch.cat([continuous, categorical, binary], dim=1)

    latent_risk = (
        0.95 * continuous[:, 0]
        - 0.70 * continuous[:, 1]
        + 0.45 * continuous[:, 2] * continuous[:, 3]
        + 0.35 * (cat0 == 3).float()
        + 0.30 * (cat1 == 2).float()
        - 0.40 * binary[:, 0]
        + 0.55 * binary[:, 1]
        + 0.30 * (binary[:, 2] * (cat2 >= 3).float())
        + 0.10 * torch.randn(num_samples, generator=generator)
    )

    hazard_rate = torch.exp(-1.0 + 0.80 * latent_risk).clamp(min=0.02, max=8.0)
    event_times = torch.distributions.Exponential(rate=hazard_rate).sample()

    censor_scale = float(torch.quantile(event_times, q=0.75).item())
    censor_times = torch.rand(num_samples, generator=generator) * (1.3 * censor_scale)

    observed_times = torch.minimum(event_times, censor_times).float()
    events = (event_times <= censor_times).to(dtype=torch.bool)

    missing_probability = torch.tensor(
        [0.06, 0.10, 0.08, 0.15, 0.05, 0.12, 0.10, 0.20, 0.07, 0.10, 0.12],
        dtype=torch.float32,
    )
    if missing_probability.shape[0] != spec.num_features:
        msg = "Missing-probability vector must match number of features"
        raise RuntimeError(msg)

    observed_mask = (
        torch.rand((num_samples, spec.num_features), generator=generator)
        > missing_probability.unsqueeze(0)
    )
    values = values.masked_fill(~observed_mask, 0.0)

    return values.float(), observed_mask, observed_times, events


def _make_survival_thresholds(times: torch.Tensor, num_bins: int) -> torch.Tensor:
    low = float(torch.quantile(times, q=0.10).item())
    high = float(torch.quantile(times, q=0.90).item())
    if high <= low:
        high = low + 1.0
    return torch.linspace(low, high, steps=num_bins, dtype=torch.float32)


def _survival_threshold_loss(
    logits: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    observed_survival = times.view(-1, 1) >= thresholds.view(1, -1)
    supervision_mask = torch.where(
        events.view(-1, 1),
        torch.ones_like(observed_survival),
        observed_survival,
    )
    targets = observed_survival.to(dtype=logits.dtype)
    per_bin_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    supervised_counts = supervision_mask.sum(dim=1).clamp(min=1)
    return ((per_bin_loss * supervision_mask).sum(dim=1) / supervised_counts).mean()


def _survival_bin_widths(thresholds: torch.Tensor) -> torch.Tensor:
    if thresholds.numel() == 1:
        return torch.ones_like(thresholds)
    trailing_width = thresholds[-1] - thresholds[-2]
    if trailing_width <= 0:
        trailing_width = thresholds.new_tensor(1.0)
    return torch.diff(thresholds, append=thresholds[-1:] + trailing_width)


def _risk_from_logits(logits: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    survival_probabilities = logits.sigmoid()
    return -(survival_probabilities * _survival_bin_widths(thresholds).view(1, -1)).sum(dim=1)


def _cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(events):
        return risk_scores.new_tensor(0.0)
    sorted_indices = torch.argsort(times, descending=True)
    sorted_risk = risk_scores[sorted_indices]
    sorted_events = events[sorted_indices]
    partial_log_likelihood = sorted_risk[sorted_events] - torch.logcumsumexp(sorted_risk, dim=0)[
        sorted_events
    ]
    if partial_log_likelihood.numel() == 0:
        return risk_scores.new_tensor(0.0)
    return -partial_log_likelihood.mean()


def _combined_survival_loss(
    logits: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    thresholds: torch.Tensor,
    threshold_weight: float,
    cox_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold_loss = _survival_threshold_loss(logits, times, events, thresholds)
    risk = _risk_from_logits(logits, thresholds)
    cox_loss = _cox_partial_likelihood_loss(risk, times, events)
    total = threshold_weight * threshold_loss + cox_weight * cox_loss
    return total, threshold_loss, cox_loss


def _concordance_index(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> float:
    risk = risk_scores.detach().cpu()
    t = times.detach().cpu()
    e = events.detach().cpu()

    concordant = 0.0
    comparable = 0.0

    n = int(t.shape[0])
    for i in range(n):
        if not bool(e[i]):
            continue
        for j in range(n):
            if i == j:
                continue
            if t[i] < t[j]:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / comparable if comparable > 0 else 0.0


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    thresholds: torch.Tensor,
    threshold_weight: float,
    cox_weight: float,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    all_logits: list[torch.Tensor] = []
    all_times: list[torch.Tensor] = []
    all_events: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_values_cpu, batch_mask_cpu, batch_times_cpu, batch_events_cpu in loader:
            batch_values = batch_values_cpu.to(device=device, dtype=torch.float32)
            batch_mask = batch_mask_cpu.to(device=device, dtype=torch.bool)
            batch_times = batch_times_cpu.to(device=device, dtype=torch.float32)
            batch_events = batch_events_cpu.to(device=device, dtype=torch.bool)

            logits = model(batch_values, batch_mask)
            loss, _, _ = _combined_survival_loss(
                logits,
                batch_times,
                batch_events,
                thresholds,
                threshold_weight,
                cox_weight,
            )

            batch_size = int(batch_values.shape[0])
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size

            all_logits.append(logits.detach())
            all_times.append(batch_times.detach())
            all_events.append(batch_events.detach())

    if total_examples == 0:
        msg = "Validation loader was empty"
        raise RuntimeError(msg)

    logits_tensor = torch.cat(all_logits, dim=0)
    times_tensor = torch.cat(all_times, dim=0)
    events_tensor = torch.cat(all_events, dim=0)

    full_threshold_loss = float(
        _survival_threshold_loss(logits_tensor, times_tensor, events_tensor, thresholds).item()
    )
    risk = _risk_from_logits(logits_tensor, thresholds)
    full_cox_loss = float(_cox_partial_likelihood_loss(risk, times_tensor, events_tensor).item())
    c_index = _concordance_index(risk, times_tensor, events_tensor)

    return total_loss / total_examples, {
        "val_threshold_loss": full_threshold_loss,
        "val_cox_loss": full_cox_loss,
        "val_c_index": c_index,
        "val_event_fraction": float(events_tensor.float().mean().item()),
    }


def parse_args() -> argparse.Namespace:
    default_config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    config_data = _load_yaml_config(default_config_path)

    default_seed = _coerce_int(config_data.get("seed"), 7)
    default_epochs = _coerce_int(config_data.get("epochs"), 30)
    default_batch_size = _coerce_int(config_data.get("batch_size"), 64)
    default_lr = _coerce_float(config_data.get("initial_learning_rate"), 2e-3)
    default_weight_decay = _coerce_float(config_data.get("weight_decay"), 1e-4)
    default_momentum = _coerce_float(config_data.get("momentum"), 0.9)
    default_num_bins = _coerce_int(config_data.get("image_label_survival_num_bins"), 12)
    default_threshold_weight = _coerce_float(config_data.get("image_label_survival_mtlr_gain"), 1.0)
    default_cox_weight = _coerce_float(config_data.get("image_label_survival_cox_gain"), 0.5)
    default_d_model = _coerce_int(config_data.get("tabular_transformer_d_model"), 64)
    default_num_heads = _coerce_int(config_data.get("tabular_transformer_num_heads"), 4)
    default_num_layers = _coerce_int(config_data.get("tabular_transformer_num_layers"), 2)
    default_ffn_width = _coerce_int(
        config_data.get("tabular_transformer_feedforward_channels"),
        128,
    )
    default_dropout = _coerce_float(config_data.get("tabular_transformer_dropout"), 0.1)
    default_grad_clip = _coerce_float(config_data.get("grad_clip_max_norm"), 2.0)
    default_muon_weight = _coerce_float(config_data.get("muon_weight"), 0.5)
    default_optimizer = str(config_data.get("optimizer", "SGD"))
    default_encoder_version = "v1"

    parser = argparse.ArgumentParser(
        description=(
            "Train a TabularTransformerEncoder on synthetic tabular survival data with missing "
            "values and print train/validation losses and validation metrics."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config_path)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--num-samples", type=int, default=1400)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--weight-decay", type=float, default=default_weight_decay)
    parser.add_argument("--momentum", type=float, default=default_momentum)
    parser.add_argument("--num-bins", type=int, default=default_num_bins)
    parser.add_argument("--optimizer", type=str, default=default_optimizer)
    parser.add_argument("--threshold-weight", type=float, default=default_threshold_weight)
    parser.add_argument("--cox-weight", type=float, default=default_cox_weight)
    parser.add_argument("--d-model", type=int, default=default_d_model)
    parser.add_argument("--num-heads", type=int, default=default_num_heads)
    parser.add_argument("--num-layers", type=int, default=default_num_layers)
    parser.add_argument("--ffn-width", type=int, default=default_ffn_width)
    parser.add_argument("--dropout", type=float, default=default_dropout)
    parser.add_argument("--grad-clip", type=float, default=default_grad_clip)
    parser.add_argument("--muon-weight", type=float, default=default_muon_weight)
    parser.add_argument(
        "--tabular-encoder-version",
        choices=("v1", "v2"),
        default=default_encoder_version,
    )
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)

    spec = ToyDatasetSpec()
    values, observed_mask, times, events = _make_toy_survival_data(
        num_samples=args.num_samples,
        spec=spec,
        seed=args.seed,
    )

    split_index = int(0.8 * values.shape[0])
    train_values = values[:split_index]
    train_mask = observed_mask[:split_index]
    train_times = times[:split_index]
    train_events = events[:split_index]

    val_values = values[split_index:]
    val_mask = observed_mask[split_index:]
    val_times = times[split_index:]
    val_events = events[split_index:]

    thresholds = _make_survival_thresholds(train_times, num_bins=args.num_bins)

    train_dataset = SurvivalTensorDataset(train_values, train_mask, train_times, train_events)
    val_dataset = SurvivalTensorDataset(val_values, val_mask, val_times, val_events)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularSurvivalModel(
        encoder_version=args.tabular_encoder_version,
        num_features=spec.num_features,
        num_bins=args.num_bins,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_channels=args.ffn_width,
        dropout=args.dropout,
    ).to(device)
    thresholds = thresholds.to(device)

    optimizer = build_optimizer(
        model=model,
        name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        decay=args.weight_decay,
        muon_weight=args.muon_weight,
    )

    LOGGER.info("Toy tabular survival experiment")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Config defaults source: %s", args.config)
    LOGGER.info(
        "Optimizer: %s | lr=%s | momentum=%s | weight_decay=%s | muon_weight=%s",
        args.optimizer,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.muon_weight,
    )
    LOGGER.info("Tabular encoder: %s", args.tabular_encoder_version)
    LOGGER.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))
    LOGGER.info(
        "Feature breakdown: continuous=%d, categorical=%d, binary=%d",
        spec.num_continuous,
        spec.num_categorical,
        spec.num_binary,
    )
    LOGGER.info(
        "Missing fraction (overall): %.3f, Event fraction: %.3f",
        float((~observed_mask).float().mean().item()),
        float(events.float().mean().item()),
    )
    LOGGER.info("%s", "-" * 88)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0
        seen_examples = 0

        for batch_values_cpu, batch_mask_cpu, batch_times_cpu, batch_events_cpu in train_loader:
            batch_values = batch_values_cpu.to(device=device, dtype=torch.float32)
            batch_mask = batch_mask_cpu.to(device=device, dtype=torch.bool)
            batch_times = batch_times_cpu.to(device=device, dtype=torch.float32)
            batch_events = batch_events_cpu.to(device=device, dtype=torch.bool)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_values, batch_mask)
            loss, _, _ = _combined_survival_loss(
                logits,
                batch_times,
                batch_events,
                thresholds,
                threshold_weight=args.threshold_weight,
                cox_weight=args.cox_weight,
            )
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            batch_size = int(batch_values.shape[0])
            seen_examples += batch_size
            running_train_loss += float(loss.item()) * batch_size

        train_loss = running_train_loss / max(seen_examples, 1)
        val_loss, val_metrics = _evaluate(
            model,
            val_loader,
            thresholds,
            threshold_weight=args.threshold_weight,
            cox_weight=args.cox_weight,
            device=device,
        )

        LOGGER.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_c_index=%.4f | "
            "val_thr=%.4f | val_cox=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["val_c_index"],
            val_metrics["val_threshold_loss"],
            val_metrics["val_cox_loss"],
        )

    LOGGER.info("%s", "-" * 88)
    final_val_loss, final_metrics = _evaluate(
        model,
        val_loader,
        thresholds,
        threshold_weight=args.threshold_weight,
        cox_weight=args.cox_weight,
        device=device,
    )
    LOGGER.info("Final train/val summary: val_loss=%.4f", final_val_loss)
    LOGGER.info("Validation metrics:")
    for key, value in final_metrics.items():
        LOGGER.info("  %s: %.6f", key, value)


if __name__ == "__main__":
    main()
