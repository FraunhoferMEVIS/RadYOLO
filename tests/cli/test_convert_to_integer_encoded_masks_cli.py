from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from radyolo import main


def test_convert_to_integer_encoded_masks_cli_invokes_converter(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    captured: dict[str, object] = {}

    def fake_convert_dataset_to_integer_encoded_masks(dataset_path: Path) -> Path:
        captured["dataset_path"] = dataset_path
        output_dir = dataset_path / "masks_integer_encoded"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    monkeypatch.setattr(
        "radyolo.convert_dataset_to_integer_encoded_masks",
        fake_convert_dataset_to_integer_encoded_masks,
    )

    runner = CliRunner()
    result = runner.invoke(main, ["convert-to-integer-encoded-masks", str(dataset_dir)])

    assert result.exit_code == 0
    assert captured["dataset_path"] == dataset_dir
    assert f"Created {dataset_dir / 'masks_integer_encoded'}" in result.output
