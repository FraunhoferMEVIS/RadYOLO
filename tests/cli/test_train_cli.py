from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from radyolo import main
from radyolo.config.config import load_model_checkpoint


def test_train_cli(minimal_train_config: Path, tmp_path: Path) -> None:
    test_config = tmp_path / "train_separate_objectness.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["output_folder_base"] = (tmp_path / "runs").as_posix()
    config["experiment_name"] = "e2e_separate_objectness"
    config["separate_objectness_head"] = True
    config["nms_iou_threshold_candidates"] = [0.1, 0.2]
    with test_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    runner = CliRunner()

    result = runner.invoke(main, ["train", str(test_config)])

    assert result.exit_code == 0
    run_dir = Path(config["output_folder_base"]) / str(config["experiment_name"])
    best_path = run_dir / "weights_best.pt"
    copied_config = run_dir / "train_config.yaml"
    nms_optimization_csv = run_dir / "nms_optimization.csv"
    assert best_path.exists(), "weights_best.pt missing"
    assert (run_dir / "results.csv").exists(), "results.csv missing"
    assert nms_optimization_csv.exists(), "nms_optimization.csv missing"
    assert copied_config.exists(), "train_config.yaml missing"
    assert copied_config.read_text(encoding="utf-8") == test_config.read_text(encoding="utf-8")
    nms_rows = nms_optimization_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(nms_rows) == 5
    assert any("last,weights_last.pt,0.1" in row for row in nms_rows[1:])
    assert any("best,weights_best.pt,0.2" in row for row in nms_rows[1:])
    checkpoint = load_model_checkpoint(best_path)
    assert checkpoint["train_config"].separate_objectness_head is True
