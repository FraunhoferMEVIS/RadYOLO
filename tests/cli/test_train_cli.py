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
    config["project"] = (tmp_path / "runs").as_posix()
    config["name"] = "e2e_separate_objectness"
    config["separate_objectness_head"] = True
    Path(config["project"]).mkdir(parents=True, exist_ok=True)
    with test_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    runner = CliRunner()

    result = runner.invoke(main, ["train", str(test_config)])

    assert result.exit_code == 0
    run_dir = Path(config["project"]) / str(config["name"])
    best_path = run_dir / "weights" / "best.pt"
    assert best_path.exists(), "best.pt missing"
    assert (run_dir / "results.csv").exists(), "results.csv missing"
    checkpoint = load_model_checkpoint(best_path)
    assert checkpoint["train_config"].separate_objectness_head is True
