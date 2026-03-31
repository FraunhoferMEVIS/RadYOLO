from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from radyolo import main


def test_train_cli(minimal_train_config: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["train", str(minimal_train_config)])

    assert result.exit_code == 0
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    run_dir = Path(config["project"]) / str(config["name"])
    assert (run_dir / "weights" / "best.pt").exists(), "best.pt missing"
    assert (run_dir / "results.csv").exists(), "results.csv missing"
