from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

from radyolo import main

if TYPE_CHECKING:
    from pathlib import Path


def test_create_json_labels_from_txt_cli_creates_json_files(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "case1.txt").write_text("0 0.5 0.5 0.5 0.2 0.2 0.2\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["create-json-labels-from-txt", str(labels_dir)])

    assert result.exit_code == 0
    json_path = labels_dir / "case1.json"
    assert json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {
            "class": 0,
            "box_relative_xyzwhd": [0.5, 0.5, 0.5, 0.2, 0.2, 0.2],
        }
    ]
