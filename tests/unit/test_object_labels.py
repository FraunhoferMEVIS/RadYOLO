from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from radyolo.data import data as data_module
from radyolo.data import object_labels as object_labels_module

if TYPE_CHECKING:
    from pathlib import Path


def test_load_case_object_labels_prefers_json_over_txt(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "case_a.txt").write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n", encoding="utf-8")
    (labels_dir / "case_a.json").write_text(
        json.dumps(
            [
                {
                    "class": 1,
                    "box_relative_xyzwhd": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                }
            ]
        ),
        encoding="utf-8",
    )

    entries = object_labels_module.load_case_object_labels(labels_dir, "case_a")

    assert len(entries) == 1
    assert entries[0]["class_"] == 1
    assert entries[0]["box_relative_xyzwhd"] == (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


def test_load_case_object_labels_warns_for_txt(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "case_b.txt").write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n", encoding="utf-8")

    object_labels_module.reset_txt_label_deprecation_warning()
    caplog.set_level("WARNING")

    entries = object_labels_module.load_case_object_labels(labels_dir, "case_b")

    assert len(entries) == 1
    assert "deprecated" in caplog.text.lower()


def test_load_case_object_labels_raises_if_missing(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        object_labels_module.load_case_object_labels(labels_dir, "missing_case")


def test_create_json_labels_from_txt_writes_json_files(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "case_a.txt").write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n", encoding="utf-8")
    (labels_dir / "case_b.txt").write_text("2 0.7 0.6 0.5 0.4 0.3 0.2\n", encoding="utf-8")

    created = object_labels_module.create_json_labels_from_txt(labels_dir)

    assert len(created) == 2
    assert (labels_dir / "case_a.json").exists()
    payload = json.loads((labels_dir / "case_a.json").read_text(encoding="utf-8"))
    assert payload == [
        {
            "class": 0,
            "box_relative_xyzwhd": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    ]


def test_data_load_labels_supports_json(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "case_c.json").write_text(
        json.dumps(
            [
                {
                    "class": 3,
                    "box_relative_xyzwhd": [0.11, 0.22, 0.33, 0.44, 0.55, 0.66],
                }
            ]
        ),
        encoding="utf-8",
    )

    classes, bboxes = data_module._load_labels(tmp_path, "case_c")  # noqa: SLF001

    assert classes.shape == (1, 1)
    assert bboxes.shape == (1, 6)
    assert float(classes[0, 0].item()) == pytest.approx(3.0)
    assert [float(v) for v in bboxes[0].tolist()] == pytest.approx(
        [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
    )
