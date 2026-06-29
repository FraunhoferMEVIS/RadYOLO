from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ObjectLabelEntry(TypedDict):
    class_: int
    box_relative_xyzwhd: tuple[float, float, float, float, float, float]


_TXT_DEPRECATION_WARNED = False


def _warn_txt_deprecated() -> None:
    global _TXT_DEPRECATION_WARNED
    if _TXT_DEPRECATION_WARNED:
        return
    LOGGER.warning(
        "Object labels from .txt files are deprecated. "
        "Please migrate labels to .json files with entries containing "
        "'class' and 'box_relative_xyzwhd' (e.g. using "
        "'radyolo create-json-labels-from-txt <path-to-label-dir>')."
    )
    _TXT_DEPRECATION_WARNED = True


def reset_txt_label_deprecation_warning() -> None:
    """Reset TXT deprecation warning state.

    This is primarily intended for tests that need deterministic warning assertions.
    """
    global _TXT_DEPRECATION_WARNED
    _TXT_DEPRECATION_WARNED = False


def _parse_label_line(
    line: str,
    *,
    source: Path,
) -> ObjectLabelEntry:
    parts = line.strip().split()
    if len(parts) != 7:
        msg = (
            "Label line must have 7 entries: class x y z w h d (normalized), "
            f"but has {len(parts)} entries in {source}."
        )
        raise ValueError(msg)
    class_id = int(float(parts[0]))
    x, y, z, w, h, d = map(float, parts[1:])
    return {
        "class_": class_id,
        "box_relative_xyzwhd": (x, y, z, w, h, d),
    }


def _load_case_object_labels_from_txt(label_path: Path) -> list[ObjectLabelEntry]:
    _warn_txt_deprecated()
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    entries: list[ObjectLabelEntry] = []
    for line in lines:
        if not line.strip():
            continue
        entries.append(_parse_label_line(line, source=label_path))
    return entries


def _load_case_object_labels_from_json(label_path: Path) -> list[ObjectLabelEntry]:
    parsed = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        msg = f"Label file {label_path} must contain a list of objects."
        raise TypeError(msg)

    entries: list[ObjectLabelEntry] = []
    for index, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            msg = f"Label entry {index} in {label_path} must be a JSON object."
            raise TypeError(msg)

        if "class" not in entry:
            msg = f"Label entry {index} in {label_path} is missing 'class'."
            raise ValueError(msg)
        if "box_relative_xyzwhd" not in entry:
            msg = f"Label entry {index} in {label_path} is missing 'box_relative_xyzwhd'."
            raise ValueError(msg)

        class_id = int(entry["class"])
        box = entry["box_relative_xyzwhd"]
        if not isinstance(box, list) or len(box) != 6:
            msg = (
                f"Label entry {index} in {label_path} must define "
                "'box_relative_xyzwhd' as a list with 6 numbers."
            )
            raise ValueError(msg)

        x, y, z, w, h, d = map(float, box)
        entries.append(
            {
                "class_": class_id,
                "box_relative_xyzwhd": (x, y, z, w, h, d),
            }
        )
    return entries


def load_case_object_labels(
    labels_dir: Path,
    case_name: str,
) -> list[ObjectLabelEntry]:
    json_path = labels_dir / f"{case_name}.json"
    txt_path = labels_dir / f"{case_name}.txt"

    if json_path.exists():
        return _load_case_object_labels_from_json(json_path)
    if txt_path.exists():
        return _load_case_object_labels_from_txt(txt_path)

    msg = (
        f"No object labels found for case '{case_name}' in {labels_dir}. "
        "Expected either a .json or .txt label file."
    )
    raise FileNotFoundError(msg)


def create_json_labels_from_txt(
    labels_dir: Path,
) -> list[Path]:
    if not labels_dir.exists() or not labels_dir.is_dir():
        msg = f"Label directory not found: {labels_dir}"
        raise FileNotFoundError(msg)

    created_paths: list[Path] = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        entries = _load_case_object_labels_from_txt(txt_path)
        json_payload = [
            {
                "class": entry["class_"],
                "box_relative_xyzwhd": list(entry["box_relative_xyzwhd"]),
            }
            for entry in entries
        ]
        json_path = txt_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(json_payload, indent=2),
            encoding="utf-8",
        )
        created_paths.append(json_path)
    return created_paths
