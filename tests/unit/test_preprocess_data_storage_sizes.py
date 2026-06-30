from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool

import blosc2
import numpy as np

from radyolo.config.spatial import PatchSize
from radyolo.data.preprocess_data import (
    DEFAULT_CACHE_SAFETY_FACTOR,
    DEFAULT_L3_CACHE_SIZE_PER_CORE_IN_BYTES,
    _compute_storage_shapes,
    preprocess_dataset,
    _save_preprocessed_image,
)


def test_compute_smart_storage_shapes_is_dtype_aware() -> None:
    image_shape = (128, 128, 128)

    float32_block, float32_chunk = _compute_storage_shapes(image_shape, np.dtype(np.float32))
    float16_block, float16_chunk = _compute_storage_shapes(image_shape, np.dtype(np.float16))

    assert all(0 < dim <= bound for dim, bound in zip(float32_block, image_shape, strict=True))
    assert all(0 < dim <= bound for dim, bound in zip(float32_chunk, image_shape, strict=True))
    assert all(0 < dim <= bound for dim, bound in zip(float16_block, image_shape, strict=True))
    assert all(0 < dim <= bound for dim, bound in zip(float16_chunk, image_shape, strict=True))

    l1_target = int(32768 * DEFAULT_CACHE_SAFETY_FACTOR)
    l3_target = int(DEFAULT_L3_CACHE_SIZE_PER_CORE_IN_BYTES * DEFAULT_CACHE_SAFETY_FACTOR)
    assert int(np.prod(float32_block)) * np.dtype(np.float32).itemsize <= l1_target
    assert int(np.prod(float32_chunk)) * np.dtype(np.float32).itemsize <= l3_target
    assert int(np.prod(float16_block)) * np.dtype(np.float16).itemsize <= l1_target
    assert int(np.prod(float16_chunk)) * np.dtype(np.float16).itemsize <= l3_target

    assert int(np.prod(float16_block)) >= int(np.prod(float32_block))
    assert int(np.prod(float16_chunk)) >= int(np.prod(float32_chunk))


def test_save_preprocessed_image_uses_smart_blosc2_chunks_and_blocks(tmp_path) -> None:
    image = np.zeros((128, 128, 128), dtype=np.float32)
    expected_block, expected_chunk = _compute_storage_shapes(tuple(image.shape), image.dtype)

    _save_preprocessed_image(
        image=image,
        out_images_dir=tmp_path,
        case_id="case_blosc2",
        use_full_image_chunk=False,
    )

    blosc_array = blosc2.open(str(tmp_path / "case_blosc2.b2nd"), mode="r")
    assert blosc_array.blocks == expected_block
    assert blosc_array.chunks == expected_chunk


def test_save_preprocessed_image_uses_full_image_chunk_for_fixed_extent(tmp_path) -> None:
    image = np.zeros((20, 18, 16), dtype=np.float32)

    _save_preprocessed_image(
        image=image,
        out_images_dir=tmp_path,
        case_id="case_extent",
        use_full_image_chunk=True,
    )

    blosc_array = blosc2.open(str(tmp_path / "case_extent.b2nd"), mode="r")
    assert blosc_array.chunks == image.shape


def test_preprocess_dataset_falls_back_to_serial_when_process_pool_breaks(
    synthetic_dataset,
    tmp_path,
    monkeypatch,
) -> None:
    output_dataset = tmp_path / "preprocessed_fallback"
    original_preprocess_cases_serially = None

    def _broken_pool(*_args, **_kwargs) -> None:
        raise BrokenProcessPool("simulated worker crash")

    def _recording_serial(*args, **kwargs) -> None:
        nonlocal original_preprocess_cases_serially
        serial_calls.append((args, kwargs))
        assert original_preprocess_cases_serially is not None
        original_preprocess_cases_serially(*args, **kwargs)

    serial_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    import radyolo.data.preprocess_data as preprocess_data_module

    original_preprocess_cases_serially = preprocess_data_module._preprocess_cases_serially
    monkeypatch.setattr(
        preprocess_data_module,
        "_preprocess_cases_in_process_pool",
        _broken_pool,
    )
    monkeypatch.setattr(
        preprocess_data_module,
        "_preprocess_cases_serially",
        _recording_serial,
    )

    result = preprocess_dataset(
        dataset_path=synthetic_dataset,
        output_path=output_dataset,
        workers=2,
        target_image_size=PatchSize(x=16, y=16, z=32),
    )

    assert result == output_dataset
    assert len(serial_calls) == 1
    assert (output_dataset / "images" / "case1.b2nd").exists()
    assert (output_dataset / "masks" / "case1.b2nd").exists()
