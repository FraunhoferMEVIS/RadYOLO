from __future__ import annotations

import blosc2
import numpy as np

from radyolo.data.preprocess_data import (
    DEFAULT_CACHE_SAFETY_FACTOR,
    DEFAULT_L3_CACHE_SIZE_PER_CORE_IN_BYTES,
    _compute_storage_shapes,
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