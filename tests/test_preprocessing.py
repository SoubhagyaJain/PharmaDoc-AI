import cv2
import numpy as np
import pytest

from pharmadoc_ai.preprocessing import (
    document_region,
    grayscale,
    is_blank,
    preprocess,
    resize_image,
    rotate_image,
)


def test_original_is_unchanged_and_blank_is_safe():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    copied = image.copy()
    assert is_blank(image)
    for profile in ("raw", "contrast", "adaptive", "deskew"):
        prepared = preprocess(image, profile)
        assert prepared.image.dtype == np.uint8
        assert np.array_equal(copied, image)
    assert document_region(image) is None


def test_rotation_box_maps_back():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    prepared = preprocess(image, "raw", rotation=90)
    # Whole rotated image corresponds to the whole original, within a pixel.
    h, w = prepared.image.shape[:2]
    assert prepared.original_box((0, 0, w, h)) == pytest.approx((0, 0, 1, 1), abs=0.01)


def test_resize_and_grayscale():
    image = np.zeros((100, 400, 3), np.uint8)
    assert resize_image(image, 200).shape == (50, 200, 3)
    assert grayscale(image).shape == (100, 400)
    with pytest.raises(ValueError):
        preprocess(image, "invented")


def test_small_skew_is_reduced():
    image = np.full((500, 800, 3), 255, np.uint8)
    for y in range(100, 400, 40):
        cv2.putText(
            image, "SYNTHETIC DOCUMENT TEXT", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
    skewed, _ = rotate_image(image, 4)
    prepared = preprocess(skewed, "deskew")
    assert not np.allclose(prepared.inverse, np.eye(3))
