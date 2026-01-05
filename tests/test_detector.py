# tests/test_detector.py
"""
Unit tests for YOLOv8-Seg based object detector.
"""

import pytest
import numpy as np
import torch

from models.detector import YOLOv8SegDetector
from config import DetectorConfig


@pytest.fixture(scope="module")
def detector():
    """Shared detector instance (loaded once per module)."""
    config = DetectorConfig(
        weights="weights/yolov8m-seg.pt",           # ← use real path or mock in CI
        conf_threshold=0.25,
        iou_threshold=0.45,
        target_classes=["person"]
    )
    return YOLOv8SegDetector(config)


@pytest.fixture
def random_rgb_image():
    """Random 640×480 RGB image (BGR uint8)."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def black_image():
    """Pure black image — very unlikely to trigger detections."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_detector_initialization(detector):
    assert detector.model is not None
    assert isinstance(detector.model, torch.nn.Module)
    assert detector.device.type in {"cpu", "cuda"}


def test_detect_returns_list_of_dicts(detector, random_rgb_image):
    detections = detector.detect(random_rgb_image)
    assert isinstance(detections, list)
    if detections:  # may be empty — that's ok
        det = detections[0]
        assert isinstance(det, dict)
        assert all(k in det for k in ["class", "confidence", "bbox", "mask"])


def test_detection_mask_matches_image_shape(detector, random_rgb_image):
    detections = detector.detect(random_rgb_image)
    for det in detections:
        mask = det["mask"]
        assert mask.ndim == 2
        assert mask.shape == random_rgb_image.shape[:2]
        assert mask.dtype in {np.uint8, np.bool_, np.float32}


def test_combined_mask_is_binary(detector, random_rgb_image):
    detections = detector.detect(random_rgb_image)
    if detections:
        combined = detector.get_combined_mask(detections)
        assert combined is not None
        assert combined.dtype == np.uint8
        assert set(np.unique(combined)) <= {0, 255}


def test_get_combined_mask_empty_list_returns_none(detector):
    result = detector.get_combined_mask([])
    assert result is None


def test_target_class_filtering(detector, random_rgb_image):
    # Filter to a class very unlikely to appear in random noise
    detections = detector.detect(random_rgb_image, target_classes=["airplane"])
    for det in detections:
        assert det["class"] == "airplane"


def test_no_detections_on_black_image(detector, black_image):
    detections = detector.detect(black_image)
    assert len(detections) == 0
