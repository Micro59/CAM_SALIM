"""
models/detector.py

YOLOv8-Segmentation concrete implementation of the BaseDetector interface.
Handles object detection + instance segmentation with class filtering and mask resizing.
Supports both single-image and batch inference.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO

from .base import BaseDetector
# Assuming config is imported from your config module
from config.config import DetectorConfig


class YOLOv8SegDetector(BaseDetector):
    """
    YOLOv8-Segmentation model wrapper implementing the BaseDetector interface.

    Features:
    - Automatic device selection (CUDA if available, else CPU)
    - Optional FP16 (half-precision) inference on GPU
    - Warm-up inference to reduce first-run latency
    - Class filtering via target_classes
    - Proper mask resizing to original image resolution
    - Combined mask generation (union of all instance masks)
    - Batch inference support
    """

    def __init__(self, config: DetectorConfig):
        """
        Initialize the YOLOv8-Seg detector from configuration.

        Parameters
        ----------
        config : DetectorConfig
            Configuration object containing model weights, thresholds, input size, etc.
        """
        self.config = config

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"YOLOv8SegDetector using device: {self.device}")

        # Load model
        self.model = YOLO(config.weights)
        self.model.to(self.device)

        # Enable FP16 if on GPU
        if self.device.type == "cuda":
            self.model.model.half()  # FP16 inference

        # Perform warm-up to initialize buffers / reduce first inference latency
        self._warmup()

    def _warmup(self) -> None:
        """Run dummy inference to warm up the model."""
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_img, verbose=False)

    def detect(
        self,
        image: np.ndarray,
        target_classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect and segment objects in a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (H × W × 3), uint8
        target_classes : List[str], optional
            Classes to keep (overrides config.target_classes if provided)

        Returns
        -------
        List[Dict[str, Any]]
            List of detection dictionaries (see BaseDetector.detect docstring)
        """
        # Use config classes if not overridden
        if target_classes is None:
            target_classes = self.config.target_classes

        # Run inference
        results = self.model(
            image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            verbose=False
        )[0]

        return self._parse_result(results, image.shape, target_classes)

    def _parse_result(
        self,
        result: Any,
        orig_shape: Tuple[int, int, int],
        target_classes: List[str]
    ) -> List[Dict[str, Any]]:
        """Internal helper to parse Ultralytics results into standardized format."""
        detections = []

        if result.masks is None:
            return detections

        masks = result.masks.data.cpu().numpy()         # (N, H_model, W_model)
        boxes = result.boxes.xyxy.cpu().numpy()         # (N, 4)
        classes = result.boxes.cls.cpu().numpy()        # (N,)
        confs = result.boxes.conf.cpu().numpy()         # (N,)

        orig_h, orig_w = orig_shape[:2]

        for mask, box, cls_id, conf in zip(masks, boxes, classes, confs):
            class_name = self.model.names[int(cls_id)]

            if class_name not in target_classes:
                continue

            # Resize segmentation mask to original image size
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Optional: ensure binary mask (some models return probabilities)
            # mask_resized = (mask_resized > 0).astype(np.uint8) * 255

            detections.append({
                "class": class_name,
                "confidence": float(conf),
                "bbox": box.tolist(),
                "mask": mask_resized
            })

        return detections

    def get_combined_mask(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Merge all instance masks into a single binary mask (union).

        Parameters
        ----------
        detections : List[Dict[str, Any]]
            Output from detect() method

        Returns
        -------
        np.ndarray
            Binary mask (H × W), uint8 values 0 or 255
            Returns zero array if no detections
        """
        if not detections:
            # Return empty mask with shape matching first image (or default)
            # In practice, caller should know original shape — here we return None-like
            return np.zeros((1, 1), dtype=np.uint8) * 255  # placeholder

        # Start with first mask
        combined = detections[0]["mask"].copy()

        # Logical OR (maximum) with remaining masks
        for det in detections[1:]:
            combined = np.maximum(combined, det["mask"])

        return (combined > 0).astype(np.uint8) * 255

    def detect_batch(
        self,
        images: List[np.ndarray],
        target_classes: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batched detection on multiple images.

        Parameters
        ----------
        images : List[np.ndarray]
            List of input images (each H × W × 3, BGR, uint8)
        target_classes : List[str], optional

        Returns
        -------
        List[List[Dict[str, Any]]]
            One list of detections per input image
        """
        if target_classes is None:
            target_classes = self.config.target_classes

        results = self.model(
            images,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            verbose=False
        )

        batch_detections = []
        for result, img in zip(results, images):
            detections = self._parse_result(result, img.shape, target_classes)
            batch_detections.append(detections)

        return batch_detections


# Quick test / validation block
if __name__ == "__main__":
    from config.config import DetectorConfig

    cfg = DetectorConfig(
        weights="yolov8m-seg.pt",
        conf_threshold=0.25,
        iou_threshold=0.45,
        input_size=640,
        target_classes=["person"]
    )

    detector = YOLOv8SegDetector(cfg)
    print("YOLOv8SegDetector initialized successfully.")
