"""
models/base.py

Abstract base classes defining the core interfaces for the concealment/cloaking pipeline.

All concrete model implementations (YOLOv8, LaMa, ESRGAN, PatchGAN, etc.) should inherit from
these base classes to ensure consistent API across the system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np
import torch


class BaseDetector(ABC):
    """
    Abstract base class for all object detection + segmentation models.

    Concrete implementations must provide detection with optional class filtering
    and the ability to merge instance masks into a single binary mask.
    """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        target_classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform object detection and instance segmentation.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (shape: H × W × 3), uint8
        target_classes : List[str], optional
            If provided, only return detections matching these class names.
            If None, return all detected objects.

        Returns
        -------
        List[Dict[str, Any]]
            List of detection dictionaries, each containing:
            - 'class'       : str                   → class name
            - 'confidence'  : float                 → detection score [0,1]
            - 'bbox'        : List[float]           → [x1, y1, x2, y2] in pixels
            - 'mask'        : np.ndarray            → binary mask (H × W), usually uint8 0/255
        """
        pass

    @abstractmethod
    def get_combined_mask(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Merge multiple instance masks into a single binary mask (union).

        Parameters
        ----------
        detections : List[Dict[str, Any]]
            Output from the detect() method

        Returns
        -------
        np.ndarray
            Combined binary mask (H × W), typically uint8 with values 0 or 255
            Returns array of zeros if no detections.
        """
        pass


class BaseInpainter(ABC):
    """
    Abstract base class for generative inpainting models (e.g. LaMa, MAT, etc.).

    Responsible for filling masked regions plausibly using surrounding context.
    """

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the regions specified by the mask.

        Parameters
        ----------
        image : np.ndarray
            Input RGB/BGR image (H × W × 3), uint8
        mask : np.ndarray
            Binary mask where >0 indicates regions to inpaint (H × W), usually uint8

        Returns
        -------
        np.ndarray
            Inpainted image of same shape and dtype as input (H × W × 3)
        """
        pass


class BaseShadowDetector(ABC):
    """
    Abstract base class for shadow detection / removal components.

    Typically used to extend the object mask to include cast shadows.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """
        Detect shadow regions associated with the provided object mask.

        Parameters
        ----------
        image : np.ndarray
            Original BGR image (H × W × 3)
        object_mask : np.ndarray
            Binary mask of detected objects (H × W)

        Returns
        -------
        np.ndarray
            Binary shadow mask (H × W), values 0 or 255
        """
        pass


class BaseEnhancer(ABC):
    """
    Abstract base class for image quality enhancement / super-resolution models
    (e.g. ESRGAN, Real-ESRGAN, SwinIR, etc.).
    """

    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image resolution and/or perceptual quality.

        Parameters
        ----------
        image : np.ndarray
            Input image (H × W × 3), usually RGB/BGR, uint8

        Returns
        -------
        np.ndarray
            Enhanced image (may have different resolution), uint8
        """
        pass


class BaseDiscriminator(ABC):
    """
    Abstract base class for realism / quality assessment models
    (e.g. PatchGAN-style discriminator, BRISQUE, MUSIQ, etc.).
    """

    @abstractmethod
    def evaluate(self, image: np.ndarray) -> float:
        """
        Assess the perceptual realism / quality of the input image.

        Parameters
        ----------
        image : np.ndarray
            Image to evaluate (H × W × 3), usually RGB/BGR, uint8

        Returns
        -------
        float
            Realism / quality score, typically normalized to [0, 1]
            (higher = more realistic / higher quality)
        """
        pass


# Optional: Type alias for convenience in other modules
Detection = Dict[str, Any]                  # single detection dict
Detections = List[Detection]                # list of detections
