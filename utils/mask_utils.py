"""
utils/mask_utils.py

Collection of helper functions for binary mask refinement, combination, and analysis.
Used in object removal / invisibility cloaking pipelines to improve mask quality
before inpainting, reduce artifacts, and handle shadow integration.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def refine_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    close_iterations: int = 2,
    open_iterations: int = 1,
    dilation: int = 0
) -> np.ndarray:
    """
    Refine a binary segmentation mask using morphological operations.

    Operations in order:
    1. Closing → fills small holes inside the object
    2. Opening → removes small noise / isolated pixels
    3. Optional dilation → expands boundary (useful before inpainting)

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H × W), values typically 0 or 255 (uint8)
    kernel_size : int, optional
        Diameter of the elliptical structuring element (default: 5)
    close_iterations : int, optional
        Number of closing operations (default: 2)
    open_iterations : int, optional
        Number of opening operations (default: 1)
    dilation : int, optional
        Additional boundary expansion in pixels (default: 0)

    Returns
    -------
    np.ndarray
        Refined binary mask of same shape and dtype as input
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be 2D (H × W)")

    # Ensure mask is binary uint8 (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255

    # Elliptical kernel gives smoother, more natural boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Fill holes inside objects
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    # Remove small noise blobs
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    # Optional boundary expansion (helps avoid edge artifacts in inpainting)
    if dilation > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation * 2 + 1, dilation * 2 + 1)
        )
        refined = cv2.dilate(refined, dilate_kernel, iterations=1)

    return refined


def create_edge_mask(
    mask: np.ndarray,
    inner_erosion: int = 2,
    outer_dilation: int = 5
) -> np.ndarray:
    """
    Generate a boundary/transition mask around the object.

    Useful for:
    - Boundary-aware inpainting
    - Feathering/blending regions
    - Guided refinement near object edges

    Parameters
    ----------
    mask : np.ndarray
        Binary object mask (H × W)
    inner_erosion : int, optional
        Erosion iterations inward (default: 2)
    outer_dilation : int, optional
        Dilation iterations outward (default: 5)

    Returns
    -------
    np.ndarray
        Binary edge mask (H × W), values 0 or 255
        Represents the transition zone between object and background
    """
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Expand outward
    dilated = cv2.dilate(mask, kernel, iterations=outer_dilation)

    # Shrink inward
    eroded = cv2.erode(mask, kernel, iterations=inner_erosion)

    # Edge region = dilated area minus eroded area
    edge_mask = cv2.subtract(dilated, eroded)

    return edge_mask


def unify_masks(
    object_mask: np.ndarray,
    shadow_mask: Optional[np.ndarray] = None,
    method: str = 'union'
) -> np.ndarray:
    """
    Combine object mask and optional shadow mask into a single mask to inpaint.

    Parameters
    ----------
    object_mask : np.ndarray
        Binary mask of detected objects
    shadow_mask : np.ndarray, optional
        Binary mask of detected shadows
    method : str, optional
        Combination strategy:
        - 'union'         → inpaint object + shadow (default)
        - 'intersection'  → only where both overlap
        - 'object_only'   → ignore shadow mask

    Returns
    -------
    np.ndarray
        Combined binary mask (H × W), uint8 0/255
    """
    object_mask = (object_mask > 0).astype(np.uint8) * 255

    if shadow_mask is None or method == 'object_only':
        return object_mask

    shadow_mask = (shadow_mask > 0).astype(np.uint8) * 255

    if method == 'union':
        return np.maximum(object_mask, shadow_mask)
    elif method == 'intersection':
        return np.minimum(object_mask, shadow_mask)
    else:
        raise ValueError(f"Unsupported unification method: '{method}'. "
                         "Use 'union', 'intersection', or 'object_only'.")


def get_mask_bbox(
    mask: np.ndarray,
    padding: int = 10
) -> Tuple[int, int, int, int]:
    """
    Compute the tight bounding box of the non-zero region in the mask,
    with optional padding.

    Useful for:
    - Cropping regions for faster inpainting
    - Focusing processing on relevant areas

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H × W)
    padding : int, optional
        Pixels to add around the bounding box (default: 10)

    Returns
    -------
    Tuple[int, int, int, int]
        (x1, y1, x2, y2) coordinates of the padded bounding box
    """
    mask = (mask > 0).astype(np.uint8)

    coords = np.where(mask)
    if len(coords[0]) == 0:
        # No mask pixels → return full image
        return (0, 0, mask.shape[1], mask.shape[0])

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Apply padding with boundary clamping
    h, w = mask.shape
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w, x_max + padding)
    y2 = min(h, y_max + padding)

    return (x1, y1, x2, y2)


# Quick test / visualization block
if __name__ == "__main__":
    # Dummy mask for testing
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_mask, (128, 128), 60, 255, -1)  # filled circle

    refined = refine_mask(test_mask, kernel_size=5, close_iterations=2, dilation=3)
    edge = create_edge_mask(test_mask, inner_erosion=3, outer_dilation=8)
    bbox = get_mask_bbox(test_mask, padding=15)

    print(f"Bounding box with padding: {bbox}")

    # Optional: visualize (requires matplotlib)
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # axs[0].imshow(test_mask, cmap='gray'); axs[0].set_title("Original")
    # axs[1].imshow(refined, cmap='gray'); axs[1].set_title("Refined")
    # axs[2].imshow(edge, cmap='gray'); axs[2].set_title("Edge Mask")
    # plt.show()
