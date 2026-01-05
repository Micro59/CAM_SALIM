"""
utils/color_utils.py

Utilities for color and lighting matching between images, primarily using the LAB color space.
These functions are critical for realistic object insertion/removal in invisibility cloaking pipelines,
especially when replacing masked regions with background content while preserving plausible lighting.

Main features:
- Statistical color/luminance transfer (mean & std matching)
- Optional preservation of chromaticity (adjust only L channel)
- Seamless blending with Poisson or simple alpha methods
"""

import cv2
import numpy as np
from typing import Optional


def adjust_lighting_lab(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    preserve_color: bool = True
) -> np.ndarray:
    """
    Match the lighting (and optionally color) of a source image region to a target image
    using statistical color transfer in LAB color space.

    This is commonly used to make an extracted / inpainted region blend naturally
    with the surrounding scene lighting.

    Parameters
    ----------
    source : np.ndarray
        Source image (the content we're adjusting — usually the inpainted or background region)
        Shape: (H, W, 3), BGR, uint8
    target : np.ndarray
        Target/reference image (the scene we're matching to — usually the original input frame)
        Shape: (H, W, 3), BGR, uint8
    mask : np.ndarray
        Binary mask defining the region to adjust
        Shape: (H, W), uint8, values 0 or 255
    preserve_color : bool, optional
        If True: only match luminance (L channel) — preserves original colors (recommended)
        If False: match all LAB channels (mean & std) — can shift hue/saturation

    Returns
    -------
    np.ndarray
        Adjusted source image in BGR format, same shape/dtype as input
    """
    if source.shape[:2] != target.shape[:2] or source.shape[:2] != mask.shape:
        raise ValueError("Source, target, and mask must have matching spatial dimensions")

    # Convert both images to LAB (L: 0-100 scaled to 0-255, a/b: 0-255 centered ~128)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    mask_bool = mask > 0
    if not np.any(mask_bool):
        return source.copy()

    epsilon = 1e-6

    # Decide which channels to adjust
    channels = [0] if preserve_color else [0, 1, 2]  # L only, or L,a,b

    result_lab = source_lab.copy()

    for ch in channels:
        # Extract masked pixel values
        src_vals = source_lab[:, :, ch][mask_bool]
        tgt_vals = target_lab[:, :, ch][mask_bool]

        if len(src_vals) == 0 or len(tgt_vals) == 0:
            continue

        mu_src = np.mean(src_vals)
        sigma_src = np.std(src_vals) + epsilon
        mu_tgt = np.mean(tgt_vals)
        sigma_tgt = np.std(tgt_vals) + epsilon

        # Statistical color transfer: match mean and std
        adjusted = (source_lab[:, :, ch] - mu_src) * (sigma_tgt / sigma_src) + mu_tgt

        # Clip to valid LAB range
        if ch == 0:
            adjusted = np.clip(adjusted, 0, 255)     # L: 0–255
        else:
            adjusted = np.clip(adjusted, 0, 255)     # a,b: theoretically -128 to 127 → 0-255

        result_lab[:, :, ch] = adjusted

    # Convert back to BGR
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return result_bgr


def blend_seamless(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    blend_mode: str = 'poisson'
) -> np.ndarray:
    """
    Seamlessly composite foreground into background using Poisson blending or simple alpha.

    Poisson blending (OpenCV's seamlessClone) is usually superior for natural-looking results,
    especially around object boundaries with lighting gradients.

    Parameters
    ----------
    foreground : np.ndarray
        Image patch / content to insert (H × W × 3), BGR, uint8
    background : np.ndarray
        Destination image (H × W × 3), BGR, uint8
    mask : np.ndarray
        Binary mask of the foreground region (H × W), uint8 0/255
    blend_mode : str, optional
        - 'poisson'  → use cv2.seamlessClone (NORMAL_CLONE mode) — recommended
        - 'alpha'    → simple linear blending (faster but often shows seams)

    Returns
    -------
    np.ndarray
        Blended composite image, same shape as background
    """
    if foreground.shape != background.shape:
        raise ValueError("Foreground and background must have the same shape")

    mask = (mask > 0).astype(np.uint8) * 255  # ensure binary 0/255

    if blend_mode.lower() == 'poisson':
        # Compute center of mass for cloning location
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            # Fallback: image center
            cy, cx = mask.shape[0] // 2, mask.shape[1] // 2

        # Poisson seamless cloning (preserves gradients)
        result = cv2.seamlessClone(
            foreground,
            background,
            mask,
            (cx, cy),
            cv2.NORMAL_CLONE  # or cv2.MIXED_CLONE for slightly different behavior
        )

    else:  # 'alpha' — simple weighted blending
        mask_float = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_float] * 3, axis=-1)

        result = (foreground.astype(np.float32) * mask_3ch +
                  background.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)

    return result


# ────────────────────────────────────────────────────────────────
# Quick test / visualization block
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Dummy data for testing
    h, w = 400, 600
    target = np.full((h, w, 3), (180, 120, 90), dtype=np.uint8)   # darker background
    source = np.full((h, w, 3), (220, 200, 180), dtype=np.uint8)  # brighter patch

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), 120, 255, -1)

    adjusted = adjust_lighting_lab(source, target, mask, preserve_color=True)

    blended_poisson = blend_seamless(adjusted, target, mask, 'poisson')
    blended_alpha   = blend_seamless(adjusted, target, mask, 'alpha')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flat

    axes[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original Source")
    axes[1].imshow(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)); axes[1].set_title("Lighting Adjusted")
    axes[2].imshow(cv2.cvtColor(blended_poisson, cv2.COLOR_BGR2RGB)); axes[2].set_title("Poisson Blend")
    axes[3].imshow(cv2.cvtColor(blended_alpha, cv2.COLOR_BGR2RGB)); axes[3].set_title("Alpha Blend")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
