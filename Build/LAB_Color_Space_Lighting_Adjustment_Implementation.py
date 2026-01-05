import cv2
import numpy as np


def adjust_lighting_lab(source_img, target_img, mask):
    """
    Adjust lighting of source image region to match target background.

    Parameters:
    -----------
    source_img : ndarray
        Source image in BGR format
    target_img : ndarray
        Target/background image in BGR format
    mask : ndarray
        Binary mask indicating region to adjust

    Returns:
    --------
    ndarray : Adjusted image in BGR format
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Extract luminance channels
    L_source = source_lab[:, :, 0]
    L_target = target_lab[:, :, 0]

    # Calculate statistics for masked regions
    mask_bool = mask > 0
    epsilon = 1e-6

    mu_source = np.mean(L_source[mask_bool])
    sigma_source = np.std(L_source[mask_bool]) + epsilon
    mu_target = np.mean(L_target[mask_bool])
    sigma_target = np.std(L_target[mask_bool]) + epsilon

    # Apply statistical transfer (simple mean & std matching)
    L_adjusted = (L_source - mu_source) * (sigma_target / sigma_source) + mu_target
    L_adjusted = np.clip(L_adjusted, 0, 255)

    # Reconstruct adjusted image
    result_lab = source_lab.copy()
    result_lab[:, :, 0] = L_adjusted
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return result_bgr
