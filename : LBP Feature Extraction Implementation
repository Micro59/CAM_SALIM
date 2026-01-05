from skimage.feature import local_binary_pattern
import numpy as np
import cv2


def extract_lbp_features(image, P=8, R=1, method='uniform'):
    """
    Extract Local Binary Pattern (LBP) features from an image.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or BGR color)
    P : int, optional
        Number of circularly symmetric neighbor points (default: 8)
    R : int or float, optional
        Radius of circle (default: 1)
    method : str, optional
        LBP variant to use: 'uniform', 'default', 'ror', 'var' (default: 'uniform')

    Returns
    -------
    tuple
        (lbp_image, histogram)
        - lbp_image : ndarray of the computed LBP image
        - histogram : normalized histogram of LBP patterns (1D array)
    """
    # Convert to grayscale if the input is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)  # ensure correct dtype

    # Compute LBP
    lbp = local_binary_pattern(gray, P=P, R=R, method=method)

    # Determine number of bins for the histogram
    if method == 'uniform':
        n_bins = P + 2
    else:
        n_bins = 2 ** P

    # Compute normalized histogram
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return lbp, hist


def compare_lbp_histograms(hist1, hist2, method='chi-square'):
    """
    Compare two LBP histograms using a specified distance metric.

    Parameters
    ----------
    hist1 : ndarray
        First normalized histogram
    hist2 : ndarray
        Second normalized histogram (must have same length as hist1)
    method : str, optional
        Distance metric: 'chi-square', 'euclidean', 'intersection' (default: 'chi-square')

    Returns
    -------
    float
        Distance/similarity score between the two histograms
        (lower is better for chi-square & euclidean, higher is better for intersection)
    """
    if hist1.shape != hist2.shape:
        raise ValueError("Histograms must have the same dimensions")

    if method == 'chi-square':
        epsilon = 1e-10
        distance = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))
    elif method == 'euclidean':
        distance = np.linalg.norm(hist1 - hist2)
    elif method == 'intersection':
        distance = 1 - np.sum(np.minimum(hist1, hist2))
    else:
        raise ValueError(f"Unsupported comparison method: {method}")

    return distance
