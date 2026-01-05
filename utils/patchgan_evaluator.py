# patchgan_evaluator.py
"""
PatchGAN-based Image Realism Evaluator

A convenient wrapper around a trained PatchGAN discriminator
to evaluate perceptual realism of images (real or generated).

Main features:
  - Single image evaluation
  - Batch evaluation
  - Spatial realism heatmap generation
  - Threshold-based quality gating
  - Detailed quality report with weak region detection

Typical usage:
    evaluator = PatchGANEvaluator(config)
    score = evaluator.evaluate(image_bgr)
    assessment = evaluator.get_quality_assessment(image_bgr)
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any

# Replace these with your actual imports
# from models.discriminator import PatchGANDiscriminator
# from configs.discriminator import DiscriminatorConfig
from your_project.models import PatchGANDiscriminator      # ← update this
from your_project.configs import DiscriminatorConfig       # ← update this


class BaseDiscriminator:
    """Placeholder base class – replace with your actual base class if needed."""
    pass


class PatchGANEvaluator(BaseDiscriminator):
    """
    Evaluator wrapper for a trained PatchGAN discriminator.
    Provides high-level methods to assess image realism.
    """

    def __init__(self, config: DiscriminatorConfig):
        """
        Args:
            config: DiscriminatorConfig with at least:
                - weights (str): path to trained checkpoint
                - patch_size (int): typically 70 for classic PatchGAN
                - realism_threshold (float): decision boundary, e.g. 0.9
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model architecture
        self.model = PatchGANDiscriminator(input_channels=3, ndf=64)

        # Load trained weights
        checkpoint = torch.load(config.weights, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.patch_size = config.patch_size
        self.threshold = config.realism_threshold

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 numpy image → normalized NCHW tensor [-1,1]."""
        # BGR → RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()

        # [0,255] → [-1,1]
        tensor = torch.from_numpy(image).float() / 127.5 - 1.0
        # HWC → NCHW + batch dim
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device)

    def evaluate(self, image: np.ndarray) -> float:
        """
        Compute average realism score (probability) across all patches.

        Returns:
            float: score ∈ [0, 1]  (higher = more realistic)
        """
        tensor = self._preprocess(image)

        with torch.no_grad():
            pred = self.model(tensor)
            prob = torch.sigmoid(pred)
            score = prob.mean().item()

        return score

    def evaluate_batch(self, images: List[np.ndarray]) -> List[float]:
        """Efficient batch inference for multiple images."""
        if not images:
            return []

        tensors = [self._preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            preds = self.model(batch)
            probs = torch.sigmoid(preds)
            # Average over spatial dimensions → one score per image
            scores = probs.view(len(images), -1).mean(dim=1)

        return scores.cpu().tolist()

    def evaluate_with_heatmap(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Returns overall score + upsampled realism heatmap (same size as input image).

        Returns:
            Tuple[float, np.ndarray]: (mean_score, heatmap[H,W])
        """
        tensor = self._preprocess(image)

        with torch.no_grad():
            pred = self.model(tensor)           # [1,1,h,w]
            prob = torch.sigmoid(pred)

            score = prob.mean().item()
            heatmap_small = prob.squeeze().cpu().numpy()   # [h,w]

            # Upsample to original resolution
            heatmap = cv2.resize(
                heatmap_small,
                (image.shape[1], image.shape[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            )

        return score, heatmap

    def passes_threshold(self, image: np.ndarray) -> bool:
        """Quick realism check against configured threshold."""
        return self.evaluate(image) >= self.threshold

    def get_quality_assessment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive realism report.

        Returns:
            dict with:
              - score (float)
              - passes (bool)
              - heatmap (np.ndarray[H,W])
              - weak_regions (list[tuple[int,int]])
              - min/max patch scores
              - patch_size & threshold used
        """
        score, heatmap = self.evaluate_with_heatmap(image)

        weak_mask = heatmap < self.threshold
        weak_y, weak_x = np.where(weak_mask)
        weak_regions = list(zip(weak_y.tolist(), weak_x.tolist()))

        return {
            "score": round(float(score), 4),
            "passes": score >= self.threshold,
            "heatmap": heatmap,
            "weak_regions": weak_regions,
            "min_patch_score": float(heatmap.min()),
            "max_patch_score": float(heatmap.max()),
            "patch_size": self.patch_size,
            "threshold_used": self.threshold
        }


if __name__ == "__main__":
    print("PatchGANEvaluator loaded.")
    print("Example usage:")
    print("  evaluator = PatchGANEvaluator(config)")
    print("  score = evaluator.evaluate(bgr_image)")
    print("  result = evaluator.get_quality_assessment(bgr_image)")
