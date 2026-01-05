# models/shadow_detector.py
"""
Shadow Detection Module

Combines predictions from two deep learning models (DeepShadow and Scotch & Soda)
to detect shadows cast by objects, given an input image and object segmentation mask.

Supports configurable fusion strategy (union / intersection) and post-processing (dilation).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from .base import BaseShadowDetector


class CombinedShadowDetector(BaseShadowDetector):
    """
    Combined shadow detector using DeepShadow (SAM-ViT based) and Scotch & Soda models.
    
    This detector takes an RGB image and a binary object mask, and returns a shadow mask
    highlighting regions likely to contain object shadows.
    """

    def __init__(self, config):
        """
        Initialize the combined shadow detector with both models.
        
        Args:
            config: Configuration object containing:
                - deepshadow_weights: path to DeepShadow model weights
                - scotch_weights: path to Scotch & Soda model weights
                - threshold: binarization threshold (0–1)
                - dilation: kernel radius for dilation post-processing (0 = disabled)
                - fusion: 'union', 'intersection', or None (default: DeepShadow only)
        """
        super().__init__(config)
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading DeepShadow model...")
        self.deepshadow = self._load_deepshadow(config.deepshadow_weights)
        
        print("Loading Scotch & Soda model...")
        self.scotch = self._load_scotch(config.scotch_weights)
        
        self.threshold = config.threshold
        self.dilation = config.dilation
        self.fusion = config.fusion.lower() if config.fusion else 'deepshadow'
        
        print(f"Shadow detector initialized (fusion={self.fusion}, threshold={self.threshold}, "
              f"dilation={self.dilation}, device={self.device})")

    def _load_deepshadow(self, weights_path: str):
        """Load DeepShadow model from weights."""
        from .deepshadow_arch import DeepShadowNet
        model = DeepShadowNet()
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_scotch(self, weights_path: str):
        """Load Scotch & Soda model from weights."""
        from .scotch_arch import ScotchSodaNet
        model = ScotchSodaNet()
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image: np.ndarray, object_mask: np.ndarray) -> torch.Tensor:
        """
        Prepare input tensor: RGB image + object mask channel.
        
        Args:
            image: RGB image (H,W,3) uint8 [0–255]
            object_mask: binary mask (H,W) uint8 [0 or 255]
            
        Returns:
            torch.Tensor: (1, 4, H, W) float32 tensor on device
        """
        # Normalize RGB to [0,1]
        img_tensor = torch.from_numpy(image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)   # → (1,3,H,W)

        # Normalize mask to [0,1]
        mask_tensor = torch.from_numpy(object_mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)     # → (1,1,H,W)

        # Concatenate → (1,4,H,W)
        input_tensor = torch.cat([img_tensor, mask_tensor], dim=1)
        
        return input_tensor.to(self.device)

    def detect(
        self,
        image: np.ndarray,
        object_mask: np.ndarray,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Detect shadows associated with the masked objects.
        
        Args:
            image: RGB input image (H,W,3) uint8
            object_mask: Binary mask of objects to consider (H,W) uint8 [0/255]
            return_probs: If True, return soft probability map instead of binary mask
            
        Returns:
            np.ndarray: Shadow mask (H,W) uint8 [0–255]
        """
        h, w = image.shape[:2]

        input_tensor = self._preprocess(image, object_mask)

        with torch.no_grad():
            # ── DeepShadow ───────────────────────────────────────
            deep_logits = self.deepshadow(input_tensor)
            deep_prob = torch.sigmoid(deep_logits).squeeze()     # (H,W)

            # ── Scotch & Soda ────────────────────────────────────
            scotch_logits = self.scotch(input_tensor)
            scotch_prob = torch.sigmoid(scotch_logits).squeeze() # (H,W)

        # Move to CPU & numpy
        deep_prob_np  = deep_prob.cpu().numpy()
        scotch_prob_np = scotch_prob.cpu().numpy()

        # Resize if models changed resolution (rare but possible)
        if deep_prob_np.shape != (h, w):
            deep_prob_np = cv2.resize(deep_prob_np, (w, h), interpolation=cv2.INTER_LINEAR)
        if scotch_prob_np.shape != (h, w):
            scotch_prob_np = cv2.resize(scotch_prob_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Fusion strategy ──────────────────────────────────
        if self.fusion == 'union':
            combined_prob = np.maximum(deep_prob_np, scotch_prob_np)
        elif self.fusion == 'intersection':
            combined_prob = np.minimum(deep_prob_np, scotch_prob_np)
        elif self.fusion == 'deepshadow':
            combined_prob = deep_prob_np
        elif self.fusion == 'scotch':
            combined_prob = scotch_prob_np
        else:
            # fallback
            combined_prob = deep_prob_np

        if return_probs:
            # Return soft probabilities [0–255]
            return (combined_prob * 255).astype(np.uint8)

        # Binarize
        binary = (combined_prob > self.threshold).astype(np.float32)

        # Convert to uint8 mask
        shadow_mask = (binary * 255).astype(np.uint8)

        # ── Optional dilation ────────────────────────────────
        if self.dilation > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation * 2 + 1, self.dilation * 2 + 1)
            )
            shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)

        return shadow_mask
