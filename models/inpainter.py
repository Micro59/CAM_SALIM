# models/inpainter.py
"""
LaMa Inpainting Module

High-quality generative inpainting using the LaMa (Large Mask Inpainting) model.
Supports:
- Full-image inference for small/medium images
- Tiled processing with overlap blending for large images
- Raised-cosine feathering to reduce seam artifacts
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional

from .base import BaseInpainter


class LaMaInpainter(BaseInpainter):
    """
    LaMa-based inpainting implementation with tiled support for high-resolution images.
    
    Handles arbitrary mask shapes and sizes efficiently while preserving
    boundary consistency through overlapping tiles and smooth blending.
    """

    def __init__(self, config):
        """
        Initialize the LaMa inpainter.

        Args:
            config: Configuration object with:
                - weights: path to LaMa model weights (.pth)
                - resolution: internal processing resolution (e.g. 512)
                - tile_size: side length of processing tiles (e.g. 512 or 768)
                - tile_overlap: overlap between tiles in pixels (e.g. 64–128)
        """
        super().__init__(config)
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading LaMa model...")
        self.model = self._load_model(config.weights)
        
        self.resolution = config.resolution
        self.tile_size = config.tile_size
        self.tile_overlap = config.tile_overlap
        
        print(f"LaMa inpainter ready (device={self.device}, tile={self.tile_size}px, "
              f"overlap={self.tile_overlap}px)")

    def _load_model(self, weights_path: str):
        """Load pretrained LaMa model weights."""
        from .lama_arch import LaMaModel
        model = LaMaModel()
        state_dict = torch.load(weights_path, map_location=self.device)
        # Handle possible wrapped state dict
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert numpy image & mask → model-ready tensors (BGR → RGB not needed if trained on BGR).
        
        Returns:
            (img_tensor, mask_tensor): both (1,C,H,W) float32 [0,1] on device
        """
        img = torch.from_numpy(image).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)          # → (1,3,H,W)

        msk = torch.from_numpy(mask).float() / 255.0
        msk = msk.unsqueeze(0).unsqueeze(0)               # → (1,1,H,W)

        return img.to(self.device), msk.to(self.device)

    def _postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Convert model output tensor back to uint8 numpy image."""
        output = output.squeeze(0).permute(1, 2, 0)       # → (H,W,3)
        output = output.clamp_(0, 1)
        output = (output * 255).byte()
        return output.cpu().numpy()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Main inpainting entry point.
        
        Automatically chooses between direct or tiled inference based on image size.
        
        Args:
            image: BGR input image (H,W,3) uint8
            mask: Binary mask (H,W) uint8 [0 or 255] – regions to inpaint
            
        Returns:
            Inpainted image (H,W,3) uint8
        """
        h, w = image.shape[:2]

        if max(h, w) > self.tile_size:
            return self._inpaint_tiled(image, mask)

        # ── Direct inference path (small/medium images) ──────────────────────
        # Resize to model working resolution
        img_resized = cv2.resize(image, (self.resolution, self.resolution))
        mask_resized = cv2.resize(
            mask,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_NEAREST
        )

        img_t, mask_t = self._preprocess(img_resized, mask_resized)

        with torch.no_grad():
            # Prepare input: masked image + mask channel
            masked_img = img_t * (1 - mask_t) + mask_t * 0.5   # mild gray fill
            input_t = torch.cat([masked_img, mask_t], dim=1)   # → (1,4,H,W)

            pred = self.model(input_t)

            # Blend: keep original outside mask, use prediction inside
            result = pred * mask_t + img_t * (1 - mask_t)

        result_np = self._postprocess(result)

        # Return to original resolution
        return cv2.resize(result_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

    def _inpaint_tiled(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Tiled inpainting with overlap blending for large/high-resolution images.
        Uses raised-cosine window to feather seams.
        """
        h, w = image.shape[:2]
        ts = self.tile_size
        ov = self.tile_overlap
        stride = ts - ov

        # Accumulators
        output = np.zeros((h, w, 3), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        # Precompute blending kernel (raised cosine)
        blend_kernel = self._create_blend_weights(ts)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Tile bounds (handle edges)
                y1, y2 = max(0, y), min(y + ts, h)
                x1, x2 = max(0, x), min(x + ts, w)
                tile_y1, tile_x1 = y1 - y, x1 - x

                tile_img = image[y1:y2, x1:x2]
                tile_mask = mask[y1:y2, x1:x2]

                # Skip tile if nothing to inpaint
                if tile_mask.max() == 0:
                    output[y1:y2, x1:x2] += tile_img.astype(np.float32)
                    weights[y1:y2, x1:x2] += 1.0
                    continue

                # Inpaint this tile
                tile_result = self._inpaint_single_tile(tile_img, tile_mask)

                # Apply feathering weights
                th, tw = tile_result.shape[:2]
                tile_w = blend_kernel[:th, :tw]

                output[y1:y2, x1:x2] += tile_result.astype(np.float32) * tile_w[..., None]
                weights[y1:y2, x1:x2] += tile_w

        # Normalize
        weights = np.maximum(weights, 1e-6)
        output = (output / weights[..., None]).clip(0, 255)

        return output.astype(np.uint8)

    def _inpaint_single_tile(self, tile_img: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
        """Helper: inpaint one tile at model resolution."""
        img_resized = cv2.resize(tile_img, (self.resolution, self.resolution))
        mask_resized = cv2.resize(tile_mask, (self.resolution, self.resolution),
                                 interpolation=cv2.INTER_NEAREST)

        img_t, mask_t = self._preprocess(img_resized, mask_resized)

        with torch.no_grad():
            masked = img_t * (1 - mask_t) + mask_t * 0.5
            input_t = torch.cat([masked, mask_t], dim=1)
            pred = self.model(input_t)
            result = pred * mask_t + img_t * (1 - mask_t)

        result_np = self._postprocess(result)
        return cv2.resize(result_np, tile_img.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)

    def _create_blend_weights(self, size: int) -> np.ndarray:
        """Generate 2D raised-cosine blending kernel for tile overlap."""
        ramp_len = self.tile_overlap
        ramp = np.linspace(0, 1, ramp_len)
        ramp = 0.5 * (1 - np.cos(np.pi * ramp))  # raised cosine

        weights_1d = np.ones(size, dtype=np.float32)
        weights_1d[:ramp_len] = ramp
        weights_1d[-ramp_len:] = ramp[::-1]

        return np.outer(weights_1d, weights_1d)
