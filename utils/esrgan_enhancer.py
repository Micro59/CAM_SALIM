# esrgan_enhancer.py
"""
ESRGAN Image Super-Resolution & Enhancement Module

Provides high-quality upscaling and detail enhancement using the ESRGAN model
(RRDBNet architecture). Supports tiled processing for large images to avoid OOM.

Typical usage:
    enhancer = ESRGANEnhancer(config)
    enhanced = enhancer.enhance(bgr_image, outscale=4.0)
"""

import cv2
import numpy as np
import torch
from typing import Optional

# Replace these with your actual project structure
from .base import BaseEnhancer
from .rrdbnet_arch import RRDBNet     # ← Make sure this import works


class ESRGANEnhancer(BaseEnhancer):
    """
    ESRGAN-based super-resolution and enhancement.

    Handles:
    - Arbitrary output scaling (via final resize)
    - Tiled inference for very large images
    - Proper RGB/BGR conversion
    - Model loading with EMA / regular params support
    """

    def __init__(self, config):
        """
        Args:
            config: expected attributes
                - weights (str): path to ESRGAN .pth file
                - scale (int/float): model trained scale (usually 4)
                - tile_size (int): side length for tiled processing (e.g. 400)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = config.scale
        self.tile_size = config.tile_size

        # Load model once at initialization
        self.model = self._load_model(config.weights)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, weights_path: str) -> RRDBNet:
        """Load pretrained RRDBNet (ESRGAN architecture)."""
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale
        )

        state_dict = torch.load(weights_path, map_location=self.device)

        # Handle common checkpoint formats
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=True)
        return model

    def enhance(
        self,
        image: np.ndarray,
        outscale: Optional[float] = None
    ) -> np.ndarray:
        """
        Enhance / upscale input image using ESRGAN.

        Args:
            image:      Input image (H, W, 3) in BGR, uint8
            outscale:   Desired final scale factor (default = model scale)

        Returns:
            Enhanced image (H*outscale, W*outscale, 3) BGR uint8
        """
        if outscale is None:
            outscale = self.scale

        h, w = image.shape[:2]

        # Decide whether to use tiled or full inference
        if max(h, w) * outscale > 2000 or max(h, w) > 1500:
            return self._enhance_tiled(image, outscale)

        # Full-image inference (faster for medium-sized images)
        img_tensor = self._preprocess(image)

        with torch.no_grad():
            output = self.model(img_tensor)

        result = self._postprocess(output)

        # Final resize if requested scale ≠ model scale
        if abs(outscale - self.scale) > 1e-6:
            out_h = int(round(h * outscale))
            out_w = int(round(w * outscale))
            result = cv2.resize(
                result,
                (out_w, out_h),
                interpolation=cv2.INTER_LANCZOS4
            )

        return result

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """BGR uint8 → RGB float32 [0,1] NCHW tensor on device."""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Tensor [0,1] NCHW → BGR uint8 numpy array."""
        output = tensor.squeeze(0).permute(1, 2, 0)
        output = output.clamp_(0, 1)
        output = (output * 255).byte().cpu().numpy()
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def _enhance_tiled(self, image: np.ndarray, outscale: float) -> np.ndarray:
        """
        Tiled inference + blending to handle very large images without OOM.
        Uses simple overlapping average blending.
        """
        h, w = image.shape[:2]
        ts = self.tile_size
        overlap = ts // 4
        stride = ts - overlap

        out_h = int(round(h * outscale))
        out_w = int(round(w * outscale))

        # Accumulators
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weights = np.zeros((out_h, out_w), dtype=np.float32)

        y = 0
        while y < h:
            x = 0
            while x < w:
                # Input tile region
                y_end = min(y + ts, h)
                x_end = min(x + ts, w)
                y_start = max(0, y_end - ts)
                x_start = max(0, x_end - ts)

                tile = image[y_start:y_end, x_start:x_end]

                # Inference
                tile_tensor = self._preprocess(tile)
                with torch.no_grad():
                    tile_out = self.model(tile_tensor)
                tile_result = self._postprocess(tile_out)

                # Output coordinates (model-scale)
                oy_start = int(round(y_start * outscale))
                ox_start = int(round(x_start * outscale))
                th, tw = tile_result.shape[:2]

                # Accumulate
                output[oy_start:oy_start+th, ox_start:ox_start+tw] += tile_result.astype(np.float32)
                weights[oy_start:oy_start+th, ox_start:ox_start+tw] += 1.0

                x += stride
            y += stride

        # Normalize blended result
        weights = np.maximum(weights, 1e-6)[..., None]
        output = (output / weights).astype(np.uint8)

        # Final resize if needed
        if abs(outscale - self.scale) > 1e-6:
            output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        return output


if __name__ == "__main__":
    print("ESRGANEnhancer loaded.")
    print("Example:")
    print("  enhancer = ESRGANEnhancer(config)")
    print("  result = enhancer.enhance(cv2.imread('input.jpg'), outscale=4.0)")
    print("  cv2.imwrite('enhanced.jpg', result)")
