# temporal_transformer.py
"""
Temporal Transformer for Video Frame Temporal Smoothing / Consistency

Uses patch-based temporal attention across a sliding window of frames
to produce a temporally coherent version of the center frame.

Main features:
- Patch embedding + temporal positional encoding
- Transformer encoder over temporal + spatial tokens
- Blended output (model prediction + original frame)
- Full-sequence smoothing with proper boundary padding

Typical usage:
    model = TemporalTransformer(window_size=5, blend_weight=0.4)
    smoothed_frames = model.smooth_sequence(list_of_bgr_frames)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class TemporalTransformer(nn.Module):
    """
    Patch-based Temporal Transformer for enforcing temporal consistency in video.

    Architecture:
      1. Patch embedding (Conv2d) per frame
      2. Flatten patches → tokens
      3. Add learnable temporal positional encoding
      4. Transformer encoder over (T × N) tokens
      5. Extract center-frame tokens
      6. Reconstruct center frame via transposed convolution
      7. Optional blending with original frame
    """

    def __init__(
        self,
        window_size: int = 5,
        blend_weight: float = 0.4,          # how much of model output to keep
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            window_size:    Number of frames in each temporal window (odd recommended)
            blend_weight:   Weight of model output in final blend (0.0 = original, 1.0 = pure model)
            patch_size:     Spatial patch size (must divide H and W)
            embed_dim:      Token dimension
            num_heads:      Number of attention heads
            num_layers:     Number of transformer encoder layers
            dropout:        Dropout rate in transformer
        """
        super().__init__()

        assert window_size % 2 == 1, "window_size should be odd for symmetric context"

        self.window_size = window_size
        self.blend_weight = blend_weight
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding (same for every frame)
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable temporal positional encoding (one per frame in window)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, window_size, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Reconstruct image from center-frame patches
        self.decoder = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=3,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Optional small conv after decoder for refinement (can be disabled)
        self.postprocess = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(3, affine=True)
        )

    def extract_patches(self, frame: torch.Tensor) -> torch.Tensor:
        """(C,H,W) → (1, N_patches, embed_dim)"""
        # (1, embed_dim, H/p, W/p)
        embedded = self.patch_embed(frame.unsqueeze(0))
        # (1, N, embed_dim)
        return embedded.flatten(2).transpose(1, 2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (T, C, H, W) tensor in range [0,1]

        Returns:
            Smoothed center frame: (C, H, W) in [0,1]
        """
        T, C, H, W = frames.shape
        assert T == self.window_size, f"Expected {self.window_size} frames, got {T}"

        # Patchify all frames
        patches = [self.extract_patches(frames[t]) for t in range(T)]
        patches = torch.cat(patches, dim=1)     # (1, T×N, D)

        # Add temporal positional encoding
        temporal_pos = self.temporal_pos_embed.repeat_interleave(
            patches.shape[1] // T, dim=1
        )
        patches = patches + temporal_pos

        # Run through transformer
        transformed = self.transformer(patches)     # (1, T×N, D)

        # Extract only center frame tokens
        center_idx = T // 2
        N = patches.shape[1] // T
        center_tokens = transformed[:, center_idx * N : (center_idx + 1) * N, :]

        # Reconstruct spatial grid
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        center_tokens = center_tokens.transpose(1, 2).view(1, self.embed_dim, h_p, w_p)

        # Decode to RGB
        reconstructed = self.decoder(center_tokens)     # (1, 3, H, W)
        reconstructed = torch.tanh(reconstructed)       # bound to [-1,1]
        reconstructed = (reconstructed + 1) / 2         # → [0,1]

        # Optional light refinement
        reconstructed = self.postprocess(reconstructed)

        return reconstructed.squeeze(0)

    @torch.no_grad()
    def smooth_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Smooth an entire video sequence using sliding window.

        Args:
            frames: List of BGR uint8 images (same size)

        Returns:
            List of smoothed BGR uint8 images
        """
        if not frames:
            return []

        device = next(self.parameters()).device
        pad = self.window_size // 2

        # Mirror-pad sequence at boundaries
        padded = (
            [frames[0]] * pad +
            frames +
            [frames[-1]] * pad
        )

        smoothed = []

        for i in range(len(frames)):
            window = padded[i : i + self.window_size]

            # Convert to tensor [T, C, H, W] ∈ [0,1]
            window_tensor = torch.stack([
                torch.from_numpy(f).float().permute(2, 0, 1) / 255.0
                for f in window
            ]).to(device)

            smoothed_tensor = self(window_tensor)           # (C, H, W) [0,1]

            # Blend with original
            original = torch.from_numpy(frames[i]).float()
            original = original.permute(2, 0, 1).to(device) / 255.0

            blended = self.blend_weight * smoothed_tensor + \
                      (1 - self.blend_weight) * original

            # Back to uint8 BGR
            result = (blended.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            smoothed.append(result)

        return smoothed


if __name__ == "__main__":
    print("TemporalTransformer loaded.")
    print("Minimal example:")
    print("""
model = TemporalTransformer(window_size=5, blend_weight=0.35).cuda().eval()

# frames = [cv2.imread(f) for f in video_paths]
# smoothed = model.smooth_sequence(frames)
""")
