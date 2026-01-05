# models/discriminator.py
"""
PatchGAN Discriminator (70×70 receptive field)

Classic PatchGAN architecture as used in pix2pix / CycleGAN / LaMa-style pipelines.
Outputs an N×N feature map where each element classifies whether the corresponding
70×70 patch of the input image is real or fake.

Used here for:
- Local realism assessment of inpainted regions
- Optional quality scoring / guidance in the dynamic cloaking pipeline
"""

import torch
import torch.nn as nn
from typing import Optional


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator producing a spatial grid of real/fake predictions.
    
    Architecture follows the standard 70×70 receptive field design:
    - 4 downsampling + 1 final convolution
    - LeakyReLU activations
    - Instance Normalization after the first layer
    - Normal weight initialization (mean=0, std=0.02)
    """

    def __init__(
        self,
        input_channels: int = 3,
        ndf: int = 64,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        use_sigmoid: bool = False
    ):
        """
        Args:
            input_channels: Number of input channels (3 for RGB, 4 if mask concatenated)
            ndf: Number of discriminator filters in the first layer (64 is standard)
            norm_layer: Normalization layer (InstanceNorm2d is typical for PatchGAN)
            use_sigmoid: Whether to apply sigmoid at the end (usually False for LSGAN)
        """
        super().__init__()
        self.ndf = ndf

        # ── Layer sequence ───────────────────────────────────────────────────
        # Conv1: no norm, stride 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Conv2–4: with normalization, stride 2 until Conv4
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final 1×1 prediction map (no activation here)
        self.output = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1, bias=False)

        # Optional final sigmoid (usually off when using LSGAN loss)
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Custom weight initialization (normal dist mean=0, std=0.02)."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('InstanceNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Patch-wise predictions (B, 1, H', W') – typically H'/W' ≈ H/16
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)

        if self.use_sigmoid:
            x = self.sigmoid(x)

        return x

    def get_receptive_field(self) -> int:
        """
        Effective receptive field size of this PatchGAN architecture.
        (Fixed at 70×70 for this 5-layer design with kernel=4, stride=2×3 + stride=1×2)
        """
        return 70


class PatchGANEvaluator:
    """
    Wrapper around PatchGANDiscriminator for inference-time quality assessment.
    Computes average patch realism score and optional per-patch heatmap.
    """

    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PatchGANDiscriminator(
            input_channels=config.get('input_channels', 3),
            ndf=config.get('ndf', 64)
        ).to(self.device)
        self.model.eval()

        # Load pretrained weights if provided
        if hasattr(config, 'weights') and config.weights:
            state_dict = torch.load(config.weights, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded PatchGAN weights from {config.weights}")

    @torch.no_grad()
    def get_quality_assessment(
        self,
        image: torch.Tensor,
        return_heatmap: bool = True
    ) -> dict:
        """
        Evaluate realism of an image (or inpainted output).

        Args:
            image: (B,3,H,W) tensor in [-1,1] or [0,1] range (will be normalized)
            return_heatmap: Whether to return spatial prediction map

        Returns:
            dict with:
                - 'score': Average patch realism (higher = more realistic)
                - 'heatmap': (optional) upsampled prediction map
        """
        # Normalize to [-1,1] if needed
        if image.max() <= 1.0:
            image = image * 2.0 - 1.0

        image = image.to(self.device)

        pred = self.model(image)           # (B,1,H',W')
        score = pred.mean().item()         # scalar realism proxy

        result = {'score': float(score)}

        if return_heatmap:
            # Upsample to input resolution for visualization
            heatmap = torch.sigmoid(pred) * 255.0
            heatmap = nn.functional.interpolate(
                heatmap,
                size=image.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            result['heatmap'] = heatmap.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        return result
