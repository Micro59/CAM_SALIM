"""
config/config.py

Central configuration management using dataclasses + YAML loading.
Supports easy overriding, type safety, and defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class DetectorConfig:
    model: str = "yolov8m-seg"
    weights: str = "weights/yolov8m-seg.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_size: int = 640
    target_classes: List[str] = field(default_factory=lambda: ["person"])


@dataclass
class ShadowConfig:
    deepshadow_weights: str = "weights/deepshadow.pt"
    scotch_weights: str = "weights/scotch_soda.pt"
    threshold: float = 0.5
    dilation: int = 3
    fusion: str = "union"  # union | intersection | deepshadow_only


@dataclass
class InpainterConfig:
    model: str = "lama"
    weights: str = "weights/big-lama.pt"
    resolution: int = 512
    tile_size: int = 512
    tile_overlap: int = 64
    refinement_passes: int = 1


@dataclass
class EnhancerConfig:
    model: str = "esrgan"
    weights: str = "weights/esrgan_x4.pt"
    scale: int = 4
    tile_size: int = 256


@dataclass
class DiscriminatorConfig:
    weights: str = "weights/patchgan.pt"
    patch_size: int = 70
    realism_threshold: float = 0.90


@dataclass
class TemporalConfig:
    window_size: int = 5
    blend_weight: float = 0.4
    patch_size: int = 16
    num_heads: int = 8


@dataclass
class OutputConfig:
    format: str = "mp4"
    quality: int = 95
    save_intermediates: bool = False


@dataclass
class SystemConfig:
    device: str = "cuda"
    precision: str = "fp16"          # fp16 | fp32 | bf16
    num_workers: int = 4

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    inpainter: InpainterConfig = field(default_factory=InpainterConfig)
    enhancer: EnhancerConfig = field(default_factory=EnhancerConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'SystemConfig':
        """
        Load configuration from YAML file with graceful fallback to defaults.
        
        Args:
            path: Path to config YAML file
            
        Returns:
            Fully populated SystemConfig instance
        """
        path = Path(path)
        if not path.exists():
            print(f"Config file not found: {path}. Using defaults.")
            return cls()

        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # Nested sections
        system_data = data.get('system', {})
        detector_data = data.get('detector', {})
        shadow_data = data.get('shadow', {})
        inpainter_data = data.get('inpainter', {})
        enhancer_data = data.get('enhancer', {})
        discriminator_data = data.get('discriminator', {})
        temporal_data = data.get('temporal', {})
        output_data = data.get('output', {})

        return cls(
            device=system_data.get('device', cls.device),
            precision=system_data.get('precision', cls.precision),
            num_workers=system_data.get('num_workers', cls.num_workers),

            detector=DetectorConfig(**detector_data),
            shadow=ShadowConfig(**shadow_data),
            inpainter=InpainterConfig(**inpainter_data),
            enhancer=EnhancerConfig(**enhancer_data),
            discriminator=DiscriminatorConfig(**discriminator_data),
            temporal=TemporalConfig(**temporal_data),
            output=OutputConfig(**output_data)
        )


# Convenience global access (optional)
def load_config(config_path: str = "config/default_config.yaml") -> SystemConfig:
    """Load the default or custom configuration."""
    return SystemConfig.from_yaml(config_path)


if __name__ == "__main__":
    # Quick validation / debug
    cfg = load_config()
    print(f"Device: {cfg.device}")
    print(f"Target classes: {cfg.detector.target_classes}")
    print(f"Realism threshold: {cfg.discriminator.realism_threshold}")
