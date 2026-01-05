# pipelines/dynamic_pipeline.py
"""
Dynamic Invisibility Cloaking/Concealment Pipeline

Processes images or videos without requiring a pre-recorded background.
Uses generative inpainting (LaMa), shadow-aware masking, optional temporal smoothing,
and quality assessment via PatchGAN discriminator.

Main components:
- YOLOv8 segmentation → object removal mask
- Combined shadow detection → shadow inclusion in removal mask
- LaMa inpainting → background synthesis
- Optional ESRGAN upscaling
- Optional PatchGAN realism scoring
- Optional temporal consistency for video
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, List, Dict, Any, Callable

from models.detector import YOLOv8SegDetector
from models.shadow_detector import CombinedShadowDetector
from models.inpainter import LaMaInpainter
from models.enhancer import ESRGANEnhancer
from models.discriminator import PatchGANEvaluator
from utils.mask_utils import refine_mask, unify_masks
from utils.color_utils import adjust_lighting_lab


class DynamicPipeline:
    """
    Dynamic pipeline for object removal and realistic background synthesis
    without needing a static background image.
    """

    def __init__(self, config):
        """
        Initialize all components of the dynamic cloaking pipeline.

        Args:
            config: Configuration object (expected to have .detector, .shadow,
                    .inpainter, .enhancer, .discriminator, .temporal attributes)
        """
        self.config = config
        self.logger = logging.getLogger('DynamicPipeline')

        self.logger.info("Loading detector...")
        self.detector = YOLOv8SegDetector(config.detector)

        self.logger.info("Loading shadow detector...")
        self.shadow_detector = CombinedShadowDetector(config.shadow)

        self.logger.info("Loading inpainter...")
        self.inpainter = LaMaInpainter(config.inpainter)

        self.enhancer = None
        if hasattr(config, 'enhancer') and config.enhancer.enabled:
            self.logger.info("Loading enhancer...")
            self.enhancer = ESRGANEnhancer(config.enhancer)

        self.discriminator = None
        if hasattr(config, 'discriminator') and config.discriminator.enabled:
            self.logger.info("Loading discriminator...")
            self.discriminator = PatchGANEvaluator(config.discriminator)

        self.logger.info("Dynamic pipeline initialized.")

    def process_image(
        self,
        input_image: np.ndarray,
        target_classes: Optional[List[str]] = None,
        return_intermediates: bool = False,
        max_refinement_passes: int = 1
    ) -> Dict[str, Any]:
        """
        Process a single image through the full dynamic cloaking pipeline.

        Args:
            input_image: BGR input image (numpy array)
            target_classes: Optional list of class names to remove
            return_intermediates: If True, include all stage outputs in result
            max_refinement_passes: Number of iterative inpainting refinements

        Returns:
            Dict with:
                - 'output': final result image
                - 'processing_time': total time (seconds)
                - 'detections': list of detection dicts
                - 'realism_score': PatchGAN realism score (if enabled)
                - 'timing': per-stage timing breakdown
                - 'intermediates': (optional) dict of stage images
        """
        start_time = time.time()
        intermediates = {}
        timing = {}

        # ── 1. Object Detection & Segmentation ───────────────────────────────
        t = time.time()
        detections = self.detector.detect(input_image, target_classes)
        timing['detection'] = time.time() - t

        if not detections:
            return {
                'output': input_image.copy(),
                'processing_time': time.time() - start_time,
                'detections': [],
                'realism_score': 1.0,
                'message': 'No objects detected',
                'timing': timing
            }

        object_mask = self.detector.get_combined_mask(detections)

        if return_intermediates:
            intermediates['object_mask'] = object_mask.copy()
            intermediates['detections'] = detections

        # ── 2. Shadow Detection ──────────────────────────────────────────────
        t = time.time()
        shadow_mask = self.shadow_detector.detect(input_image, object_mask)
        timing['shadow_detection'] = time.time() - t

        if return_intermediates:
            intermediates['shadow_mask'] = shadow_mask.copy()

        # ── 3. Mask Unification (object + shadow) ────────────────────────────
        t = time.time()
        unified_mask = unify_masks(
            object_mask,
            shadow_mask,
            method=self.config.shadow.get('fusion', 'union')
        )
        timing['mask_unification'] = time.time() - t

        if return_intermediates:
            intermediates['unified_mask'] = unified_mask.copy()

        # ── 4. Mask Refinement ───────────────────────────────────────────────
        t = time.time()
        refined_mask = refine_mask(
            unified_mask,
            kernel_size=5,
            close_iterations=2,
            open_iterations=1,
            dilation=self.config.shadow.get('dilation', 0)
        )
        timing['mask_refinement'] = time.time() - t

        if return_intermediates:
            intermediates['refined_mask'] = refined_mask.copy()

        # ── 5. Generative Inpainting ─────────────────────────────────────────
        t = time.time()
        inpainted = self.inpainter.inpaint(input_image, refined_mask)

        # Optional multi-pass refinement (edge-focused)
        for _ in range(1, max_refinement_passes):
            if self.config.inpainter.get('refinement_passes', 1) > _:
                edge_mask = cv2.erode(refined_mask, np.ones((5, 5), np.uint8), iterations=2)
                inpainted = self.inpainter.inpaint(inpainted, edge_mask)

        timing['inpainting'] = time.time() - t

        if return_intermediates:
            intermediates['inpainted'] = inpainted.copy()

        # ── 6. Optional Super-Resolution Enhancement ─────────────────────────
        t = time.time()
        if self.enhancer is not None:
            enhanced = self.enhancer.enhance(inpainted)
            output = cv2.resize(
                enhanced,
                (input_image.shape[1], input_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
        else:
            output = inpainted
        timing['enhancement'] = time.time() - t

        if return_intermediates:
            intermediates['enhanced'] = output.copy()

        # ── 7. Optional Realism Assessment ───────────────────────────────────
        t = time.time()
        realism_score = None
        if self.discriminator is not None:
            quality = self.discriminator.get_quality_assessment(output)
            realism_score = quality['score']
            if return_intermediates:
                intermediates['quality_heatmap'] = quality.get('heatmap')
        timing['quality_assessment'] = time.time() - t

        total_time = time.time() - start_time

        result = {
            'output': output,
            'processing_time': round(total_time, 3),
            'detections': detections,
            'realism_score': round(realism_score, 4) if realism_score is not None else None,
            'timing': {k: round(v, 3) for k, v in timing.items()}
        }

        if return_intermediates:
            result['intermediates'] = intermediates

        self.logger.info(
            f"Image processed in {total_time:.2f}s | "
            f"realism: {realism_score:.4f if realism_score else 'N/A'} | "
            f"objects: {len(detections)}"
        )

        return result

    # ── Video & Batch Processing ─────────────────────────────────────────────
    # (Implementation for process_video and process_batch remains similar to your original)
    # You can keep them as-is or let me know if you want them polished further.
