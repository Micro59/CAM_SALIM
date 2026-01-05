# pipelines/static_pipeline.py
"""
Static Invisibility Cloaking Pipeline

A complete pipeline for static background replacement / object removal
using segmentation, inpainting, lighting matching and optional super-resolution.
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, List

# Assuming these are your custom modules
from models.detector import YOLOv8SegDetector
from models.inpainter import LaMaInpainter
from models.enhancer import ESRGANEnhancer
from utils.mask_utils import refine_mask, create_edge_mask
from utils.color_utils import adjust_lighting_lab


class StaticPipeline:
    """
    Static invisibility cloaking pipeline.
    
    Processes images (and potentially video frames) using a pre-recorded
    background to hide selected objects via segmentation + inpainting.
    """

    def __init__(self, config):
        """
        Initialize the static pipeline with all required models.
        
        Args:
            config: Configuration object containing model paths and settings
        """
        self.config = config
        
        print("Loading detector...")
        self.detector = YOLOv8SegDetector(config.detector)
        
        print("Loading inpainter...")
        self.inpainter = LaMaInpainter(config.inpainter)
        
        if config.enhancer.enabled:
            print("Loading enhancer...")
            self.enhancer = ESRGANEnhancer(config.enhancer)
        else:
            self.enhancer = None
            
        print("Static pipeline initialized.")

    def process_image(
        self,
        input_image: np.ndarray,
        background: np.ndarray,
        target_classes: Optional[List[str]] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete static cloaking pipeline.
        
        Pipeline stages:
        1. Object detection & segmentation
        2. Mask refinement
        3. Background lighting/color matching
        4. Simple compositing
        5. Edge-aware inpainting
        6. Optional super-resolution enhancement
        
        Args:
            input_image: BGR input image containing objects to remove
            background: BGR background image to reveal
            target_classes: List of class names to cloak (if None → all detected)
            return_intermediates: If True, returns visualization of each stage
            
        Returns:
            Dictionary containing:
                - 'output': final cloaked image (uint8)
                - 'processing_time': total time in seconds
                - 'detections': list of detection dictionaries
                - 'timing': breakdown of time per stage
                - 'intermediates': (optional) dict of stage images
                - 'message': (optional) status message
        """
        start_time = time.time()
        intermediates = {}

        # Resize background to match input if needed
        if input_image.shape[:2] != background.shape[:2]:
            background = cv2.resize(
                background,
                (input_image.shape[1], input_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # ── Stage 1: Detection and segmentation ───────────────────────────────
        t1 = time.time()
        detections = self.detector.detect(input_image, target_classes)
        detection_time = time.time() - t1

        if not detections:
            return {
                'output': input_image.copy(),
                'processing_time': time.time() - start_time,
                'detections': [],
                'message': 'No objects detected to remove'
            }

        object_mask = self.detector.get_combined_mask(detections)
        if return_intermediates:
            intermediates['raw_mask'] = object_mask.copy()

        # ── Stage 2: Mask refinement ──────────────────────────────────────────
        t2 = time.time()
        refined_mask = refine_mask(
            object_mask,
            kernel_size=5,
            close_iterations=2,
            open_iterations=1
        )
        if return_intermediates:
            intermediates['refined_mask'] = refined_mask.copy()
        refinement_time = time.time() - t2

        # ── Stage 3: Lighting & color matching ────────────────────────────────
        t3 = time.time()
        adjusted_bg = adjust_lighting_lab(
            background, input_image, refined_mask
        )
        if return_intermediates:
            intermediates['adjusted_background'] = adjusted_bg.copy()
        lighting_time = time.time() - t3

        # ── Stage 4: Naive compositing ────────────────────────────────────────
        t4 = time.time()
        mask_3ch = np.stack([refined_mask / 255.0] * 3, axis=-1)
        composite = (adjusted_bg * mask_3ch + input_image * (1 - mask_3ch)).astype(np.uint8)
        if return_intermediates:
            intermediates['composite'] = composite.copy()
        composite_time = time.time() - t4

        # ── Stage 5: Edge inpainting ──────────────────────────────────────────
        t5 = time.time()
        edge_mask = create_edge_mask(refined_mask)
        output = self.inpainter.inpaint(composite, edge_mask)
        if return_intermediates:
            intermediates['edge_mask'] = edge_mask.copy()
            intermediates['inpainted'] = output.copy()
        inpaint_time = time.time() - t5

        # ── Stage 6: Optional super-resolution enhancement ────────────────────
        enhance_time = 0.0
        if self.enhancer is not None:
            t6 = time.time()
            enhanced = self.enhancer.enhance(output)
            # Resize back if enhancer changed resolution
            output = cv2.resize(
                enhanced,
                (input_image.shape[1], input_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
            enhance_time = time.time() - t6

        # ── Final result ──────────────────────────────────────────────────────
        total_time = time.time() - start_time

        result = {
            'output': output,
            'processing_time': round(total_time, 3),
            'detections': detections,
            'timing': {
                'detection':    round(detection_time, 3),
                'refinement':   round(refinement_time, 3),
                'lighting':     round(lighting_time, 3),
                'compositing':  round(composite_time, 3),
                'inpainting':   round(inpaint_time, 3),
                'enhancement':  round(enhance_time, 3)
            }
        }

        if return_intermediates:
            result['intermediates'] = intermediates

        return result
