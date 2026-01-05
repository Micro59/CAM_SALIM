import cv2
import numpy as np


class DynamicVideoProcessor:
    """
    Advanced dynamic video processing pipeline with temporal coherence for high-quality
    object removal / background replacement (e.g. invisibility cloaking).

    Pipeline stages per frame:
      1. YOLOv8-Seg detection & masking
      2. Shadow detection
      3. Mask unification & morphological refinement
      4. Generative inpainting (LaMa)
      5. Super-resolution enhancement (ESRGAN)
      6. Realism assessment (PatchGAN) + optional iterative refinement
      → Temporal smoothing across frames using transformer-based coherence

    Designed for better temporal stability and visual realism compared to static pipelines.
    """

    def __init__(self, config: dict):
        """
        Initialize all components from configuration.

        Parameters
        ----------
        config : dict
            Must contain keys:
            - 'yolo_weights'         : str
            - 'deepshadow_weights'   : str
            - 'scotch_weights'       : str
            - 'lama_weights'         : str
            - 'esrgan_weights'       : str
            - 'patchgan_weights'     : str
            Optional:
            - 'conf_threshold'       : float (default 0.25)
            - 'temporal_window'      : int    (default 5)
            - 'patch_size'           : int    (default 16)
            - 'num_heads'            : int    (default 8)
            - 'blend_weight'         : float  (default 0.4)
            - 'realism_threshold'    : float  (default 0.9)
            - 'max_iterations'       : int    (default 3)
        """
        from your_module import (           # Replace with actual imports
            YOLOv8SegWrapper,
            ShadowDetector,
            LaMaInpainter,
            ESRGANEnhancer,
            PatchGANAssessor,
            TemporalTransformer
        )

        self.detector = YOLOv8SegWrapper(
            weights=config['yolo_weights'],
            conf_threshold=config.get('conf_threshold', 0.25)
        )

        self.shadow_detector = ShadowDetector(
            deepshadow_weights=config['deepshadow_weights'],
            scotch_weights=config['scotch_weights']
        )

        self.inpainter = LaMaInpainter(config['lama_weights'])
        self.enhancer = ESRGANEnhancer(config['esrgan_weights'])
        self.assessor = PatchGANAssessor(config['patchgan_weights'])

        self.temporal_transformer = TemporalTransformer(
            window_size=config.get('temporal_window', 5),
            patch_size=config.get('patch_size', 16),
            num_heads=config.get('num_heads', 8),
            blend_weight=config.get('blend_weight', 0.4)
        )

        self.realism_threshold = config.get('realism_threshold', 0.90)
        self.max_refine_iterations = config.get('max_iterations', 3)


    def process_frame(self, frame: np.ndarray, target_classes: list[str]) -> tuple[np.ndarray, float]:
        """
        Process a single frame through the full dynamic pipeline.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame (H×W×3)
        target_classes : list of str
            Classes to detect/remove (e.g. ["person"])

        Returns
        -------
        tuple (enhanced_frame, realism_score)
            - enhanced_frame : np.ndarray (BGR)
            - realism_score  : float ∈ [0,1]
        """
        # ─── Stage 1: Detection & Segmentation ───────────────────────────────
        detections = self.detector.detect_and_segment(frame, target_classes=target_classes)

        if not detections:
            return frame, 1.0  # No objects → original frame is perfect

        object_mask = self.detector.get_combined_mask(detections)

        # ─── Stage 2: Shadow Detection ───────────────────────────────────────
        shadow_mask = self.shadow_detector.detect(frame, object_mask)

        # ─── Stage 3: Unified & Refined Mask ─────────────────────────────────
        unified_mask = np.maximum(object_mask, shadow_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_mask = cv2.morphologyEx(unified_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)

        # ─── Stage 4: Inpainting ─────────────────────────────────────────────
        inpainted = self.inpainter.inpaint(frame, refined_mask)

        # ─── Stage 5: Super-Resolution Enhancement ───────────────────────────
        enhanced = self.enhancer.enhance(inpainted)
        # Resize back to original resolution (ESRGAN often upsamples ×4)
        enhanced = cv2.resize(enhanced, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # ─── Stage 6: Realism Assessment + Iterative Refinement ──────────────
        realism_score = self.assessor.evaluate(enhanced)
        iteration = 0

        while realism_score < self.realism_threshold and iteration < self.max_refine_iterations:
            # Slightly expand mask for next attempt
            dilated_mask = cv2.dilate(refined_mask, kernel, iterations=2)
            inpainted = self.inpainter.inpaint(enhanced, dilated_mask)
            enhanced = self.enhancer.enhance(inpainted)
            enhanced = cv2.resize(enhanced, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            realism_score = self.assessor.evaluate(enhanced)
            iteration += 1

        return enhanced, realism_score


    def process_video(self, input_path: str, output_path: str, target_classes: list[str]) -> None:
        """
        Process entire video with per-frame dynamic pipeline + temporal smoothing.

        Parameters
        ----------
        input_path : str
            Path to input video
        output_path : str
            Where to save the processed video
        target_classes : list of str
            Classes to remove/cloak
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Consider 'X264' for better quality/size
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {total_frames} frames @ {fps:.1f} fps")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # ─── Process all frames ──────────────────────────────────────────────
        processed_frames = []
        for i, frame in enumerate(frames):
            enhanced, score = self.process_frame(frame, target_classes)
            processed_frames.append(enhanced)
            print(f"  Frame {i+1}/{total_frames} — realism: {score:.4f}")

        # ─── Apply temporal coherence / smoothing ────────────────────────────
        print("Applying temporal transformer smoothing...")
        smoothed_frames = self.temporal_transformer.smooth_sequence(processed_frames)

        # ─── Write output ────────────────────────────────────────────────────
        for frame in smoothed_frames:
            writer.write(frame)

        writer.release()
        print(f"Output saved to: {output_path}")


# ────────────────────────────────────────────────────────────────
# Example / test block
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy config for demonstration (replace with real paths & models)
    dummy_config = {
        'yolo_weights': 'yolov8m-seg.pt',
        'deepshadow_weights': 'deepshadow.pth',
        'scotch_weights': 'scotch.pth',
        'lama_weights': 'lama.pth',
        'esrgan_weights': 'esrgan_x4.pth',
        'patchgan_weights': 'patchgan_discriminator.pth',
        'conf_threshold': 0.25,
        'temporal_window': 5,
        'patch_size': 16,
        'num_heads': 8,
        'blend_weight': 0.4,
        'realism_threshold': 0.90,
        'max_iterations': 3
    }

    # processor = DynamicVideoProcessor(dummy_config)
    # processor.process_video(
    #     input_path="input.mp4",
    #     output_path="cloaked_dynamic.mp4",
    #     target_classes=["person", "car"]
    # )
