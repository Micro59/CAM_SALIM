import cv2
import numpy as np


class StaticVideoProcessor:
    """
    Video processing pipeline for static invisibility cloaking / background replacement.

    Applies per-frame object masking + lighting adjustment (via StaticImageProcessor),
    followed by optional temporal smoothing to reduce flickering.
    """

    def __init__(self, config: dict):
        """
        Initialize the video processor with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - 'image_processor': StaticImageProcessor instance (required)
            - 'temporal_window': int, smoothing window size (default: 5)
            - 'blend_weight': float, blending factor for temporal smoothing (default: 0.3)
            - 'batch_size': int, processing batch size hint (default: 8)
        """
        # Expect StaticImageProcessor instance to be passed in config
        self.image_processor = config['image_processor']  # must be provided

        # Temporal smoothing parameters
        self.temporal_smoother = TemporalTransformer(
            window_size=config.get('temporal_window', 5),
            blend_weight=config.get('blend_weight', 0.3)
        )

        self.batch_size = config.get('batch_size', 8)


    def process_video(
        self,
        input_path: str,
        background_path: str,
        output_path: str,
        target_classes: list[str]
    ) -> None:
        """
        Process an input video by cloaking specified objects and blending into a background.

        Supports both static background images and looping background videos.

        Parameters
        ----------
        input_path : str
            Path to the input video file
        background_path : str
            Path to background image (jpg/png) or video file
        output_path : str
            Path where the output cloaked video will be saved
        target_classes : list of str
            Class names (from YOLO model) to cloak / replace with background
        """
        # Open input video
        cap_input = cv2.VideoCapture(input_path)
        if not cap_input.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        # Open background (image or video)
        cap_bg = cv2.VideoCapture(background_path)
        if not cap_bg.isOpened():
            raise ValueError(f"Cannot open background: {background_path}")

        # Get input video properties
        fps = cap_input.get(cv2.CAP_PROP_FPS)
        width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if background is static image (≤1 frame) or video
        bg_frame_count = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
        is_static_bg = (bg_frame_count <= 1)

        if is_static_bg:
            ret_bg, static_bg_frame = cap_bg.read()
            if not ret_bg:
                raise ValueError("Failed to read static background image")
            static_bg_frame = cv2.resize(static_bg_frame, (width, height))

        # Prepare output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'X264' for better compression
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Buffers for temporal smoothing
        processed_buffer = []

        print(f"Processing {total_frames} frames (batch smoothing every {self.temporal_smoother.window_size} frames)...")

        frame_idx = 0
        while True:
            ret_in, frame_in = cap_input.read()
            if not ret_in:
                break

            # Get corresponding background frame
            if is_static_bg:
                frame_bg = static_bg_frame.copy()
            else:
                ret_bg, frame_bg = cap_bg.read()
                if not ret_bg:
                    # Loop background video
                    cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_bg, frame_bg = cap_bg.read()
                if not ret_bg:
                    raise RuntimeError("Background video ended unexpectedly")
                frame_bg = cv2.resize(frame_bg, (width, height))

            # Apply static per-frame cloaking
            processed_frame = self.image_processor.process(
                frame_in, frame_bg, target_classes
            )

            # Collect for temporal smoothing
            processed_buffer.append(processed_frame)

            # When we have enough frames → smooth & write
            if len(processed_buffer) >= self.temporal_smoother.window_size:
                smoothed = self.temporal_smoother.smooth(
                    processed_buffer[-self.temporal_smoother.window_size:]
                )
                writer.write(smoothed)

                # Keep overlap for next window (sliding window)
                processed_buffer = processed_buffer[-(self.temporal_smoother.window_size - 1):]

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

        # Flush remaining frames (no smoothing or simple write)
        for frame in processed_buffer:
            writer.write(frame)

        # Cleanup
        cap_input.release()
        cap_bg.release()
        writer.release()

        print(f"Output video saved to: {output_path}")


# ────────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy config & processor (replace with your actual classes)
    class DummyImageProcessor:
        def process(self, frame, bg, classes):
            return frame  # placeholder

    class TemporalTransformer:
        def __init__(self, window_size=5, blend_weight=0.3):
            self.window_size = window_size
            self.blend_weight = blend_weight

        def smooth(self, frames):
            return frames[-1]  # placeholder

    config = {
        'image_processor': DummyImageProcessor(),
        'temporal_window': 5,
        'blend_weight': 0.3,
        'batch_size': 8
    }

    processor = StaticVideoProcessor(config)

    # Example call (uncomment with real paths)
    # processor.process_video(
    #     input_path="input.mp4",
    #     background_path="background.jpg",  # or .mp4
    #     output_path="cloaked_output.mp4",
    #     target_classes=["person"]
    # )
