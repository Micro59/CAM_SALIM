# video_utils.py
"""
Video I/O utilities using OpenCV

Provides:
- Video metadata extraction (VideoInfo dataclass)
- Memory-efficient frame generator for processing large videos
- Context manager for safe & convenient video writing

All functions work with BGR uint8 frames (OpenCV default).
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Container for basic video metadata."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float          # in seconds
    codec: str               # fourcc as 4-char string (e.g. 'mp4v', 'avc1')


def get_video_info(video_path: str) -> VideoInfo:
    """
    Extract essential metadata from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        VideoInfo object with width, height, fps, frame_count, duration, codec

    Raises:
        RuntimeError: If video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Avoid division by zero
    duration = frame_count / fps if fps > 0 else 0.0

    # Get fourcc as readable 4-char string
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))

    cap.release()

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        codec=codec.strip('\x00')   # remove null padding if any
    )


def frame_generator(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Lazy generator that yields (frame_index, frame) tuples one by one.

    Ideal for processing very large videos without loading everything into memory.

    Usage:
        for idx, frame in frame_generator("video.mp4"):
            # process frame (BGR uint8)
            ...

    Yields:
        Tuple[int, np.ndarray]: (0-based frame index, BGR image)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1

    cap.release()


class VideoWriter:
    """
    Context manager for writing video files safely.

    Automatically releases the writer on exit (even on exceptions).

    Supports common codecs like 'mp4v' (MPEG-4), 'XVID', 'avc1' (H.264).

    Example:
        with VideoWriter("output.mp4", fps=30, width=1920, height=1080) as writer:
            for frame in processed_frames:
                writer.write(frame)
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",          # 'mp4v' → .mp4, 'XVID' → .avi, 'avc1' → better H.264
        is_color: bool = True
    ):
        """
        Args:
            output_path: Where to save the video (extension should match codec)
            fps:        Frames per second
            width:      Output frame width
            height:     Output frame height
            codec:      FourCC codec identifier ('mp4v', 'XVID', 'avc1', etc.)
            is_color:   True for color (3 channels), False for grayscale
        """
        self.output_path = output_path
        self.fps = fps
        self.width = int(width)
        self.height = int(height)
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.is_color = is_color
        self.writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        self.writer = cv2.VideoWriter(
            self.output_path,
            self.fourcc,
            self.fps,
            (self.width, self.height),
            isColor=self.is_color
        )
        if not self.writer.isOpened():
            raise RuntimeError(
                f"Cannot open VideoWriter with codec '{self.fourcc}' "
                f"→ check codec compatibility & output path"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.release()
        # Let exceptions propagate (don't swallow them)

    def write(self, frame: np.ndarray) -> bool:
        """
        Write a single frame to the video.

        Args:
            frame: BGR (or grayscale) uint8 image of size (height, width, [3])

        Returns:
            bool: True if frame was written successfully
        """
        if self.writer is None:
            raise RuntimeError("VideoWriter not initialized — use inside 'with' block")

        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame size {frame.shape[:2]} does not match writer "
                f"({self.height}, {self.width})"
            )

        if self.is_color and frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif not self.is_color and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return self.writer.write(frame)


if __name__ == "__main__":
    print("Video utilities loaded.")
    print("Example usage:")
    print("""
info = get_video_info("input.mp4")
print(info)

with VideoWriter("output.mp4", fps=info.fps, width=info.width, height=info.height) as writer:
    for idx, frame in frame_generator("input.mp4"):
        # optional processing here
        writer.write(frame)
        if idx % 100 == 0:
            print(f"Processed frame {idx}")
""")
