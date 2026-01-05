# main.py
"""
Hybrid Deep Learning Invisibility Cloaking/Concealment System – Command Line Interface

Processes images, videos, or folders of images to remove / cloak specified
classes (typically 'person') using either static background replacement or
dynamic background generation.

Supported modes:
  - static   : Replace detected objects with provided background
  - dynamic  : Generate plausible background using inpainting / diffusion
  - auto     : Choose mode based on whether --background is provided

Usage examples:
  python main.py -i input.mp4 -o output.mp4 -b background.jpg --mode static
  python main.py -i person.jpg -o cloaked.jpg --classes person car
  python main.py -i frames/ -o results/ --config custom.yaml
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

# Project imports – adjust according to your folder structure
from config import SystemConfig
from pipelines.static_pipeline import StaticPipeline
from pipelines.dynamic_pipeline import DynamicPipeline
from utils.video_utils import get_video_info, frame_generator, VideoWriter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Learning Invisibility Cloaking System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input image, video, or directory path"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output path (image/video) or directory"
    )
    parser.add_argument(
        "--background", "-b", type=str, default=None,
        help="Background image/video for static mode"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="config/default_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--mode", "-m", choices=["static", "dynamic", "auto"],
        default="auto", help="Processing mode"
    )
    parser.add_argument(
        "--classes", nargs="+", default=["person"],
        help="Object classes to cloak (COCO-style names)"
    )
    parser.add_argument(
        "--save-intermediates", action="store_true",
        help="Save visualization of detections, masks, etc."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for video processing (if supported by pipeline)"
    )

    return parser.parse_args()


def determine_mode(args: argparse.Namespace) -> str:
    """Resolve final processing mode."""
    if args.mode != "auto":
        return args.mode

    if args.background is not None:
        return "static"
    return "dynamic"


def load_background_video(path: str) -> list:
    """Load all frames from background video into memory (for static mode)."""
    return [frame for _, frame in frame_generator(path)]


def process_single_image(
    pipeline: Any,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    mode: str
) -> None:
    """Process one image."""
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Failed to load image: {input_path}")
        return

    start_time = time.perf_counter()

    if mode == "static":
        bg = cv2.imread(args.background)
        if bg is None:
            raise ValueError(f"Failed to load background: {args.background}")
        result = pipeline.process_image(
            img, background=bg, target_classes=args.classes,
            return_intermediates=args.save_intermediates
        )
    else:
        result = pipeline.process_image(
            img, target_classes=args.classes,
            return_intermediates=args.save_intermediates
        )

    processing_time = time.perf_counter() - start_time

    cv2.imwrite(str(output_path), result["output"])

    print(f"  Saved: {output_path}")
    print(f"  Time: {processing_time:.3f}s")
    print(f"  Detections: {len(result.get('detections', []))}")


def process_video(
    pipeline: Any,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    mode: str
) -> None:
    """Process video file frame by frame."""
    info = get_video_info(str(input_path))
    print(f"Video: {info.width}×{info.height} @ {info.fps:.2f} fps")
    print(f"Total frames: {info.frame_count:,}")

    bg_frames = None
    if mode == "static":
        bg_path = Path(args.background)
        if bg_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            print("Loading background video into memory...")
            bg_frames = load_background_video(str(bg_path))
        else:
            bg_img = cv2.imread(str(bg_path))
            if bg_img is None:
                raise ValueError("Failed to load static background image")
            bg_frames = [bg_img]  # repeat same frame

    writer = VideoWriter(
        str(output_path),
        fps=info.fps,
        width=info.width,
        height=info.height,
        codec="mp4v"  # or "avc1" if you have H.264 support
    )

    start_time = time.perf_counter()

    with writer:
        for idx, frame in tqdm(
            frame_generator(str(input_path)),
            total=info.frame_count,
            desc="Cloaking",
            unit="frame"
        ):
            if mode == "static":
                # Cycle through background frames if video background
                bg_frame = bg_frames[idx % len(bg_frames)]
                result = pipeline.process_image(
                    frame, background=bg_frame, target_classes=args.classes
                )
            else:
                result = pipeline.process_image(
                    frame, target_classes=args.classes
                )

            writer.write(result["output"])

    total_time = time.perf_counter() - start_time
    print(f"\nSaved: {output_path}")
    print(f"Total time: {total_time:.1f}s ({total_time/info.frame_count:.3f}s/frame)")


def process_directory(
    pipeline: Any,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    mode: str
) -> None:
    """Process all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {input_dir}")

    for img_path in tqdm(image_files, desc="Processing images"):
        out_name = img_path.stem + "_cloaked" + img_path.suffix
        out_path = output_dir / out_name
        process_single_image(pipeline, img_path, out_path, args, mode)


def main():
    args = parse_args()
    config = SystemConfig.from_yaml(args.config)

    mode = determine_mode(args)
    print(f"Starting cloaking pipeline in {mode.upper()} mode")
    print(f"Target classes: {', '.join(args.classes)}")

    # Initialize the appropriate pipeline
    if mode == "static":
        if not args.background:
            raise ValueError("Static mode requires --background")
        pipeline = StaticPipeline(config)
    else:
        pipeline = DynamicPipeline(config)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        if input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            process_video(pipeline, input_path, output_path, args, mode)
        elif input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            process_single_image(pipeline, input_path, output_path, args, mode)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")

    elif input_path.is_dir():
        if output_path.is_file():
            raise ValueError("--output must be a directory when --input is a directory")
        output_path.mkdir(parents=True, exist_ok=True)
        process_directory(pipeline, input_path, output_path, args, mode)

    else:
        raise ValueError(f"Input not found or unsupported: {input_path}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
