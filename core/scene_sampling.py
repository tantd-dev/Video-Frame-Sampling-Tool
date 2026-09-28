"""
Scene-change Sampling: Extract frames where significant scene changes occur.

Uses FFmpeg's scene detection filter to identify frames with visual changes
above a configurable threshold.

This is scene-change based sampling, NOT semantic keyframe detection.
"""

import os
import re
import subprocess
from typing import Callable, Optional

from core.ffmpeg_runner import FFmpegRunner, FFmpegNotFoundError
from utils.filesystem import ensure_directory
from utils.timestamp import seconds_to_timestamp_tag


def detect_scene_changes(
    input_path: str,
    threshold: float = 0.30,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[float]:
    """
    Detect scene changes in a video using FFmpeg's scene filter.

    Args:
        input_path: Path to video file.
        threshold: Scene change threshold (0.0 to 1.0). Lower = more sensitive.
        status_callback: Called with status text.

    Returns:
        List of timestamps (in seconds) where scene changes occur.
    """
    if status_callback:
        status_callback("Detecting scene changes...")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for long videos
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
    except FileNotFoundError:
        raise FFmpegNotFoundError(
            "FFmpeg was not found. Please install FFmpeg and add it to PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Scene detection timed out. Video may be too long.")

    # Parse timestamps from FFmpeg showinfo output (in stderr)
    timestamps = []
    # showinfo outputs lines like: [Parsed_showinfo_1 ...] n:   0 pts:  12345 pts_time:42.5 ...
    pattern = re.compile(r"pts_time:\s*([0-9]+\.?[0-9]*)")

    for line in result.stderr.split('\n'):
        if "pts_time" in line:
            match = pattern.search(line)
            if match:
                ts = float(match.group(1))
                timestamps.append(ts)

    return sorted(set(timestamps))


def run_scene_sampling(
    input_path: str,
    output_dir: str,
    threshold: float = 0.30,
    output_width: int = 1280,
    image_format: str = "jpg",
    jpeg_quality: int = 85,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    runner: Optional[FFmpegRunner] = None,
) -> int:
    """
    Run scene-change based sampling.

    1. Detect scene changes using FFmpeg's scene filter.
    2. Extract frames at detected timestamps.

    Args:
        input_path: Path to video file.
        output_dir: Directory to save frames.
        threshold: Scene change sensitivity (0.0-1.0).
        output_width: Output image width.
        image_format: Output image format.
        jpeg_quality: JPEG quality.
        progress_callback: Progress callback (0-100).
        status_callback: Status text callback.
        runner: FFmpegRunner instance.

    Returns:
        Number of frames exported.
    """
    ensure_directory(output_dir)

    if runner is None:
        runner = FFmpegRunner()

    # Step 1: Detect scene changes
    if status_callback:
        status_callback("Analyzing video for scene changes...")

    timestamps = detect_scene_changes(input_path, threshold, status_callback)

    if not timestamps:
        if status_callback:
            status_callback("No scene changes detected. Try lowering the threshold.")
        return 0

    if status_callback:
        status_callback(f"Found {len(timestamps)} scene changes. Extracting frames...")

    # Step 2: Extract frames at scene-change timestamps
    output_files = runner.extract_frames_at_times(
        input_path=input_path,
        output_dir=output_dir,
        timestamps=timestamps,
        output_width=output_width,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        progress_callback=progress_callback,
        status_callback=status_callback,
    )

    return len(output_files)
