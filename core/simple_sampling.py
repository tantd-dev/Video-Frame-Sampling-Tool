"""
Simple Sampling: Extract frames at fixed time intervals.
"""

from typing import Callable, Optional
from core.ffmpeg_runner import FFmpegRunner
from utils.filesystem import ensure_directory


def compute_simple_timestamps(duration_seconds: float, interval_seconds: float) -> list[float]:
    """
    Compute timestamps for simple interval-based sampling.

    Args:
        duration_seconds: Total video duration in seconds.
        interval_seconds: Interval between frames in seconds.

    Returns:
        List of timestamps in seconds.

    Example:
        duration=907, interval=10 -> [0, 10, 20, ..., 900]
    """
    if interval_seconds <= 0:
        raise ValueError("Sampling interval must be greater than 0.")
    if duration_seconds <= 0:
        raise ValueError("Video duration must be greater than 0.")

    timestamps = []
    t = 0.0
    while t <= duration_seconds:
        timestamps.append(t)
        t += interval_seconds

    return timestamps


def run_simple_sampling(
    input_path: str,
    output_dir: str,
    duration_seconds: float,
    interval_seconds: float,
    output_width: int = 1280,
    image_format: str = "jpg",
    jpeg_quality: int = 85,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    runner: Optional[FFmpegRunner] = None,
) -> int:
    """
    Run simple sampling: extract one frame every N seconds.

    Args:
        input_path: Path to video file.
        output_dir: Directory to save frames.
        duration_seconds: Video duration in seconds.
        interval_seconds: Seconds between frames.
        output_width: Output image width (height auto-calculated).
        image_format: Output image format (jpg, png, etc.).
        jpeg_quality: JPEG quality (1-100).
        progress_callback: Called with progress 0-100.
        status_callback: Called with status text.
        runner: FFmpegRunner instance (created if None).

    Returns:
        Number of frames exported.
    """
    ensure_directory(output_dir)

    if runner is None:
        runner = FFmpegRunner()

    if status_callback:
        status_callback("Computing simple sampling timestamps...")

    timestamps = compute_simple_timestamps(duration_seconds, interval_seconds)

    if status_callback:
        status_callback(f"Extracting {len(timestamps)} frames (simple sampling)...")

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
