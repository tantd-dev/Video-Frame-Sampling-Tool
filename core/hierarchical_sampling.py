"""
Hierarchical Processing: Two-level frame extraction.

Level 1 (Coarse): Extract frames at a wide interval to get an overview.
Level 2 (Fine): For time ranges where coarse frames show significant change,
    extract additional frames at a finer interval.

MVP Implementation:
    1. Extract coarse frames at the coarse interval.
    2. Compare consecutive coarse frames using perceptual hashing.
    3. Identify "interesting" segments where visual change exceeds a threshold.
    4. For each interesting segment, extract fine-grained frames.

Limitations (MVP):
    - Uses perceptual hashing for change detection (not semantic analysis).
    - "Interesting" is defined purely by visual difference between coarse frames.
    - May miss subtle but semantically important changes.
"""

import os
import tempfile
import shutil
from typing import Callable, Optional

from PIL import Image
import imagehash

from core.ffmpeg_runner import FFmpegRunner
from core.simple_sampling import compute_simple_timestamps
from core.smart_sampling import compute_perceptual_hash, compute_similarity
from utils.filesystem import ensure_directory
from utils.timestamp import seconds_to_timestamp_tag


def run_hierarchical_processing(
    input_path: str,
    output_dir: str,
    duration_seconds: float,
    coarse_interval: float = 30.0,
    fine_interval: float = 5.0,
    scene_threshold: float = 0.30,
    output_width: int = 1280,
    image_format: str = "jpg",
    jpeg_quality: int = 85,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    runner: Optional[FFmpegRunner] = None,
) -> int:
    """
    Run hierarchical two-level processing.

    Args:
        input_path: Path to video file.
        output_dir: Base output directory (will contain coarse/ and fine/ subdirs).
        duration_seconds: Video duration.
        coarse_interval: Seconds between coarse frames.
        fine_interval: Seconds between fine frames in interesting segments.
        scene_threshold: Visual similarity threshold to consider a segment "interesting".
            Lower value = more segments marked as interesting.
        output_width: Output image width.
        image_format: Output image format.
        jpeg_quality: JPEG quality.
        progress_callback: Progress callback (0-100).
        status_callback: Status text callback.
        runner: FFmpegRunner instance.

    Returns:
        Total number of frames exported (coarse + fine).
    """
    coarse_dir = os.path.join(output_dir, "coarse")
    fine_dir = os.path.join(output_dir, "fine")
    ensure_directory(coarse_dir)
    ensure_directory(fine_dir)

    if runner is None:
        runner = FFmpegRunner()

    format_ext = image_format.lower()
    if format_ext == "jpeg":
        format_ext = "jpg"

    # =============================================
    # Level 1: Coarse sampling
    # =============================================
    coarse_timestamps = compute_simple_timestamps(duration_seconds, coarse_interval)

    if status_callback:
        status_callback(f"Hierarchical: extracting {len(coarse_timestamps)} coarse frames...")

    coarse_files = runner.extract_frames_at_times(
        input_path=input_path,
        output_dir=coarse_dir,
        timestamps=coarse_timestamps,
        output_width=output_width,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        progress_callback=lambda p: progress_callback(p * 0.4) if progress_callback else None,
        status_callback=status_callback,
    )

    coarse_count = len(coarse_files)

    if coarse_count < 2:
        if status_callback:
            status_callback("Video too short for hierarchical processing.")
        if progress_callback:
            progress_callback(100)
        return coarse_count

    # =============================================
    # Analyze coarse frames to find interesting segments
    # =============================================
    if status_callback:
        status_callback("Analyzing coarse frames for interesting segments...")

    interesting_segments = []  # list of (start_time, end_time) tuples

    # Compute hashes for all coarse frames
    coarse_hashes = []
    for f in coarse_files:
        try:
            h = compute_perceptual_hash(f)
            coarse_hashes.append(h)
        except Exception:
            coarse_hashes.append(None)

    # Compare consecutive coarse frames
    # A segment between two coarse frames is "interesting" if they are visually different
    # (similarity below 1.0 - scene_threshold)
    similarity_cutoff = 1.0 - scene_threshold  # e.g. threshold=0.30 -> cutoff=0.70

    for i in range(len(coarse_hashes) - 1):
        if coarse_hashes[i] is None or coarse_hashes[i + 1] is None:
            # Can't compare — assume interesting
            interesting_segments.append((coarse_timestamps[i], coarse_timestamps[i + 1]))
            continue

        sim = compute_similarity(coarse_hashes[i], coarse_hashes[i + 1])
        if sim < similarity_cutoff:
            # Significant visual change — this segment is interesting
            interesting_segments.append((coarse_timestamps[i], coarse_timestamps[i + 1]))

    if progress_callback:
        progress_callback(50)

    if status_callback:
        status_callback(
            f"Found {len(interesting_segments)} interesting segments. "
            f"Extracting fine frames..."
        )

    # =============================================
    # Level 2: Fine sampling in interesting segments
    # =============================================
    fine_count = 0

    if interesting_segments:
        # Collect all fine timestamps
        all_fine_timestamps = []
        for seg_start, seg_end in interesting_segments:
            t = seg_start
            while t <= seg_end:
                all_fine_timestamps.append(t)
                t += fine_interval

        # Remove duplicates and sort
        all_fine_timestamps = sorted(set(all_fine_timestamps))

        # Remove timestamps that already exist as coarse frames
        coarse_set = set(coarse_timestamps)
        all_fine_timestamps = [t for t in all_fine_timestamps if t not in coarse_set]

        if all_fine_timestamps:
            if status_callback:
                status_callback(
                    f"Extracting {len(all_fine_timestamps)} fine frames..."
                )

            fine_files = runner.extract_frames_at_times(
                input_path=input_path,
                output_dir=fine_dir,
                timestamps=all_fine_timestamps,
                output_width=output_width,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                progress_callback=lambda p: progress_callback(50 + p * 0.5) if progress_callback else None,
                status_callback=status_callback,
            )
            fine_count = len(fine_files)

    if progress_callback:
        progress_callback(100)

    return coarse_count + fine_count


def detect_hierarchical_timestamps(
    input_path: str,
    duration_seconds: float,
    coarse_interval: float = 30.0,
    fine_interval: float = 5.0,
    scene_threshold: float = 0.30,
    runner: Optional[FFmpegRunner] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[float]:
    """
    Generate candidate timestamps using hierarchical two-level analysis.

    Extracts temporary thumbnails of coarse frames to find interesting segments,
    then adds fine-interval timestamps in those segments.
    Returns sorted unique candidate timestamps.
    """
    if runner is None:
        runner = FFmpegRunner()

    coarse_timestamps = compute_simple_timestamps(duration_seconds, coarse_interval)
    if len(coarse_timestamps) < 2:
        return coarse_timestamps

    if status_callback:
        status_callback(f"Hierarchical candidate detection: analyzing {len(coarse_timestamps)} coarse intervals...")

    with tempfile.TemporaryDirectory(prefix="hier_analysis_") as temp_dir:
        # Extract small thumbnails (320px) for fast hashing
        coarse_files = runner.extract_frames_at_times(
            input_path=input_path,
            output_dir=temp_dir,
            timestamps=coarse_timestamps,
            output_width=320,
            image_format="jpg",
            jpeg_quality=70,
            progress_callback=lambda p: progress_callback(p * 0.5) if progress_callback else None,
            status_callback=None,
            filename_prefix="hier_coarse",
        )

        coarse_hashes = []
        for f in coarse_files:
            try:
                coarse_hashes.append(compute_perceptual_hash(f))
            except Exception:
                coarse_hashes.append(None)

        similarity_cutoff = 1.0 - scene_threshold
        interesting_segments = []
        for i in range(len(coarse_hashes) - 1):
            if coarse_hashes[i] is None or coarse_hashes[i + 1] is None:
                interesting_segments.append((coarse_timestamps[i], coarse_timestamps[i + 1]))
                continue
            sim = compute_similarity(coarse_hashes[i], coarse_hashes[i + 1])
            if sim < similarity_cutoff:
                interesting_segments.append((coarse_timestamps[i], coarse_timestamps[i + 1]))

        all_timestamps = list(coarse_timestamps)
        for seg_start, seg_end in interesting_segments:
            t = seg_start
            while t <= seg_end:
                all_timestamps.append(t)
                t += fine_interval

    if progress_callback:
        progress_callback(100)

    return sorted(set(all_timestamps))
