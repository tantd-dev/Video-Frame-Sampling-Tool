"""
Heuristic Smart Sampling: Extract candidate frames then remove visually redundant ones.

This is a HEURISTIC-based smart sampling approach. It does NOT use any LLM, VLM,
or semantic understanding. It uses perceptual hashing to measure visual similarity
between consecutive candidate frames and removes highly similar (redundant) frames.

Pipeline:
    Video → Candidate sampling → Extract candidate frames → Compare visual similarity
    → Remove highly redundant frames → Export remaining frames
"""

import os
import tempfile
from typing import Callable, Optional

from PIL import Image
import imagehash

from core.ffmpeg_runner import FFmpegRunner
from core.simple_sampling import compute_simple_timestamps
from utils.filesystem import ensure_directory
from utils.timestamp import seconds_to_timestamp_tag


def compute_perceptual_hash(image_path: str) -> imagehash.ImageHash:
    """Compute the perceptual hash of an image file."""
    img = Image.open(image_path)
    return imagehash.phash(img)


def compute_similarity(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
    """
    Compute similarity between two perceptual hashes.

    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    # Hamming distance: 0 = identical
    # phash produces 64-bit hashes, so max distance is 64
    distance = hash1 - hash2
    similarity = 1.0 - (distance / 64.0)
    return max(0.0, min(1.0, similarity))


def run_smart_sampling(
    input_path: str,
    output_dir: str,
    duration_seconds: float,
    candidate_interval: float = 10.0,
    similarity_threshold: float = 0.90,
    output_width: int = 1280,
    image_format: str = "jpg",
    jpeg_quality: int = 85,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    runner: Optional[FFmpegRunner] = None,
) -> int:
    """
    Run heuristic smart sampling.

    1. Generate candidate timestamps at regular intervals.
    2. Extract candidate frames to a temp directory.
    3. Compute perceptual hash for each candidate.
    4. Compare consecutive frames — keep only those with similarity below threshold.
    5. Copy kept frames to the output directory.

    Args:
        input_path: Path to video file.
        output_dir: Directory to save final frames.
        duration_seconds: Video duration.
        candidate_interval: Seconds between candidate frames.
        similarity_threshold: Frames more similar than this are dropped (0.0-1.0).
        output_width: Output image width.
        image_format: Output image format.
        jpeg_quality: JPEG quality.
        progress_callback: Progress callback (0-100).
        status_callback: Status text callback.
        runner: FFmpegRunner instance.

    Returns:
        Number of frames exported after redundancy removal.
    """
    ensure_directory(output_dir)

    if runner is None:
        runner = FFmpegRunner()

    # Step 1: Generate candidate timestamps
    timestamps = compute_simple_timestamps(duration_seconds, candidate_interval)
    total_candidates = len(timestamps)

    if status_callback:
        status_callback(f"Smart sampling: extracting {total_candidates} candidates...")

    # Step 2: Extract candidates to temp directory
    with tempfile.TemporaryDirectory(prefix="smart_sampling_") as temp_dir:
        # Extract candidates
        format_ext = image_format.lower()
        if format_ext == "jpeg":
            format_ext = "jpg"

        candidate_files = runner.extract_frames_at_times(
            input_path=input_path,
            output_dir=temp_dir,
            timestamps=timestamps,
            output_width=output_width,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            progress_callback=lambda p: progress_callback(p * 0.6) if progress_callback else None,
            status_callback=status_callback,
            filename_prefix="candidate",
        )

        if not candidate_files:
            if status_callback:
                status_callback("No candidate frames extracted.")
            return 0

        # Step 3: Compute hashes and filter
        if status_callback:
            status_callback("Comparing frames for visual similarity...")

        kept_indices = []  # indices into timestamps/candidate_files
        prev_hash = None

        for i, cfile in enumerate(candidate_files):
            if runner.is_cancelled:
                from core.ffmpeg_runner import FFmpegError
                raise FFmpegError("Operation cancelled by user.")

            try:
                current_hash = compute_perceptual_hash(cfile)
            except Exception:
                # If we can't hash a frame, keep it
                kept_indices.append(i)
                continue

            if prev_hash is None:
                # Always keep the first frame
                kept_indices.append(i)
                prev_hash = current_hash
            else:
                similarity = compute_similarity(prev_hash, current_hash)
                if similarity < similarity_threshold:
                    # Frame is sufficiently different — keep it
                    kept_indices.append(i)
                    prev_hash = current_hash
                # else: frame is too similar to previous kept frame — skip it

            if progress_callback:
                progress_callback(60 + ((i + 1) / len(candidate_files)) * 30)

        if status_callback:
            status_callback(
                f"Smart sampling: kept {len(kept_indices)} of {total_candidates} candidates."
            )

        # Step 4: Copy kept frames to output directory with proper naming
        import shutil

        output_files = []
        for out_idx, kept_idx in enumerate(kept_indices):
            if runner.is_cancelled:
                from core.ffmpeg_runner import FFmpegError
                raise FFmpegError("Operation cancelled by user.")

            src = candidate_files[kept_idx]
            ts = timestamps[kept_idx]
            ts_tag = seconds_to_timestamp_tag(ts)
            dst_name = f"frame_{out_idx + 1:04d}_{ts_tag}.{format_ext}"
            dst = os.path.join(output_dir, dst_name)
            shutil.copy2(src, dst)
            output_files.append(dst)

        if progress_callback:
            progress_callback(100)

    return len(output_files)
