"""
Hybrid Frame Selection Pipeline.

Merges candidate frames from multiple sampling strategies into a single pool,
then applies a multi-stage selection pipeline to produce a final, high-quality
selection of frames:

1. Generate candidate frames from enabled sampling strategies.
2. Merge all candidates into a single candidate pool.
3. Deduplicate candidates with nearly identical timestamps.
4. Perform visual similarity filtering to remove highly redundant frames.
5. Preserve temporal coverage across the entire video.
6. Prefer frames with strong scene-change / visual-difference signals.
7. Apply a configurable maximum final frame count.
8. Copy/export only the selected final frames into: frames/
"""

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image
import imagehash

from core.ffmpeg_runner import FFmpegRunner, FFmpegError
from core.simple_sampling import compute_simple_timestamps
from core.scene_sampling import detect_scene_changes
from core.hierarchical_sampling import detect_hierarchical_timestamps
from core.smart_sampling import compute_perceptual_hash, compute_similarity
from utils.filesystem import ensure_directory
from utils.timestamp import format_frame_filename, seconds_to_timestamp_tag


@dataclass
class CandidateFrame:
    """A candidate frame with metadata about its source and quality signals."""

    timestamp: float               # seconds into the video
    source: str                    # which strategy produced it (e.g. "simple", "scene")
    is_scene_change: bool = False  # True if detected via scene-change analysis
    score: float = 0.0             # higher = more valuable to keep
    temp_path: str = ""            # path to extracted temp image
    phash: Optional[object] = None # perceptual hash for visual comparison


def generate_strategy_candidates(
    strategy: str,
    config: dict,
    video_path: str,
    duration_seconds: float,
    runner: Optional[FFmpegRunner] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[float]:
    """
    Generate candidate timestamps for a single enabled sampling strategy.
    """
    if runner is None:
        runner = FFmpegRunner()

    if strategy == "simple":
        interval = config.get("interval", 10)
        return compute_simple_timestamps(duration_seconds, interval)

    elif strategy == "scene":
        threshold = config.get("threshold", 0.30)
        return detect_scene_changes(video_path, threshold, status_callback)

    elif strategy == "smart":
        candidate_interval = config.get("candidate_interval", 10)
        return compute_simple_timestamps(duration_seconds, candidate_interval)

    elif strategy == "hierarchical":
        coarse_interval = config.get("coarse_interval", 30)
        fine_interval = config.get("fine_interval", 5)
        scene_threshold = config.get("scene_threshold", 0.30)
        return detect_hierarchical_timestamps(
            input_path=video_path,
            duration_seconds=duration_seconds,
            coarse_interval=coarse_interval,
            fine_interval=fine_interval,
            scene_threshold=scene_threshold,
            runner=runner,
            status_callback=status_callback,
            progress_callback=progress_callback,
        )

    return []


def merge_candidates(candidate_lists: dict[str, list[float]]) -> list[CandidateFrame]:
    """
    Step 2: Merge all candidates from enabled strategies into a single candidate pool.

    Args:
        candidate_lists: Dict mapping strategy name -> list of timestamps.

    Returns:
        Sorted list of CandidateFrame objects.
    """
    pool = []
    for source, timestamps in candidate_lists.items():
        is_scene = (source == "scene")
        for ts in timestamps:
            pool.append(CandidateFrame(
                timestamp=round(float(ts), 3),
                source=source,
                is_scene_change=is_scene,
            ))

    pool.sort(key=lambda c: c.timestamp)
    return pool


def deduplicate_by_timestamp(
    candidates: list[CandidateFrame],
    min_gap_seconds: float = 0.5,
) -> list[CandidateFrame]:
    """
    Step 3: Deduplicate candidates with nearly identical timestamps.

    When candidates are within min_gap_seconds of each other:
    - Retain scene-change signals if any candidate in the cluster had it.
    - Boost score based on how many distinct strategies suggested this point.
    - Pick the best representative timestamp.
    """
    if not candidates:
        return []

    clusters: list[list[CandidateFrame]] = []
    current_cluster = [candidates[0]]

    for c in candidates[1:]:
        if c.timestamp - current_cluster[-1].timestamp <= min_gap_seconds:
            current_cluster.append(c)
        else:
            clusters.append(current_cluster)
            current_cluster = [c]
    clusters.append(current_cluster)

    result = []
    for cluster in clusters:
        # Prefer scene-change frame
        scene_frames = [c for c in cluster if c.is_scene_change]
        if scene_frames:
            best = scene_frames[0]
        else:
            best = cluster[0]

        # Count multi-strategy agreement
        sources = set(c.source for c in cluster)
        best.score += (len(sources) - 1) * 1.5  # boost when multiple strategies agree
        if any(c.is_scene_change for c in cluster):
            best.is_scene_change = True

        result.append(best)

    return result


def extract_candidates_to_temp(
    input_path: str,
    candidates: list[CandidateFrame],
    temp_dir: str,
    output_width: int,
    image_format: str,
    jpeg_quality: int,
    runner: FFmpegRunner,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[CandidateFrame]:
    """
    Extract candidate frames into a temporary directory at the target output width.
    These extracted files serve both visual similarity analysis and final export.
    """
    if not candidates:
        return []

    if status_callback:
        status_callback(f"Extracting {len(candidates)} candidate frames to temp...")

    timestamps = [c.timestamp for c in candidates]
    temp_files = runner.extract_frames_at_times(
        input_path=input_path,
        output_dir=temp_dir,
        timestamps=timestamps,
        output_width=output_width,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        progress_callback=progress_callback,
        status_callback=None,
        filename_prefix="candidate",
    )

    # Map temp file paths back to candidates
    valid_candidates = []
    for i, c in enumerate(candidates):
        if i < len(temp_files) and os.path.isfile(temp_files[i]):
            c.temp_path = temp_files[i]
            valid_candidates.append(c)

    return valid_candidates


def compute_hashes(
    candidates: list[CandidateFrame],
    runner: FFmpegRunner,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[CandidateFrame]:
    """Compute perceptual hashes for all candidate frames."""
    if status_callback:
        status_callback("Analyzing visual content (perceptual hashing)...")

    for c in candidates:
        if runner.is_cancelled:
            raise FFmpegError("Operation cancelled by user.")
        if c.temp_path and os.path.isfile(c.temp_path):
            try:
                c.phash = compute_perceptual_hash(c.temp_path)
            except Exception:
                c.phash = None
        else:
            c.phash = None

    return candidates


def filter_visual_similarity(
    candidates: list[CandidateFrame],
    similarity_threshold: float = 0.90,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[CandidateFrame]:
    """
    Step 4: Perform visual similarity filtering to remove highly redundant frames.
    """
    if status_callback:
        status_callback(f"Filtering visual similarity (threshold={similarity_threshold})...")

    if not candidates:
        return []

    kept = [candidates[0]]

    for c in candidates[1:]:
        if c.phash is None or kept[-1].phash is None:
            kept.append(c)
            continue

        sim = compute_similarity(kept[-1].phash, c.phash)
        if sim < similarity_threshold:
            # Different enough from the previous kept frame
            kept.append(c)
        else:
            # Too similar to previous kept frame; if it is a scene change, check if it's
            # distinct from recent frames
            if c.is_scene_change:
                distinct_recent = all(
                    compute_similarity(k.phash, c.phash) < similarity_threshold
                    for k in kept[-3:]
                    if k.phash is not None
                )
                if distinct_recent:
                    kept.append(c)

    return kept


def preserve_temporal_coverage(
    candidates: list[CandidateFrame],
    duration_seconds: float,
    target_segments: int = 10,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[CandidateFrame]:
    """
    Step 5: Preserve temporal coverage across the entire video.

    Divides the video duration into equal temporal bins and boosts scores for
    frames that represent sparsely populated bins, ensuring the final selection
    doesn't leave large chronological gaps.
    """
    if not candidates or duration_seconds <= 0:
        return candidates

    if status_callback:
        status_callback("Preserving temporal coverage across video...")

    num_bins = max(3, min(target_segments, len(candidates)))
    bin_size = duration_seconds / num_bins

    # Count frames per bin
    bin_counts = [0] * num_bins
    for c in candidates:
        b = min(int(c.timestamp / bin_size), num_bins - 1)
        bin_counts[b] += 1

    # Boost frames in under-represented bins
    for c in candidates:
        b = min(int(c.timestamp / bin_size), num_bins - 1)
        if bin_counts[b] <= 1:
            c.score += 4.0  # sole representative of time block
        elif bin_counts[b] <= 2:
            c.score += 2.0

    return candidates


def prefer_scene_changes(
    candidates: list[CandidateFrame],
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[CandidateFrame]:
    """
    Step 6: Prefer frames with strong scene-change / visual-difference signals.
    """
    if status_callback:
        status_callback("Prioritizing scene-change & visual-difference signals...")

    for c in candidates:
        if c.is_scene_change:
            c.score += 3.5

    return candidates


def apply_max_frame_count(
    candidates: list[CandidateFrame],
    max_frames: int,
    duration_seconds: float = 0,
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[CandidateFrame]:
    """
    Step 7: Apply a configurable maximum final frame count.

    If candidate count exceeds max_frames, progressively prunes the lowest-value
    frames while protecting endpoints and preventing large temporal gaps.
    """
    if max_frames <= 0 or len(candidates) <= max_frames:
        return candidates

    if status_callback:
        status_callback(f"Limiting to max {max_frames} frames (from {len(candidates)})...")

    # Greedy removal of lowest damage
    while len(candidates) > max_frames:
        min_damage = float('inf')
        min_idx = -1

        for i in range(1, len(candidates) - 1):
            damage = candidates[i].score

            # Gap penalty: penalize removal if it creates an excessive temporal gap
            if duration_seconds > 0:
                new_gap = candidates[i + 1].timestamp - candidates[i - 1].timestamp
                expected_gap = duration_seconds / max_frames
                if new_gap > expected_gap * 2.0:
                    damage += 3.0

            if damage < min_damage:
                min_damage = damage
                min_idx = i

        if min_idx >= 0:
            candidates.pop(min_idx)
        else:
            candidates.pop(-2)

    return candidates


def copy_final_frames_to_output(
    candidates: list[CandidateFrame],
    frames_dir: str,
    image_format: str = "jpg",
    status_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Step 8: Copy/export only the selected final frames into frames/
    """
    ensure_directory(frames_dir)

    format_ext = image_format.lower()
    if format_ext == "jpeg":
        format_ext = "jpg"

    if status_callback:
        status_callback(f"Copying {len(candidates)} final frames into frames/...")

    exported_paths = []
    for idx, c in enumerate(candidates):
        if not c.temp_path or not os.path.isfile(c.temp_path):
            continue

        filename = format_frame_filename(idx + 1, c.timestamp, format_ext)
        dest_path = os.path.join(frames_dir, filename)
        shutil.copy2(c.temp_path, dest_path)
        exported_paths.append(dest_path)

    return exported_paths


def run_hybrid_pipeline(
    input_path: str,
    export_base_dir: str,
    candidate_lists: dict[str, list[float]],
    duration_seconds: float,
    output_width: int = 1280,
    image_format: str = "jpg",
    jpeg_quality: int = 85,
    max_frames: int = 0,
    similarity_threshold: float = 0.90,
    dedup_gap_seconds: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    runner: Optional[FFmpegRunner] = None,
) -> tuple[list[str], list[CandidateFrame], dict]:
    """
    Execute the entire hybrid frame selection pipeline (Steps 1 through 8).

    Args:
        input_path: Path to video file.
        export_base_dir: Base directory for the export (will contain 'frames/' and 'metadata.json').
        candidate_lists: Dict of strategy_name -> timestamps list (from Step 1).
        duration_seconds: Total duration of video in seconds.
        output_width: Output frame width.
        image_format: Output image format.
        jpeg_quality: JPEG quality.
        max_frames: Maximum number of final frames (0 = unlimited).
        similarity_threshold: Perceptual similarity threshold for filtering.
        dedup_gap_seconds: Minimum gap between timestamps to consider distinct.
        progress_callback: Callback receiving progress (0-100).
        status_callback: Callback receiving status text.
        runner: FFmpegRunner instance.

    Returns:
        (exported_file_paths, final_candidates, pipeline_stats)
    """
    start_time = time.time()

    if runner is None:
        runner = FFmpegRunner()

    total_candidates_in = sum(len(ts) for ts in candidate_lists.values())
    if status_callback:
        status_callback(f"Hybrid pipeline started: {total_candidates_in} initial candidates")

    # Step 2: Merge candidates into a single pool
    if status_callback:
        status_callback("Step 2/8: Merging candidate timestamps into single pool...")
    candidates = merge_candidates(candidate_lists)
    initial_merged_count = len(candidates)

    if not candidates:
        return [], [], {
            "initial_candidates": 0,
            "final_frames": 0,
            "elapsed_seconds": time.time() - start_time,
        }

    # Step 3: Deduplicate near-identical timestamps
    if status_callback:
        status_callback("Step 3/8: Deduplicating near-identical timestamps...")
    candidates = deduplicate_by_timestamp(candidates, min_gap_seconds=dedup_gap_seconds)
    deduped_count = len(candidates)

    if progress_callback:
        progress_callback(10)

    # Step 4, 5, 6, 7, 8 in temp directory
    with tempfile.TemporaryDirectory(prefix="hybrid_pool_") as temp_dir:
        # Extract candidate frames to temp
        candidates = extract_candidates_to_temp(
            input_path=input_path,
            candidates=candidates,
            temp_dir=temp_dir,
            output_width=output_width,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            runner=runner,
            progress_callback=lambda p: progress_callback(10 + p * 0.45) if progress_callback else None,
            status_callback=status_callback,
        )

        if runner.is_cancelled:
            raise FFmpegError("Operation cancelled by user.")

        if progress_callback:
            progress_callback(55)

        # Compute perceptual hashes
        candidates = compute_hashes(candidates, runner, status_callback)

        if progress_callback:
            progress_callback(65)

        # Step 4: Visual similarity filtering
        if status_callback:
            status_callback("Step 4/8: Filtering redundant frames via perceptual similarity...")
        candidates = filter_visual_similarity(
            candidates,
            similarity_threshold=similarity_threshold,
            status_callback=status_callback,
        )
        similarity_filtered_count = len(candidates)

        if progress_callback:
            progress_callback(75)

        # Step 5: Preserve temporal coverage across the entire video
        if status_callback:
            status_callback("Step 5/8: Preserving temporal coverage across video...")
        candidates = preserve_temporal_coverage(
            candidates,
            duration_seconds=duration_seconds,
            status_callback=status_callback,
        )

        # Step 6: Prefer frames with strong scene-change signals
        if status_callback:
            status_callback("Step 6/8: Preferring scene-change & visual-difference signals...")
        candidates = prefer_scene_changes(candidates, status_callback=status_callback)

        if progress_callback:
            progress_callback(85)

        # Step 7: Apply configurable maximum final frame count
        if max_frames > 0:
            if status_callback:
                status_callback(f"Step 7/8: Applying maximum final frame count ({max_frames})...")
            candidates = apply_max_frame_count(
                candidates,
                max_frames=max_frames,
                duration_seconds=duration_seconds,
                status_callback=status_callback,
            )

        final_count = len(candidates)

        if progress_callback:
            progress_callback(90)

        # Step 8: Copy/export only the selected final frames into frames/
        frames_dir = os.path.join(export_base_dir, "frames")
        if status_callback:
            status_callback("Step 8/8: Copying selected final frames into frames/...")

        exported_files = copy_final_frames_to_output(
            candidates=candidates,
            frames_dir=frames_dir,
            image_format=image_format,
            status_callback=status_callback,
        )

    elapsed_seconds = round(time.time() - start_time, 2)

    if progress_callback:
        progress_callback(100)

    stats = {
        "initial_candidates": initial_merged_count,
        "strategy_breakdown": {k: len(v) for k, v in candidate_lists.items()},
        "after_dedup": deduped_count,
        "after_similarity_filter": similarity_filtered_count,
        "final_frames": len(exported_files),
        "max_frames_limit": max_frames,
        "frames_directory": os.path.abspath(frames_dir),
        "elapsed_seconds": elapsed_seconds,
    }

    return exported_files, candidates, stats
