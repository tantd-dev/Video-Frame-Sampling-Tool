"""
Core package for Video Frame Sampling Tool.
"""

from core.hybrid_pipeline import (
    CandidateFrame,
    generate_strategy_candidates,
    merge_candidates,
    deduplicate_by_timestamp,
    filter_visual_similarity,
    preserve_temporal_coverage,
    prefer_scene_changes,
    apply_max_frame_count,
    run_hybrid_pipeline,
)
