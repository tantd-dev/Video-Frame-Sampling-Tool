"""
Batch export worker for multi-video processing.

Processes all video files in a given input folder, running the hybrid
frame sampling pipeline on each and writing summary results to an Excel
workbook.
"""

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from config import (
    DEFAULT_HYBRID_DEDUP_GAP,
    DEFAULT_HYBRID_SIMILARITY_THRESHOLD,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from core.video_info import get_video_metadata
from models.video_metadata import VideoMetadata
from utils.filesystem import sanitize_folder_name


class BatchExportWorker(QThread):
    """Background thread that processes multiple videos sequentially.

    Signals:
        overall_progress(float): 0-100 across all videos.
        video_progress(float): 0-100 for the current video.
        status_updated(str): Human-readable status text.
        video_started(int, str): (index, filename) when a new video begins.
        video_finished(int, str, dict): (index, filename, results) per video.
        video_error(int, str, str): (index, filename, error_msg) per video.
        finished_all(dict): Final aggregate results when all done.
        finished_error(str): Fatal error that stops the whole batch.
    """

    overall_progress = Signal(float)
    video_progress = Signal(float)
    status_updated = Signal(str)
    video_started = Signal(int, str)
    video_finished = Signal(int, str, dict)
    video_error = Signal(int, str, str)
    finished_all = Signal(dict)
    finished_error = Signal(str)

    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        excel_path: str,
        sampling_config: dict,
        output_config: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.excel_path = excel_path
        self.sampling_config = sampling_config
        self.output_config = output_config
        self._cancelled = False
        self._runner = None

    def cancel(self):
        """Request cancellation of the batch."""
        self._cancelled = True
        if self._runner:
            self._runner.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def _discover_videos(self) -> List[str]:
        """Find all supported video files in the input folder."""
        videos = []
        for entry in sorted(os.listdir(self.input_folder)):
            ext = Path(entry).suffix.lstrip('.').lower()
            if ext in SUPPORTED_VIDEO_EXTENSIONS:
                videos.append(os.path.join(self.input_folder, entry))
        return videos

    def run(self):
        """Execute the batch pipeline."""
        from core.ffmpeg_runner import FFmpegRunner, FFmpegError
        from core.hybrid_pipeline import (
            generate_strategy_candidates,
            run_hybrid_pipeline,
        )

        batch_start = time.time()

        # Discover video files
        self.status_updated.emit("Scanning input folder for video files...")
        video_paths = self._discover_videos()

        if not video_paths:
            self.finished_error.emit(
                f"No supported video files found in:\n{self.input_folder}\n\n"
                f"Supported extensions: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
            )
            return

        total_videos = len(video_paths)
        self.status_updated.emit(f"Found {total_videos} video(s). Starting batch processing...")

        # Excel data accumulator
        excel_rows: List[dict] = []
        success_count = 0
        error_count = 0

        for vid_idx, video_path in enumerate(video_paths):
            if self._cancelled:
                self.status_updated.emit("Batch cancelled by user.")
                break

            filename = os.path.basename(video_path)
            base_name = Path(video_path).stem
            safe_name = sanitize_folder_name(base_name)

            self.video_started.emit(vid_idx, filename)
            self.status_updated.emit(
                f"[{vid_idx + 1}/{total_videos}] Processing: {filename}"
            )

            video_start_time = time.time()

            # Create per-video output directory
            video_export_dir = os.path.join(self.output_folder, safe_name)

            try:
                # Load metadata
                metadata = get_video_metadata(video_path)

                # Create runner for this video
                self._runner = FFmpegRunner()

                duration = metadata.duration_seconds
                output_width = self.output_config["width"]
                image_format = self.output_config["format"]
                jpeg_quality = self.output_config["jpeg_quality"]
                max_frames = self.output_config.get("max_frames", 0)
                similarity_threshold = self.output_config.get(
                    "similarity_threshold", DEFAULT_HYBRID_SIMILARITY_THRESHOLD
                )

                # Generate candidates
                candidate_lists = {}
                enabled_strategies = [
                    s for s in ("simple", "scene", "smart", "hierarchical")
                    if self.sampling_config.get(s, {}).get("enabled")
                ]

                if not enabled_strategies:
                    raise ValueError("No sampling strategy selected.")

                total_strat = len(enabled_strategies)

                for strat_idx, strat in enumerate(enabled_strategies):
                    if self._cancelled:
                        raise FFmpegError("Operation cancelled by user.")

                    cfg = self.sampling_config[strat]
                    self.status_updated.emit(
                        f"[{vid_idx + 1}/{total_videos}] {filename}: "
                        f"Generating {strat} candidates..."
                    )

                    def strat_progress(p, _si=strat_idx, _ts=total_strat):
                        base = (vid_idx / total_videos) * 100.0
                        video_portion = (1.0 / total_videos) * 100.0
                        strat_portion = ((_si + p / 100.0) / _ts) * 0.2 * video_portion
                        self.overall_progress.emit(base + strat_portion)
                        self.video_progress.emit(((_si + p / 100.0) / _ts) * 20.0)

                    ts_list = generate_strategy_candidates(
                        strategy=strat,
                        config=cfg,
                        video_path=video_path,
                        duration_seconds=duration,
                        runner=self._runner,
                        status_callback=lambda s: self.status_updated.emit(
                            f"[{vid_idx + 1}/{total_videos}] {filename}: {s}"
                        ),
                        progress_callback=strat_progress,
                    )
                    candidate_lists[strat] = ts_list

                # Hybrid pipeline
                def pipeline_progress(p):
                    base = (vid_idx / total_videos) * 100.0
                    video_portion = (1.0 / total_videos) * 100.0
                    self.overall_progress.emit(base + (0.2 + (p / 100.0) * 0.8) * video_portion)
                    self.video_progress.emit(20.0 + (p / 100.0) * 80.0)

                exported_files, final_candidates, pipeline_stats = run_hybrid_pipeline(
                    input_path=video_path,
                    export_base_dir=video_export_dir,
                    candidate_lists=candidate_lists,
                    duration_seconds=duration,
                    output_width=output_width,
                    image_format=image_format,
                    jpeg_quality=jpeg_quality,
                    max_frames=max_frames,
                    similarity_threshold=similarity_threshold,
                    dedup_gap_seconds=DEFAULT_HYBRID_DEDUP_GAP,
                    progress_callback=pipeline_progress,
                    status_callback=lambda s: self.status_updated.emit(
                        f"[{vid_idx + 1}/{total_videos}] {filename}: {s}"
                    ),
                    runner=self._runner,
                )

                video_elapsed = round(time.time() - video_start_time, 2)
                pipeline_stats["total_processing_time_seconds"] = video_elapsed
                final_count = len(exported_files)

                # Write per-video metadata.json
                strat_results = {}
                for strat in enabled_strategies:
                    strat_results[strat] = {
                        "enabled": True,
                        "candidate_count": len(candidate_lists.get(strat, [])),
                        "config": self.sampling_config[strat],
                    }

                metadata_dict = {
                    "video": metadata.to_export_dict(),
                    "sampling_strategies": strat_results,
                    "hybrid_pipeline": pipeline_stats,
                    "output": {
                        "width": output_width,
                        "image_format": image_format,
                        "jpeg_quality": jpeg_quality,
                        "max_frames": max_frames,
                        "frames_directory": "frames/",
                    },
                    "total_processing_time_seconds": video_elapsed,
                    "total_processing_time_formatted": f"{video_elapsed:.2f}s",
                    "final_frames": [
                        {
                            "filename": os.path.basename(f),
                            "timestamp_seconds": final_candidates[i].timestamp if i < len(final_candidates) else 0.0,
                            "source": final_candidates[i].source if i < len(final_candidates) else "unknown",
                            "is_scene_change": final_candidates[i].is_scene_change if i < len(final_candidates) else False,
                        }
                        for i, f in enumerate(exported_files)
                    ],
                }

                metadata_path = os.path.join(video_export_dir, "metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

                # Collect Excel row
                excel_rows.append({
                    "video_name": filename,
                    "input_path": video_path,
                    "output_path": video_export_dir,
                    "frames_path": pipeline_stats.get("frames_directory", os.path.join(video_export_dir, "frames")),
                    "duration": metadata.duration_formatted,
                    "duration_seconds": round(metadata.duration_seconds, 2),
                    "resolution": metadata.resolution,
                    "fps": metadata.fps_display,
                    "video_codec": metadata.video_codec or "N/A",
                    "file_size": metadata.size_formatted,
                    "file_size_bytes": metadata.size_bytes,
                    "initial_candidates": pipeline_stats.get("initial_candidates", 0),
                    "after_dedup": pipeline_stats.get("after_dedup", 0),
                    "after_similarity": pipeline_stats.get("after_similarity_filter", 0),
                    "final_frame_count": final_count,
                    "processing_time_seconds": video_elapsed,
                    "status": "Success",
                    "error": "",
                })

                results = {
                    "final_frame_count": final_count,
                    "elapsed_seconds": video_elapsed,
                    "pipeline_stats": pipeline_stats,
                }

                self.video_finished.emit(vid_idx, filename, results)
                success_count += 1

            except Exception as e:
                video_elapsed = round(time.time() - video_start_time, 2)
                error_msg = str(e)

                excel_rows.append({
                    "video_name": filename,
                    "input_path": video_path,
                    "output_path": video_export_dir,
                    "frames_path": "",
                    "duration": "",
                    "duration_seconds": 0,
                    "resolution": "",
                    "fps": "",
                    "video_codec": "",
                    "file_size": "",
                    "file_size_bytes": 0,
                    "initial_candidates": 0,
                    "after_dedup": 0,
                    "after_similarity": 0,
                    "final_frame_count": 0,
                    "processing_time_seconds": video_elapsed,
                    "status": "Error",
                    "error": error_msg,
                })

                self.video_error.emit(vid_idx, filename, error_msg)
                error_count += 1

                if self._cancelled:
                    break

        # Write Excel report
        batch_elapsed = round(time.time() - batch_start, 2)

        if excel_rows and self.excel_path:
            try:
                self.status_updated.emit("Writing Excel report...")
                self._write_excel(excel_rows, batch_elapsed, total_videos, success_count, error_count)
            except Exception as e:
                self.status_updated.emit(f"Warning: Failed to write Excel: {e}")

        # Final overall progress
        self.overall_progress.emit(100)
        self.video_progress.emit(100)
        self.status_updated.emit(
            f"Batch complete: {success_count}/{total_videos} succeeded, "
            f"{error_count} errors. Total time: {batch_elapsed:.2f}s"
        )

        self.finished_all.emit({
            "total_videos": total_videos,
            "success_count": success_count,
            "error_count": error_count,
            "batch_elapsed_seconds": batch_elapsed,
            "excel_path": self.excel_path,
            "output_folder": self.output_folder,
        })

    def _write_excel(
        self,
        rows: List[dict],
        batch_elapsed: float,
        total_videos: int,
        success_count: int,
        error_count: int,
    ):
        """Write batch results to an Excel file (new sheet per run)."""
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        excel_path = self.excel_path

        # Load existing workbook or create new
        if os.path.isfile(excel_path):
            wb = load_workbook(excel_path)
        else:
            wb = Workbook()
            # Remove default sheet
            if "Sheet" in wb.sheetnames:
                del wb["Sheet"]

        # Create a new sheet with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sheet_name = f"Batch_{timestamp}"
        # Excel sheet names max 31 chars
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        ws = wb.create_sheet(title=sheet_name)

        # Styles
        header_font = Font(bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        success_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        error_fill = PatternFill(start_color="FCE4EC", end_color="FCE4EC", fill_type="solid")
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        # Summary section
        summary_font = Font(bold=True, size=12)
        ws.merge_cells("A1:D1")
        ws["A1"] = "Batch Processing Summary"
        ws["A1"].font = summary_font

        summary_data = [
            ("Input Folder:", self.input_folder),
            ("Output Folder:", self.output_folder),
            ("Total Videos:", total_videos),
            ("Successful:", success_count),
            ("Errors:", error_count),
            ("Total Batch Time:", f"{batch_elapsed:.2f}s"),
            ("Run Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]

        for i, (label, value) in enumerate(summary_data, start=2):
            ws[f"A{i}"] = label
            ws[f"A{i}"].font = Font(bold=True)
            ws[f"B{i}"] = value

        # Data table starts after summary
        table_start_row = len(summary_data) + 3

        # Headers
        headers = [
            "#",
            "Video Name",
            "Input Path",
            "Output Path",
            "Frames Path",
            "Duration",
            "Duration (s)",
            "Resolution",
            "FPS",
            "Video Codec",
            "File Size",
            "File Size (bytes)",
            "Initial Candidates",
            "After Dedup",
            "After Similarity",
            "Final Frames",
            "Processing Time (s)",
            "Status",
            "Error",
        ]

        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row=table_start_row, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            cell.border = thin_border

        # Data rows
        for row_idx, row_data in enumerate(rows, start=1):
            excel_row = table_start_row + row_idx
            is_error = row_data["status"] == "Error"
            row_fill = error_fill if is_error else success_fill

            values = [
                row_idx,
                row_data["video_name"],
                row_data["input_path"],
                row_data["output_path"],
                row_data["frames_path"],
                row_data["duration"],
                row_data["duration_seconds"],
                row_data["resolution"],
                row_data["fps"],
                row_data["video_codec"],
                row_data["file_size"],
                row_data["file_size_bytes"],
                row_data["initial_candidates"],
                row_data["after_dedup"],
                row_data["after_similarity"],
                row_data["final_frame_count"],
                row_data["processing_time_seconds"],
                row_data["status"],
                row_data["error"],
            ]

            for col_idx, value in enumerate(values, start=1):
                cell = ws.cell(row=excel_row, column=col_idx, value=value)
                cell.fill = row_fill
                cell.border = thin_border

        # Auto-adjust column widths (approximate)
        for col_idx, header in enumerate(headers, start=1):
            max_length = len(str(header))
            for row_idx in range(1, len(rows) + 1):
                cell_value = ws.cell(row=table_start_row + row_idx, column=col_idx).value
                if cell_value:
                    max_length = max(max_length, min(len(str(cell_value)), 50))
            ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max_length + 3

        wb.save(excel_path)
