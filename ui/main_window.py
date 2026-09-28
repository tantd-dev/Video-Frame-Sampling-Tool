"""
Main window for Video Frame Sampling Tool.

Provides a PySide6 GUI for configuring and running a hybrid video frame
sampling pipeline combining Simple, Scene-change, Smart, and Hierarchical
strategies into a single deduplicated, similarity-filtered, temporal-coverage
and scene-prioritized frame set.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Add project root to path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_HIERARCHICAL_COARSE_INTERVAL,
    DEFAULT_HIERARCHICAL_FINE_INTERVAL,
    DEFAULT_HIERARCHICAL_SCENE_THRESHOLD,
    DEFAULT_HYBRID_DEDUP_GAP,
    DEFAULT_HYBRID_MAX_FRAMES,
    DEFAULT_HYBRID_SIMILARITY_THRESHOLD,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_OUTPUT_WIDTH,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_SIMPLE_INTERVAL,
    DEFAULT_SMART_CANDIDATE_INTERVAL,
    DEFAULT_SMART_SIMILARITY_THRESHOLD,
    EXPORT_BASE_DIR,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from core.video_info import (
    FFprobeError,
    FFprobeNotFoundError,
    get_video_metadata,
    check_ffmpeg_available,
)
from models.video_metadata import VideoMetadata


# ============================================================================
# Worker Thread for Export (Hybrid Selection Pipeline)
# ============================================================================

class ExportWorker(QThread):
    """Background thread for candidate generation and hybrid frame selection."""

    progress_updated = Signal(float)   # 0-100
    status_updated = Signal(str)
    finished_ok = Signal(dict)         # results dict
    finished_error = Signal(str)       # error message

    def __init__(
        self,
        video_path: str,
        metadata: VideoMetadata,
        export_dir: str,
        sampling_config: dict,
        output_config: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.metadata = metadata
        self.export_dir = export_dir
        self.sampling_config = sampling_config
        self.output_config = output_config
        self._runner = None

    def cancel(self):
        """Request cancellation."""
        if self._runner:
            self._runner.cancel()

    def run(self):
        """Execute candidate generation and hybrid frame selection pipeline."""
        from core.ffmpeg_runner import FFmpegRunner, FFmpegError
        from core.hybrid_pipeline import (
            generate_strategy_candidates,
            run_hybrid_pipeline,
        )

        self._runner = FFmpegRunner()
        results = {}
        start_worker_time = time.time()

        try:
            duration = self.metadata.duration_seconds
            output_width = self.output_config["width"]
            image_format = self.output_config["format"]
            jpeg_quality = self.output_config["jpeg_quality"]
            max_frames = self.output_config.get("max_frames", 0)
            similarity_threshold = self.output_config.get("similarity_threshold", DEFAULT_HYBRID_SIMILARITY_THRESHOLD)

            # Step 1: Generate candidates from enabled sampling strategies
            candidate_lists = {}
            enabled_strategies = [
                s for s in ("simple", "scene", "smart", "hierarchical")
                if self.sampling_config.get(s, {}).get("enabled")
            ]

            if not enabled_strategies:
                self.finished_error.emit("No sampling strategy selected.")
                return

            self.status_updated.emit("Generating candidate frames from enabled strategies...")
            total_strat = len(enabled_strategies)

            for idx, strat in enumerate(enabled_strategies):
                if self._runner.is_cancelled:
                    raise FFmpegError("Operation cancelled by user.")

                cfg = self.sampling_config[strat]
                self.status_updated.emit(f"Step 1/8: Generating candidates for {strat} sampling...")

                def strat_progress(p):
                    self.progress_updated.emit(((idx + p / 100.0) / total_strat) * 20.0)

                ts_list = generate_strategy_candidates(
                    strategy=strat,
                    config=cfg,
                    video_path=self.video_path,
                    duration_seconds=duration,
                    runner=self._runner,
                    status_callback=lambda s: self.status_updated.emit(s),
                    progress_callback=strat_progress,
                )
                candidate_lists[strat] = ts_list
                results[strat] = {
                    "enabled": True,
                    "candidate_count": len(ts_list),
                    "config": cfg,
                }

            # Steps 2-8: Execute hybrid pipeline
            def pipeline_progress(p):
                self.progress_updated.emit(20.0 + (p / 100.0) * 80.0)

            exported_files, final_candidates, pipeline_stats = run_hybrid_pipeline(
                input_path=self.video_path,
                export_base_dir=self.export_dir,
                candidate_lists=candidate_lists,
                duration_seconds=duration,
                output_width=output_width,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                max_frames=max_frames,
                similarity_threshold=similarity_threshold,
                dedup_gap_seconds=DEFAULT_HYBRID_DEDUP_GAP,
                progress_callback=pipeline_progress,
                status_callback=lambda s: self.status_updated.emit(s),
                runner=self._runner,
            )

            total_elapsed = round(time.time() - start_worker_time, 2)
            pipeline_stats["total_processing_time_seconds"] = total_elapsed

            # Write metadata.json
            self.status_updated.emit("Writing metadata.json...")
            metadata_dict = {
                "video": self.metadata.to_export_dict(),
                "sampling_strategies": results,
                "hybrid_pipeline": pipeline_stats,
                "output": {
                    "width": output_width,
                    "image_format": image_format,
                    "jpeg_quality": jpeg_quality,
                    "max_frames": max_frames,
                    "frames_directory": "frames/",
                },
                "total_processing_time_seconds": total_elapsed,
                "total_processing_time_formatted": f"{total_elapsed:.2f}s",
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

            metadata_path = os.path.join(self.export_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            self.progress_updated.emit(100)
            self.status_updated.emit(f"Export completed in {total_elapsed:.2f}s.")
            results["pipeline_stats"] = pipeline_stats
            results["final_frame_count"] = len(exported_files)
            results["elapsed_seconds"] = total_elapsed
            self.finished_ok.emit(results)

        except FFmpegError as e:
            self.finished_error.emit(str(e))
        except Exception as e:
            self.finished_error.emit(f"Unexpected error: {e}\n{traceback.format_exc()}")


# ============================================================================
# Main Window
# ============================================================================

class MainWindow(QMainWindow):
    """Main application window."""

    # Mode constants
    MODE_SINGLE = "single"
    MODE_MULTI = "multi"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(560, 820)
        self.resize(600, 950)

        self._metadata: Optional[VideoMetadata] = None
        self._worker: Optional[ExportWorker] = None
        self._batch_worker = None
        self._start_time: float = 0.0
        self._mode: str = self.MODE_SINGLE

        # Processing time ticker
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_timer_tick)

        self._build_ui()

    # ----------------------------------------------------------------
    # UI Construction
    # ----------------------------------------------------------------

    def _build_ui(self):
        """Build the main UI layout."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(18, 18, 18, 18)

        # Title
        title = QLabel(APP_NAME)
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # --- Mode Toggle ---
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_single_btn = QPushButton("📄 Single Video")
        self.mode_multi_btn = QPushButton("📁 Multi Video (Batch)")
        self.mode_single_btn.setFixedHeight(34)
        self.mode_multi_btn.setFixedHeight(34)
        self.mode_single_btn.setCheckable(True)
        self.mode_multi_btn.setCheckable(True)
        self.mode_single_btn.setChecked(True)
        self.mode_single_btn.clicked.connect(lambda: self._switch_mode(self.MODE_SINGLE))
        self.mode_multi_btn.clicked.connect(lambda: self._switch_mode(self.MODE_MULTI))
        mode_layout.addWidget(self.mode_single_btn)
        mode_layout.addWidget(self.mode_multi_btn)
        main_layout.addWidget(mode_group)

        # --- Single Video Selection ---
        self.single_video_group = QGroupBox("Video File")
        video_layout = QHBoxLayout(self.single_video_group)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Select a video file to sample...")
        self.video_path_edit.setReadOnly(True)
        video_layout.addWidget(self.video_path_edit, stretch=1)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(85)
        self.browse_btn.clicked.connect(self._on_browse)
        video_layout.addWidget(self.browse_btn)
        main_layout.addWidget(self.single_video_group)

        # --- Multi Video Selection ---
        self.multi_video_group = QGroupBox("Video Input Folder")
        multi_video_layout = QHBoxLayout(self.multi_video_group)
        self.multi_input_edit = QLineEdit()
        self.multi_input_edit.setPlaceholderText("Select folder containing video files...")
        self.multi_input_edit.setReadOnly(True)
        multi_video_layout.addWidget(self.multi_input_edit, stretch=1)
        self.multi_input_btn = QPushButton("Browse")
        self.multi_input_btn.setFixedWidth(85)
        self.multi_input_btn.clicked.connect(self._on_browse_multi_input)
        multi_video_layout.addWidget(self.multi_input_btn)
        main_layout.addWidget(self.multi_video_group)
        self.multi_video_group.setVisible(False)

        # --- Multi Output Folder ---
        self.multi_output_group = QGroupBox("Output Folder (Multi)")
        multi_output_layout = QHBoxLayout(self.multi_output_group)
        self.multi_output_edit = QLineEdit()
        self.multi_output_edit.setPlaceholderText("Select output folder for batch results...")
        self.multi_output_edit.setReadOnly(True)
        multi_output_layout.addWidget(self.multi_output_edit, stretch=1)
        self.multi_output_btn = QPushButton("Browse")
        self.multi_output_btn.setFixedWidth(85)
        self.multi_output_btn.clicked.connect(self._on_browse_multi_output)
        multi_output_layout.addWidget(self.multi_output_btn)
        main_layout.addWidget(self.multi_output_group)
        self.multi_output_group.setVisible(False)

        # --- Excel Report File ---
        self.excel_group = QGroupBox("Excel Report File")
        excel_layout = QHBoxLayout(self.excel_group)
        self.excel_path_edit = QLineEdit()
        self.excel_path_edit.setPlaceholderText("Select or create an Excel file (.xlsx) for batch report...")
        self.excel_path_edit.setReadOnly(True)
        excel_layout.addWidget(self.excel_path_edit, stretch=1)
        self.excel_browse_btn = QPushButton("Browse")
        self.excel_browse_btn.setFixedWidth(85)
        self.excel_browse_btn.clicked.connect(self._on_browse_excel)
        excel_layout.addWidget(self.excel_browse_btn)
        main_layout.addWidget(self.excel_group)
        self.excel_group.setVisible(False)

        # --- Video Information (single mode only) ---
        self.info_group = QGroupBox("Video Information")
        info_layout = QFormLayout(self.info_group)
        info_layout.setLabelAlignment(Qt.AlignRight)

        self.info_labels = {}
        info_fields = [
            ("filename", "File name:"),
            ("filesize", "File size:"),
            ("duration", "Duration:"),
            ("resolution", "Resolution:"),
            ("fps", "FPS:"),
            ("video_codec", "Video codec:"),
            ("bitrate", "Bitrate:"),
            ("audio_codec", "Audio codec:"),
            ("audio_bitrate", "Audio bitrate:"),
        ]
        for key, label_text in info_fields:
            lbl = QLabel("—")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.info_labels[key] = lbl
            info_layout.addRow(label_text, lbl)

        main_layout.addWidget(self.info_group)

        # --- Output folder name (single mode only) ---
        self.output_group = QGroupBox("Output Folder")
        output_layout = QHBoxLayout(self.output_group)
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("e.g. video_frames_01")
        self.output_name_edit.setText("output_frames")
        output_layout.addWidget(self.output_name_edit)
        main_layout.addWidget(self.output_group)

        # --- Multi Video Info (multi mode only) ---
        self.multi_info_group = QGroupBox("Batch Information")
        multi_info_layout = QFormLayout(self.multi_info_group)
        multi_info_layout.setLabelAlignment(Qt.AlignRight)
        self.multi_video_count_label = QLabel("—")
        multi_info_layout.addRow("Videos found:", self.multi_video_count_label)
        self.multi_video_list_label = QLabel("—")
        self.multi_video_list_label.setWordWrap(True)
        self.multi_video_list_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        multi_info_layout.addRow("Files:", self.multi_video_list_label)
        main_layout.addWidget(self.multi_info_group)
        self.multi_info_group.setVisible(False)

        # --- Candidate Sampling Strategies ---
        sampling_group = QGroupBox("Candidate Sampling Strategies (Enabled sources are merged)")
        sampling_layout = QVBoxLayout(sampling_group)

        # Simple Sampling
        self.simple_check = QCheckBox("Simple Sampling (Fixed interval)")
        self.simple_check.setChecked(True)
        sampling_layout.addWidget(self.simple_check)

        simple_params = QWidget()
        simple_form = QFormLayout(simple_params)
        simple_form.setContentsMargins(24, 2, 0, 6)
        self.simple_interval_spin = QSpinBox()
        self.simple_interval_spin.setRange(1, 3600)
        self.simple_interval_spin.setValue(DEFAULT_SIMPLE_INTERVAL)
        self.simple_interval_spin.setSuffix(" seconds")
        simple_form.addRow("Interval:", self.simple_interval_spin)
        sampling_layout.addWidget(simple_params)

        # Scene-change Sampling
        self.scene_check = QCheckBox("Scene-change Sampling (FFmpeg scene detection)")
        sampling_layout.addWidget(self.scene_check)

        scene_params = QWidget()
        scene_form = QFormLayout(scene_params)
        scene_form.setContentsMargins(24, 2, 0, 6)
        self.scene_threshold_edit = QLineEdit(str(DEFAULT_SCENE_THRESHOLD))
        self.scene_threshold_edit.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self.scene_threshold_edit.setFixedWidth(80)
        scene_form.addRow("Scene threshold:", self.scene_threshold_edit)
        sampling_layout.addWidget(scene_params)

        # Smart Sampling
        self.smart_check = QCheckBox("Smart Sampling (Heuristic candidate generation)")
        sampling_layout.addWidget(self.smart_check)

        smart_params = QWidget()
        smart_form = QFormLayout(smart_params)
        smart_form.setContentsMargins(24, 2, 0, 6)
        self.smart_interval_spin = QSpinBox()
        self.smart_interval_spin.setRange(1, 3600)
        self.smart_interval_spin.setValue(DEFAULT_SMART_CANDIDATE_INTERVAL)
        self.smart_interval_spin.setSuffix(" seconds")
        smart_form.addRow("Candidate interval:", self.smart_interval_spin)
        self.smart_similarity_edit = QLineEdit(str(DEFAULT_SMART_SIMILARITY_THRESHOLD))
        self.smart_similarity_edit.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self.smart_similarity_edit.setFixedWidth(80)
        smart_form.addRow("Similarity threshold:", self.smart_similarity_edit)
        sampling_layout.addWidget(smart_params)

        # Hierarchical Processing
        self.hier_check = QCheckBox("Hierarchical Sampling (Coarse overview + Fine segments)")
        sampling_layout.addWidget(self.hier_check)

        hier_params = QWidget()
        hier_form = QFormLayout(hier_params)
        hier_form.setContentsMargins(24, 2, 0, 6)
        self.hier_coarse_spin = QSpinBox()
        self.hier_coarse_spin.setRange(1, 3600)
        self.hier_coarse_spin.setValue(DEFAULT_HIERARCHICAL_COARSE_INTERVAL)
        self.hier_coarse_spin.setSuffix(" seconds")
        hier_form.addRow("Coarse interval:", self.hier_coarse_spin)
        self.hier_fine_spin = QSpinBox()
        self.hier_fine_spin.setRange(1, 3600)
        self.hier_fine_spin.setValue(DEFAULT_HIERARCHICAL_FINE_INTERVAL)
        self.hier_fine_spin.setSuffix(" seconds")
        hier_form.addRow("Fine interval:", self.hier_fine_spin)
        self.hier_threshold_edit = QLineEdit(str(DEFAULT_HIERARCHICAL_SCENE_THRESHOLD))
        self.hier_threshold_edit.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self.hier_threshold_edit.setFixedWidth(80)
        hier_form.addRow("Change threshold:", self.hier_threshold_edit)
        sampling_layout.addWidget(hier_params)

        main_layout.addWidget(sampling_group)

        # --- Hybrid Frame Selection Settings ---
        hybrid_group = QGroupBox("Hybrid Frame Selection Settings")
        hybrid_layout = QFormLayout(hybrid_group)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 10000)
        self.max_frames_spin.setValue(DEFAULT_HYBRID_MAX_FRAMES)
        self.max_frames_spin.setSpecialValueText("Unlimited (0)")
        self.max_frames_spin.setSuffix(" frames")
        self.max_frames_spin.setToolTip("Maximum number of final frames to export. Set to 0 for unlimited.")
        hybrid_layout.addRow("Max final frames:", self.max_frames_spin)

        self.hybrid_similarity_edit = QLineEdit(str(DEFAULT_HYBRID_SIMILARITY_THRESHOLD))
        self.hybrid_similarity_edit.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self.hybrid_similarity_edit.setFixedWidth(80)
        self.hybrid_similarity_edit.setToolTip("Perceptual similarity threshold (0.01-1.0). Frames above this threshold are deemed redundant.")
        hybrid_layout.addRow("Similarity threshold:", self.hybrid_similarity_edit)

        main_layout.addWidget(hybrid_group)

        # --- Output Settings ---
        output_settings_group = QGroupBox("Output Settings")
        output_settings_layout = QFormLayout(output_settings_group)

        # Resolution
        res_widget = QWidget()
        res_layout = QHBoxLayout(res_widget)
        res_layout.setContentsMargins(0, 0, 0, 0)
        self.output_width_spin = QSpinBox()
        self.output_width_spin.setRange(320, 7680)
        self.output_width_spin.setValue(DEFAULT_OUTPUT_WIDTH)
        self.output_width_spin.setSingleStep(10)
        res_layout.addWidget(self.output_width_spin)
        res_layout.addWidget(QLabel("x"))
        self.output_height_label = QLabel("auto")
        self.output_height_label.setToolTip("Height is automatically calculated to maintain aspect ratio")
        res_layout.addWidget(self.output_height_label)
        res_layout.addStretch()
        output_settings_layout.addRow("Output resolution:", res_widget)

        # Image format
        self.format_combo = QComboBox()
        self.format_combo.addItems(SUPPORTED_IMAGE_FORMATS)
        self.format_combo.setCurrentText(DEFAULT_IMAGE_FORMAT)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        output_settings_layout.addRow("Image format:", self.format_combo)

        # JPEG quality
        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(1, 100)
        self.jpeg_quality_spin.setValue(DEFAULT_JPEG_QUALITY)
        self.jpeg_quality_label = QLabel("JPEG quality:")
        output_settings_layout.addRow(self.jpeg_quality_label, self.jpeg_quality_spin)

        main_layout.addWidget(output_settings_group)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Frames (Hybrid Pipeline)")
        self.export_btn.setFixedHeight(38)
        export_font = QFont()
        export_font.setBold(True)
        self.export_btn.setFont(export_font)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.export_btn, stretch=2)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedHeight(38)
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)

        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.setFixedHeight(38)
        self.reset_btn.setFixedWidth(115)
        self.reset_btn.setToolTip("Reset all sampling parameters and settings back to their default values")
        self.reset_btn.clicked.connect(self._on_reset_defaults)
        btn_layout.addWidget(self.reset_btn)

        main_layout.addLayout(btn_layout)

        # --- Progress & Execution Time ---
        progress_group = QGroupBox("Progress & Processing Time")
        progress_layout = QVBoxLayout(progress_group)

        # Overall progress (always shown)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Per-video progress (multi mode only)
        self.video_progress_label = QLabel("Current video progress:")
        self.video_progress_bar = QProgressBar()
        self.video_progress_bar.setRange(0, 100)
        self.video_progress_bar.setValue(0)
        progress_layout.addWidget(self.video_progress_label)
        progress_layout.addWidget(self.video_progress_bar)
        self.video_progress_label.setVisible(False)
        self.video_progress_bar.setVisible(False)

        self.time_label = QLabel("Total processing time: —")
        time_font = QFont()
        time_font.setBold(True)
        self.time_label.setFont(time_font)
        progress_layout.addWidget(self.time_label)

        main_layout.addWidget(progress_group)

        # --- Status ---
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        main_layout.addWidget(status_group)

        main_layout.addStretch()

        # Apply initial mode styling
        self._update_mode_button_styles()

    # ----------------------------------------------------------------
    # Mode Switching
    # ----------------------------------------------------------------

    def _switch_mode(self, mode: str):
        """Switch between single and multi-video modes."""
        if self._mode == mode:
            return

        self._mode = mode
        is_single = mode == self.MODE_SINGLE
        is_multi = mode == self.MODE_MULTI

        # Toggle button states
        self.mode_single_btn.setChecked(is_single)
        self.mode_multi_btn.setChecked(is_multi)

        # Single mode widgets
        self.single_video_group.setVisible(is_single)
        self.info_group.setVisible(is_single)
        self.output_group.setVisible(is_single)

        # Multi mode widgets
        self.multi_video_group.setVisible(is_multi)
        self.multi_output_group.setVisible(is_multi)
        self.excel_group.setVisible(is_multi)
        self.multi_info_group.setVisible(is_multi)
        self.video_progress_label.setVisible(is_multi)
        self.video_progress_bar.setVisible(is_multi)

        # Update export button text
        if is_single:
            self.export_btn.setText("Export Frames (Hybrid Pipeline)")
        else:
            self.export_btn.setText("🚀 Start Batch Export")

        self._update_mode_button_styles()
        self.status_label.setText(
            f"Switched to {'Single Video' if is_single else 'Multi Video (Batch)'} mode."
        )

    def _update_mode_button_styles(self):
        """Apply visual styles to mode toggle buttons."""
        active_style = (
            "QPushButton { background-color: #4472C4; color: white; "
            "font-weight: bold; border: 2px solid #3461B3; border-radius: 4px; }"
        )
        inactive_style = (
            "QPushButton { background-color: #E0E0E0; color: #555; "
            "border: 1px solid #CCC; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D0D0D0; }"
        )
        self.mode_single_btn.setStyleSheet(
            active_style if self._mode == self.MODE_SINGLE else inactive_style
        )
        self.mode_multi_btn.setStyleSheet(
            active_style if self._mode == self.MODE_MULTI else inactive_style
        )

    # ----------------------------------------------------------------
    # Event Handlers & Timer
    # ----------------------------------------------------------------

    def _on_timer_tick(self):
        """Update live processing time while worker is running."""
        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            self.time_label.setText(f"Processing time: {elapsed:.1f}s")

    def _on_browse(self):
        """Open file dialog to select a video."""
        ext_filter = "Video files ({});;All files (*.*)".format(
            " ".join(f"*.{e}" for e in SUPPORTED_VIDEO_EXTENSIONS)
        )
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", ext_filter
        )
        if not filepath:
            return

        self.video_path_edit.setText(filepath)
        self._load_metadata(filepath)

    def _on_browse_multi_input(self):
        """Open folder dialog to select input video folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Video Input Folder", ""
        )
        if not folder:
            return
        self.multi_input_edit.setText(folder)
        self._scan_input_folder(folder)

    def _on_browse_multi_output(self):
        """Open folder dialog to select output folder for batch results."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Batch Results", ""
        )
        if not folder:
            return
        self.multi_output_edit.setText(folder)

    def _on_browse_excel(self):
        """Open file dialog to select or create an Excel file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select or Create Excel Report File",
            "",
            "Excel files (*.xlsx);;All files (*.*)",
        )
        if not filepath:
            return
        # Ensure .xlsx extension
        if not filepath.lower().endswith(".xlsx"):
            filepath += ".xlsx"
        self.excel_path_edit.setText(filepath)

    def _scan_input_folder(self, folder: str):
        """Scan the input folder and display video file count."""
        video_files = []
        for entry in sorted(os.listdir(folder)):
            ext = Path(entry).suffix.lstrip('.').lower()
            if ext in SUPPORTED_VIDEO_EXTENSIONS:
                video_files.append(entry)

        count = len(video_files)
        self.multi_video_count_label.setText(str(count))

        if count == 0:
            self.multi_video_list_label.setText(
                "No supported video files found.\n"
                f"Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
            )
            self.status_label.setText("No videos found in selected folder.")
        else:
            # Show first 15 files, truncate if more
            display_list = video_files[:15]
            text = "\n".join(f"  • {f}" for f in display_list)
            if count > 15:
                text += f"\n  ... and {count - 15} more"
            self.multi_video_list_label.setText(text)
            self.status_label.setText(f"Found {count} video(s) in input folder.")

    def _load_metadata(self, filepath: str):
        """Load and display video metadata."""
        self.status_label.setText("Reading video metadata...")
        self._metadata = None

        # Clear info
        for lbl in self.info_labels.values():
            lbl.setText("—")

        try:
            metadata = get_video_metadata(filepath)
            self._metadata = metadata

            self.info_labels["filename"].setText(metadata.filename)
            self.info_labels["filesize"].setText(metadata.size_formatted)
            self.info_labels["duration"].setText(metadata.duration_formatted)
            self.info_labels["resolution"].setText(metadata.resolution)
            self.info_labels["fps"].setText(metadata.fps_display)
            self.info_labels["video_codec"].setText(metadata.video_codec or "N/A")
            self.info_labels["bitrate"].setText(metadata.video_bitrate_formatted)
            self.info_labels["audio_codec"].setText(metadata.audio_codec or "N/A")
            self.info_labels["audio_bitrate"].setText(metadata.audio_bitrate_formatted)

            # Suggest output folder name from filename
            base_name = Path(filepath).stem
            safe_name = base_name.replace(" ", "_").lower()
            self.output_name_edit.setText(safe_name)

            self.status_label.setText("Video metadata loaded successfully.")

        except FFprobeNotFoundError as e:
            QMessageBox.critical(self, "FFprobe Not Found", str(e))
            self.status_label.setText("Error: FFprobe not found.")
        except FileNotFoundError as e:
            QMessageBox.warning(self, "File Not Found", str(e))
            self.status_label.setText("Error: File not found.")
        except FFprobeError as e:
            QMessageBox.warning(self, "FFprobe Error", str(e))
            self.status_label.setText(f"Error: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read metadata:\n{e}")
            self.status_label.setText(f"Error: {e}")

    def _on_format_changed(self, fmt: str):
        """Show/hide JPEG quality when format changes."""
        is_jpeg = fmt.upper() == "JPEG"
        self.jpeg_quality_spin.setVisible(is_jpeg)
        self.jpeg_quality_label.setVisible(is_jpeg)

    def _on_reset_defaults(self):
        """Reset all sampling parameters and UI values to their defaults."""
        self.simple_check.setChecked(True)
        self.simple_interval_spin.setValue(DEFAULT_SIMPLE_INTERVAL)

        self.scene_check.setChecked(False)
        self.scene_threshold_edit.setText(str(DEFAULT_SCENE_THRESHOLD))

        self.smart_check.setChecked(False)
        self.smart_interval_spin.setValue(DEFAULT_SMART_CANDIDATE_INTERVAL)
        self.smart_similarity_edit.setText(str(DEFAULT_SMART_SIMILARITY_THRESHOLD))

        self.hier_check.setChecked(False)
        self.hier_coarse_spin.setValue(DEFAULT_HIERARCHICAL_COARSE_INTERVAL)
        self.hier_fine_spin.setValue(DEFAULT_HIERARCHICAL_FINE_INTERVAL)
        self.hier_threshold_edit.setText(str(DEFAULT_HIERARCHICAL_SCENE_THRESHOLD))

        self.max_frames_spin.setValue(DEFAULT_HYBRID_MAX_FRAMES)
        self.hybrid_similarity_edit.setText(str(DEFAULT_HYBRID_SIMILARITY_THRESHOLD))

        self.output_width_spin.setValue(DEFAULT_OUTPUT_WIDTH)
        self.format_combo.setCurrentText(DEFAULT_IMAGE_FORMAT)
        self.jpeg_quality_spin.setValue(DEFAULT_JPEG_QUALITY)

        if self._metadata and self._metadata.filename:
            base_name = Path(self._metadata.filename).stem
            safe_name = base_name.replace(" ", "_").lower()
            self.output_name_edit.setText(safe_name)
        else:
            self.output_name_edit.setText("output_frames")

        self.progress_bar.setValue(0)
        self.video_progress_bar.setValue(0)
        self.time_label.setText("Total processing time: —")
        self.status_label.setText("Ready (defaults restored).")

    # ----------------------------------------------------------------
    # Export Logic
    # ----------------------------------------------------------------

    def _build_configs(self):
        """Build sampling_config and output_config from current UI values.

        Returns:
            (sampling_config, output_config) tuple, or None if validation fails.
        """
        # Check at least one strategy selected
        any_selected = (
            self.simple_check.isChecked()
            or self.scene_check.isChecked()
            or self.smart_check.isChecked()
            or self.hier_check.isChecked()
        )
        if not any_selected:
            QMessageBox.warning(
                self, "No Strategy Selected",
                "Please select at least one candidate sampling strategy."
            )
            return None

        # Validate parameters
        try:
            scene_threshold = float(self.scene_threshold_edit.text()) if self.scene_check.isChecked() else 0.3
            smart_similarity = float(self.smart_similarity_edit.text()) if self.smart_check.isChecked() else 0.9
            hier_threshold = float(self.hier_threshold_edit.text()) if self.hier_check.isChecked() else 0.3
            hybrid_similarity = float(self.hybrid_similarity_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Parameter", "Please enter valid numeric values for thresholds.")
            return None

        if self.scene_check.isChecked() and not (0.01 <= scene_threshold <= 1.0):
            QMessageBox.warning(self, "Invalid Threshold", "Scene threshold must be between 0.01 and 1.0.")
            return None

        if self.smart_check.isChecked() and not (0.01 <= smart_similarity <= 1.0):
            QMessageBox.warning(self, "Invalid Threshold", "Smart similarity threshold must be between 0.01 and 1.0.")
            return None

        if not (0.01 <= hybrid_similarity <= 1.0):
            QMessageBox.warning(self, "Invalid Threshold", "Hybrid similarity threshold must be between 0.01 and 1.0.")
            return None

        sampling_config = {
            "simple": {
                "enabled": self.simple_check.isChecked(),
                "interval": self.simple_interval_spin.value(),
            },
            "scene": {
                "enabled": self.scene_check.isChecked(),
                "threshold": scene_threshold,
            },
            "smart": {
                "enabled": self.smart_check.isChecked(),
                "candidate_interval": self.smart_interval_spin.value(),
                "similarity_threshold": smart_similarity,
            },
            "hierarchical": {
                "enabled": self.hier_check.isChecked(),
                "coarse_interval": self.hier_coarse_spin.value(),
                "fine_interval": self.hier_fine_spin.value(),
                "scene_threshold": hier_threshold,
            },
        }

        output_config = {
            "width": self.output_width_spin.value(),
            "format": self.format_combo.currentText().lower(),
            "jpeg_quality": self.jpeg_quality_spin.value(),
            "max_frames": self.max_frames_spin.value(),
            "similarity_threshold": hybrid_similarity,
        }

        return sampling_config, output_config

    def _on_export(self):
        """Start the export process (single or batch depending on mode)."""
        if self._mode == self.MODE_SINGLE:
            self._on_export_single()
        else:
            self._on_export_batch()

    def _on_export_single(self):
        """Start the hybrid export process for a single video."""
        # Validate inputs
        if not self._metadata:
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return

        video_path = self.video_path_edit.text()
        if not os.path.isfile(video_path):
            QMessageBox.warning(self, "File Not Found", "Selected video file no longer exists.")
            return

        if not check_ffmpeg_available():
            QMessageBox.critical(
                self, "FFmpeg Not Found",
                "FFmpeg was not found. Please install FFmpeg and add it to PATH."
            )
            return

        configs = self._build_configs()
        if configs is None:
            return
        sampling_config, output_config = configs

        # Validate output folder name
        from utils.filesystem import sanitize_folder_name
        folder_name = sanitize_folder_name(self.output_name_edit.text())
        if not folder_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid output folder name.")
            return

        # Build export directory
        export_dir = os.path.join(PROJECT_ROOT, EXPORT_BASE_DIR, folder_name)

        # Disable UI during export
        self._set_ui_enabled(False)
        self.progress_bar.setValue(0)

        # Start timer for live processing time
        self._start_time = time.time()
        self.time_label.setText("Processing time: 0.0s")
        self._timer.start()

        # Start worker
        self._worker = ExportWorker(
            video_path=video_path,
            metadata=self._metadata,
            export_dir=export_dir,
            sampling_config=sampling_config,
            output_config=output_config,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.status_updated.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_export_finished)
        self._worker.finished_error.connect(self._on_export_error)
        self._worker.start()

    def _on_export_batch(self):
        """Start the batch export process for multiple videos."""
        # Validate inputs
        input_folder = self.multi_input_edit.text()
        if not input_folder or not os.path.isdir(input_folder):
            QMessageBox.warning(
                self, "No Input Folder",
                "Please select a valid input folder containing video files."
            )
            return

        output_folder = self.multi_output_edit.text()
        if not output_folder:
            QMessageBox.warning(
                self, "No Output Folder",
                "Please select an output folder for batch results."
            )
            return

        excel_path = self.excel_path_edit.text()
        if not excel_path:
            QMessageBox.warning(
                self, "No Excel File",
                "Please select or create an Excel file (.xlsx) for the batch report."
            )
            return

        if not check_ffmpeg_available():
            QMessageBox.critical(
                self, "FFmpeg Not Found",
                "FFmpeg was not found. Please install FFmpeg and add it to PATH."
            )
            return

        configs = self._build_configs()
        if configs is None:
            return
        sampling_config, output_config = configs

        # Create output folder if needed
        os.makedirs(output_folder, exist_ok=True)

        # Confirm start
        # Count videos first
        video_count = sum(
            1 for entry in os.listdir(input_folder)
            if Path(entry).suffix.lstrip('.').lower() in SUPPORTED_VIDEO_EXTENSIONS
        )
        if video_count == 0:
            QMessageBox.warning(
                self, "No Videos",
                f"No supported video files found in:\n{input_folder}"
            )
            return

        reply = QMessageBox.question(
            self, "Start Batch Processing?",
            f"Ready to process {video_count} video(s).\n\n"
            f"Input: {input_folder}\n"
            f"Output: {output_folder}\n"
            f"Excel: {excel_path}\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Disable UI during export
        self._set_ui_enabled(False)
        self.progress_bar.setValue(0)
        self.video_progress_bar.setValue(0)

        # Start timer
        self._start_time = time.time()
        self.time_label.setText("Processing time: 0.0s")
        self._timer.start()

        # Start batch worker
        from core.batch_worker import BatchExportWorker
        self._batch_worker = BatchExportWorker(
            input_folder=input_folder,
            output_folder=output_folder,
            excel_path=excel_path,
            sampling_config=sampling_config,
            output_config=output_config,
        )
        self._batch_worker.overall_progress.connect(self._on_progress)
        self._batch_worker.video_progress.connect(self._on_video_progress)
        self._batch_worker.status_updated.connect(self._on_status)
        self._batch_worker.video_started.connect(self._on_batch_video_started)
        self._batch_worker.video_finished.connect(self._on_batch_video_finished)
        self._batch_worker.video_error.connect(self._on_batch_video_error)
        self._batch_worker.finished_all.connect(self._on_batch_finished)
        self._batch_worker.finished_error.connect(self._on_batch_fatal_error)
        self._batch_worker.start()

    def _on_cancel(self):
        """Cancel the running export."""
        if self._worker:
            self.status_label.setText("Cancelling...")
            self._worker.cancel()
        if self._batch_worker:
            self.status_label.setText("Cancelling batch...")
            self._batch_worker.cancel()

    @Slot(float)
    def _on_progress(self, value: float):
        """Update progress bar."""
        self.progress_bar.setValue(int(value))

    @Slot(float)
    def _on_video_progress(self, value: float):
        """Update per-video progress bar (multi mode)."""
        self.video_progress_bar.setValue(int(value))

    @Slot(str)
    def _on_status(self, text: str):
        """Update status label."""
        self.status_label.setText(text)

    # ----------------------------------------------------------------
    # Single Video Callbacks
    # ----------------------------------------------------------------

    @Slot(dict)
    def _on_export_finished(self, results: dict):
        """Handle successful export completion."""
        self._timer.stop()
        self._set_ui_enabled(True)
        self.progress_bar.setValue(100)

        final_count = results.get("final_frame_count", 0)
        elapsed = results.get("elapsed_seconds", round(time.time() - self._start_time, 2))
        stats = results.get("pipeline_stats", {})

        self.time_label.setText(f"Total processing time: {elapsed:.2f}s")
        self.status_label.setText(
            f"Export completed in {elapsed:.2f}s. {final_count} final frames exported into frames/."
        )

        strat_breakdown = ", ".join(
            f"{k} ({v})" for k, v in stats.get("strategy_breakdown", {}).items()
        ) or "None"

        frames_dir = stats.get("frames_directory", os.path.join(self._worker.export_dir, "frames"))

        QMessageBox.information(
            self, "Hybrid Frame Selection Complete",
            f"Successfully selected and exported {final_count} frames.\n\n"
            f"Total processing time: {elapsed:.2f} seconds\n\n"
            f"Candidate sources: {strat_breakdown}\n"
            f"Initial candidate pool: {stats.get('initial_candidates', 0)} frames\n"
            f"After timestamp deduplication: {stats.get('after_dedup', 0)} frames\n"
            f"After visual similarity filtering: {stats.get('after_similarity_filter', 0)} frames\n"
            f"Final exported frames: {final_count} frames\n\n"
            f"Destination directory:\n{frames_dir}"
        )
        self._worker = None

    @Slot(str)
    def _on_export_error(self, error_msg: str):
        """Handle export error."""
        self._timer.stop()
        self._set_ui_enabled(True)

        elapsed = round(time.time() - self._start_time, 2)
        self.time_label.setText(f"Processing time: {elapsed:.2f}s (stopped)")
        self.status_label.setText(f"Error: {error_msg}")

        if "cancelled" in error_msg.lower():
            QMessageBox.information(self, "Cancelled", "Export was cancelled by user.")
        else:
            QMessageBox.critical(self, "Export Error", error_msg)
        self._worker = None

    # ----------------------------------------------------------------
    # Batch Video Callbacks
    # ----------------------------------------------------------------

    @Slot(int, str)
    def _on_batch_video_started(self, index: int, filename: str):
        """Handle the start of processing for a video in the batch."""
        self.video_progress_bar.setValue(0)

    @Slot(int, str, dict)
    def _on_batch_video_finished(self, index: int, filename: str, results: dict):
        """Handle successful completion of a single video in the batch."""
        self.video_progress_bar.setValue(100)

    @Slot(int, str, str)
    def _on_batch_video_error(self, index: int, filename: str, error_msg: str):
        """Handle an error processing a single video in the batch."""
        self.video_progress_bar.setValue(0)

    @Slot(dict)
    def _on_batch_finished(self, results: dict):
        """Handle completion of the entire batch."""
        self._timer.stop()
        self._set_ui_enabled(True)
        self.progress_bar.setValue(100)
        self.video_progress_bar.setValue(100)

        total = results.get("total_videos", 0)
        success = results.get("success_count", 0)
        errors = results.get("error_count", 0)
        elapsed = results.get("batch_elapsed_seconds", round(time.time() - self._start_time, 2))
        excel_path = results.get("excel_path", "")
        output_folder = results.get("output_folder", "")

        self.time_label.setText(f"Total processing time: {elapsed:.2f}s")
        self.status_label.setText(
            f"Batch complete: {success}/{total} videos processed successfully. "
            f"{errors} error(s). Total time: {elapsed:.2f}s"
        )

        msg = (
            f"Batch processing complete!\n\n"
            f"Total videos: {total}\n"
            f"Successful: {success}\n"
            f"Errors: {errors}\n\n"
            f"Total time: {elapsed:.2f} seconds\n\n"
            f"Output folder:\n{output_folder}\n\n"
        )
        if excel_path:
            msg += f"Excel report:\n{excel_path}"

        QMessageBox.information(self, "Batch Processing Complete", msg)
        self._batch_worker = None

    @Slot(str)
    def _on_batch_fatal_error(self, error_msg: str):
        """Handle a fatal error that stops the entire batch."""
        self._timer.stop()
        self._set_ui_enabled(True)

        elapsed = round(time.time() - self._start_time, 2)
        self.time_label.setText(f"Processing time: {elapsed:.2f}s (stopped)")
        self.status_label.setText(f"Batch error: {error_msg}")

        if "cancelled" in error_msg.lower():
            QMessageBox.information(self, "Cancelled", "Batch processing was cancelled by user.")
        else:
            QMessageBox.critical(self, "Batch Error", error_msg)
        self._batch_worker = None

    # ----------------------------------------------------------------
    # UI State Management
    # ----------------------------------------------------------------

    def _set_ui_enabled(self, enabled: bool):
        """Enable/disable UI controls during export."""
        self.export_btn.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        self.reset_btn.setEnabled(enabled)
        self.simple_check.setEnabled(enabled)
        self.scene_check.setEnabled(enabled)
        self.smart_check.setEnabled(enabled)
        self.hier_check.setEnabled(enabled)
        self.max_frames_spin.setEnabled(enabled)
        self.hybrid_similarity_edit.setEnabled(enabled)
        self.output_name_edit.setEnabled(enabled)
        self.output_width_spin.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.jpeg_quality_spin.setEnabled(enabled)

        # Multi-mode controls
        self.multi_input_btn.setEnabled(enabled)
        self.multi_output_btn.setEnabled(enabled)
        self.excel_browse_btn.setEnabled(enabled)
        self.mode_single_btn.setEnabled(enabled)
        self.mode_multi_btn.setEnabled(enabled)
