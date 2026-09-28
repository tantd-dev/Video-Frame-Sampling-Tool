# Video Frame Sampling Tool

A desktop GUI tool for extracting frames from video files using multiple sampling strategies.

**This tool focuses on video preprocessing / frame extraction only. It does NOT integrate any LLM, VLM, Ollama, LM Studio, vLLM, or AI inference.**

## Features

### Hybrid Frame Selection Pipeline
Instead of fragmenting frames into separate strategy folders, the tool executes an integrated hybrid selection pipeline:
1. **Candidate Generation**: Extract candidate timestamps from all enabled strategies (Simple, Scene-change, Smart, Hierarchical).
2. **Candidate Pooling**: Merge all candidates into a single unified pool.
3. **Timestamp Deduplication**: Filter out candidates with nearly identical timestamps within a configurable window (preferring scene-change detections).
4. **Perceptual Similarity Filtering**: Remove visually redundant frames using perceptual hashing (`imagehash.phash`).
5. **Temporal Coverage Preservation**: Distribute frames across the timeline to avoid temporal blindspots.
6. **Scene-change Prioritization**: Boost and protect frames with strong visual change signals.
7. **Max Frame Limiting**: Apply an optional configurable maximum final frame count limit.
8. **Export**: Export only the final selected frames directly into `frames/`.

### UI Enhancements
- **Processing Time Tracking**: Real-time elapsed timer and total execution time displayed on the GUI.
- **Reset to Defaults**: One-click reset button to restore all sampling parameters and UI inputs.
- **Configurable Max Frames & Thresholds**: Direct control over output limits on the GUI.

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- **PySide6** — GUI framework
- **opencv-python** — Image processing
- **ImageHash** — Perceptual hashing for smart sampling
- **Pillow** — Image manipulation

### External Dependencies

**FFmpeg and FFprobe** must be installed and available in your system PATH.

#### Windows
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via:
```bash
winget install Gyan.FFmpeg
# or
choco install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt install ffmpeg    # Debian/Ubuntu
sudo dnf install ffmpeg    # Fedora
```

Verify installation:
```bash
ffmpeg -version
ffprobe -version
```

## Usage

```bash
cd VideoFrameSamplingTool
python main.py
```

1. Click **Browse** to select a video file (MP4, MOV, MKV, AVI, WebM, M4V)
2. Video metadata is automatically displayed
3. Enter an **output folder name**
4. Select one or more **candidate sampling strategies** and configure parameters
5. Set **Max final frames** (0 for unlimited) and **similarity threshold**
6. Set **output resolution** and **image format**
7. Click **Export Frames (Hybrid Pipeline)**
8. Watch real-time progress and total processing time
9. Click **Reset Defaults** anytime to restore default parameters

## Output Structure

```
exports/
└── <folder_name>/
    ├── metadata.json        # Comprehensive pipeline statistics & frame metadata
    └── frames/              # Only the final selected frames
        ├── frame_0001_000000.jpg
        ├── frame_0002_000015.jpg
        └── ...
```

All selected frames are stored cleanly inside `frames/`.

### Frame Naming

Frames are named with sequential index and timestamp (seconds):
```
frame_0001_000000.jpg    # Frame 1 at 0 seconds
frame_0002_000010.jpg    # Frame 2 at 10 seconds
frame_0003_000025.jpg    # Frame 3 at 25 seconds
```

### metadata.json

Contains video information, hybrid selection stats, and timing:
```json

{
  "video": {
    "filename": "example.mp4",
    "size_bytes": 1234567890,
    "duration_seconds": 907,
    "width": 3840,
    "height": 2160,
    "fps": 29.97,
    "video_codec": "hevc",
    "audio_codec": "aac"
  },
  "sampling": {
    "simple": {
      "enabled": true,
      "interval_seconds": 10,
      "frame_count": 91
    }
  }
}
```

## Project Structure

```
VideoFrameSamplingTool/
├── main.py                          # Entry point
├── config.py                        # Configuration & defaults
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── ui/
│   └── main_window.py               # PySide6 main window & export worker
│
├── core/
│   ├── video_info.py                 # FFprobe metadata extraction
│   ├── ffmpeg_runner.py              # FFmpeg subprocess runner
│   ├── simple_sampling.py            # Simple interval sampling
│   ├── scene_sampling.py             # Scene-change detection sampling
│   ├── smart_sampling.py             # Heuristic smart sampling
│   └── hierarchical_sampling.py      # Two-level hierarchical processing
│
├── models/
│   └── video_metadata.py             # Video metadata data model
│
└── utils/
    ├── filesystem.py                 # File/directory utilities
    └── timestamp.py                  # Timestamp formatting
```

## Sampling Strategies

### Simple Sampling
Extracts frames at fixed intervals (e.g., every 10 seconds). Straightforward and predictable.

### Scene-change Sampling
Uses FFmpeg's `select='gt(scene,THRESHOLD)'` filter to detect visual scene changes. This is **scene-change based sampling**, not semantic keyframe detection.

### Smart Sampling (Heuristic)
This is a **heuristic** approach — it does NOT use any AI or semantic understanding.

Pipeline:
1. Extract candidate frames at regular intervals
2. Compute perceptual hash for each candidate
3. Compare consecutive frames
4. Remove frames that are too visually similar to the previous kept frame
5. Export remaining frames

### Hierarchical Processing
Two-level extraction:
1. **Level 1 (Coarse)**: Extract frames at wide intervals for overview
2. **Analysis**: Compare coarse frames to find segments with significant visual change
3. **Level 2 (Fine)**: Extract additional frames in interesting segments only

**MVP Limitations:**
- Uses perceptual hashing for change detection, not semantic analysis
- "Interesting" segments are determined solely by visual difference between coarse frames
- May miss subtle but important changes

## Large Video Support

- Videos up to 4K resolution and 2GB+ are supported
- Video is NOT loaded into RAM — FFmpeg subprocesses handle extraction
- Output frames are resized during extraction (e.g., 3840×2160 → 1280×720)
- Aspect ratio is always maintained

## Error Handling

The tool handles:
- Missing FFmpeg/FFprobe with clear error messages
- Non-existent or corrupt video files
- Invalid parameter values
- FFmpeg command failures
- Non-writable output directories

The application will **never crash** due to these errors — all are caught and reported to the user.

## License

MIT
