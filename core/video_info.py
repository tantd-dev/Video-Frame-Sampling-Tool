"""
Video metadata extraction using FFprobe.
"""

import json
import subprocess
import os
from models.video_metadata import VideoMetadata
from utils.timestamp import parse_fractional_fps


class FFprobeError(Exception):
    """Raised when FFprobe encounters an error."""
    pass


class FFprobeNotFoundError(FFprobeError):
    """Raised when FFprobe is not found on the system."""
    pass


def check_ffprobe_available() -> bool:
    """Check if FFprobe is available in PATH."""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_video_metadata(filepath: str) -> VideoMetadata:
    """
    Extract video metadata using FFprobe.

    Args:
        filepath: Path to the video file.

    Returns:
        VideoMetadata object with parsed metadata.

    Raises:
        FFprobeNotFoundError: If FFprobe is not installed.
        FFprobeError: If FFprobe fails to parse the video.
        FileNotFoundError: If the video file doesn't exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Video file not found: {filepath}")

    if not check_ffprobe_available():
        raise FFprobeNotFoundError(
            "FFmpeg/FFprobe was not found. "
            "Please install FFmpeg and add it to PATH."
        )

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        filepath,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
    except subprocess.TimeoutExpired:
        raise FFprobeError("FFprobe timed out while reading video metadata.")
    except Exception as e:
        raise FFprobeError(f"FFprobe failed: {e}")

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "Unknown error"
        raise FFprobeError(f"FFprobe returned error: {stderr}")

    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise FFprobeError("FFprobe returned invalid JSON output.")

    return _parse_probe_data(filepath, probe_data)


def _parse_probe_data(filepath: str, data: dict) -> VideoMetadata:
    """Parse FFprobe JSON output into VideoMetadata."""
    metadata = VideoMetadata()
    metadata.filepath = filepath
    metadata.filename = os.path.basename(filepath)

    # Parse format-level data
    fmt = data.get("format", {})
    metadata.size_bytes = int(fmt.get("size", 0))
    metadata.duration_seconds = float(fmt.get("duration", 0))

    # Find video and audio streams
    streams = data.get("streams", [])
    video_stream = None
    audio_stream = None

    for stream in streams:
        codec_type = stream.get("codec_type", "")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    # Parse video stream
    if video_stream:
        metadata.width = int(video_stream.get("width", 0))
        metadata.height = int(video_stream.get("height", 0))
        metadata.video_codec = video_stream.get("codec_name", "")

        # Parse FPS - handle fractional values like "30000/1001"
        fps_raw = video_stream.get("r_frame_rate", "0")
        metadata.fps_raw = fps_raw
        metadata.fps = parse_fractional_fps(fps_raw)

        # If duration not found in format, try stream
        if metadata.duration_seconds == 0:
            metadata.duration_seconds = float(video_stream.get("duration", 0))

        # Video bitrate
        bit_rate = video_stream.get("bit_rate")
        if bit_rate:
            try:
                metadata.video_bitrate = int(bit_rate)
            except (ValueError, TypeError):
                pass
        # Fallback: use format bitrate if stream bitrate not available
        if metadata.video_bitrate is None:
            fmt_bitrate = fmt.get("bit_rate")
            if fmt_bitrate:
                try:
                    metadata.video_bitrate = int(fmt_bitrate)
                except (ValueError, TypeError):
                    pass

    # Parse audio stream
    if audio_stream:
        metadata.audio_codec = audio_stream.get("codec_name", "")
        audio_br = audio_stream.get("bit_rate")
        if audio_br:
            try:
                metadata.audio_bitrate = int(audio_br)
            except (ValueError, TypeError):
                pass

    return metadata
