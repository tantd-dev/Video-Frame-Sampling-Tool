"""
Video metadata data model.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class VideoMetadata:
    """Data class holding parsed video metadata from FFprobe."""

    filename: str = ""
    filepath: str = ""
    size_bytes: int = 0
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fps_raw: str = ""  # e.g. "30000/1001"
    video_codec: str = ""
    video_bitrate: Optional[int] = None  # bits per second
    audio_codec: str = ""
    audio_bitrate: Optional[int] = None  # bits per second

    @property
    def resolution(self) -> str:
        """Return resolution as 'WxH' string."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "N/A"

    @property
    def duration_formatted(self) -> str:
        """Return duration as HH:MM:SS or MM:SS."""
        total_seconds = int(self.duration_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def size_formatted(self) -> str:
        """Return human-readable file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    @property
    def fps_display(self) -> str:
        """Return FPS as a display string, handling fractional FPS."""
        if self.fps == int(self.fps):
            return str(int(self.fps))
        return f"{self.fps:.2f}"

    @property
    def video_bitrate_formatted(self) -> str:
        """Return human-readable video bitrate."""
        if self.video_bitrate is None:
            return "N/A"
        kbps = self.video_bitrate / 1000
        if kbps >= 1000:
            return f"{kbps / 1000:.1f} Mbps"
        return f"{kbps:.0f} Kbps"

    @property
    def audio_bitrate_formatted(self) -> str:
        """Return human-readable audio bitrate."""
        if self.audio_bitrate is None:
            return "N/A"
        kbps = self.audio_bitrate / 1000
        return f"{kbps:.0f} Kbps"

    def to_export_dict(self) -> dict:
        """Return a dict suitable for metadata.json export."""
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "duration_seconds": round(self.duration_seconds, 2),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "video_codec": self.video_codec,
            "video_bitrate": self.video_bitrate,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
        }
