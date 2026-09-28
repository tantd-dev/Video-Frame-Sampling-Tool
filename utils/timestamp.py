"""
Utility functions for timestamp formatting and parsing.
"""


def seconds_to_hhmmss(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS or MM:SS format.

    Examples:
        907.0 -> "15:07"
        3661.5 -> "1:01:01"
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def seconds_to_timestamp_tag(seconds: float) -> str:
    """
    Convert seconds to a zero-padded string suitable for filenames.

    Examples:
        0.0   -> "000000"
        10.0  -> "000010"
        3661  -> "003661"
    """
    return f"{int(seconds):06d}"


def format_frame_filename(index: int, timestamp_seconds: float, extension: str = "jpg") -> str:
    """
    Generate a frame filename with index and timestamp.

    Examples:
        (1, 0.0) -> "frame_0001_000000.jpg"
        (2, 10.0) -> "frame_0002_000010.jpg"
    """
    ts_tag = seconds_to_timestamp_tag(timestamp_seconds)
    return f"frame_{index:04d}_{ts_tag}.{extension}"


def parse_fractional_fps(fps_string: str) -> float:
    """
    Parse FPS string which may be fractional.

    Examples:
        "30" -> 30.0
        "30000/1001" -> 29.97...
        "24/1" -> 24.0
    """
    if "/" in fps_string:
        parts = fps_string.split("/")
        try:
            numerator = float(parts[0])
            denominator = float(parts[1])
            if denominator == 0:
                return 0.0
            return numerator / denominator
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(fps_string)
    except ValueError:
        return 0.0
