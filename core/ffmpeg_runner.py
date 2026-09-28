"""
FFmpeg command runner with progress tracking.
"""

import os
import re
import subprocess
import threading
from typing import Callable, Optional


class FFmpegError(Exception):
    """Raised when FFmpeg encounters an error."""
    pass


class FFmpegNotFoundError(FFmpegError):
    """Raised when FFmpeg is not installed."""
    pass


class FFmpegRunner:
    """
    Runs FFmpeg commands as subprocesses with progress reporting.
    Designed to be used from a worker thread, not the UI thread.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the running FFmpeg process."""
        self._cancelled = True
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except OSError:
                pass

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run_command(
        self,
        cmd: list[str],
        duration_seconds: float = 0,
        progress_callback: Optional[Callable[[float], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run an FFmpeg command with optional progress tracking.

        Args:
            cmd: FFmpeg command as list of strings.
            duration_seconds: Total video duration for progress calculation.
            progress_callback: Called with progress percentage (0-100).
            status_callback: Called with status text updates.

        Returns:
            CompletedProcess with stdout/stderr.

        Raises:
            FFmpegError: If the command fails.
            FFmpegNotFoundError: If FFmpeg is not found.
        """
        self._cancelled = False

        # Add progress flag to FFmpeg command if not present
        if "-progress" not in cmd and duration_seconds > 0:
            cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
        except FileNotFoundError:
            raise FFmpegNotFoundError(
                "FFmpeg was not found. Please install FFmpeg and add it to PATH."
            )

        stdout_lines = []
        stderr_data = ""

        # Read progress from stdout
        if self._process.stdout:
            for line in self._process.stdout:
                if self._cancelled:
                    break
                stdout_lines.append(line)

                # Parse progress info
                if duration_seconds > 0 and progress_callback:
                    match = re.match(r"out_time_us=(\d+)", line.strip())
                    if match:
                        current_us = int(match.group(1))
                        current_seconds = current_us / 1_000_000
                        progress = min(100.0, (current_seconds / duration_seconds) * 100)
                        progress_callback(progress)

        # Wait for process to finish
        self._process.wait()

        if self._process.stderr:
            stderr_data = self._process.stderr.read()

        returncode = self._process.returncode

        if self._cancelled:
            raise FFmpegError("Operation cancelled by user.")

        if returncode != 0 and returncode is not None:
            # Extract useful error info from stderr
            error_msg = stderr_data.strip().split('\n')[-1] if stderr_data.strip() else "Unknown error"
            raise FFmpegError(f"FFmpeg failed (exit code {returncode}): {error_msg}")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=returncode or 0,
            stdout="".join(stdout_lines),
            stderr=stderr_data,
        )

    def extract_frames_at_times(
        self,
        input_path: str,
        output_dir: str,
        timestamps: list[float],
        output_width: int = 1280,
        output_height: int = -2,
        image_format: str = "jpg",
        jpeg_quality: int = 85,
        progress_callback: Optional[Callable[[float], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        filename_prefix: str = "frame",
    ) -> list[str]:
        """
        Extract specific frames at given timestamps using FFmpeg.

        Uses seeking for each frame to avoid loading entire video.
        This is slower for many frames but memory-efficient.

        Returns list of output file paths.
        """
        from utils.timestamp import seconds_to_timestamp_tag

        output_files = []
        total = len(timestamps)

        # Build scale filter
        if output_height == -2 or output_height <= 0:
            scale_filter = f"scale={output_width}:-2"
        else:
            scale_filter = f"scale={output_width}:{output_height}"

        # Determine output format options
        format_ext = image_format.lower()
        if format_ext == "jpeg":
            format_ext = "jpg"

        for i, ts in enumerate(timestamps):
            if self._cancelled:
                raise FFmpegError("Operation cancelled by user.")

            ts_tag = seconds_to_timestamp_tag(ts)
            filename = f"{filename_prefix}_{i + 1:04d}_{ts_tag}.{format_ext}"
            output_path = os.path.join(output_dir, filename)

            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-ss", str(ts),
                "-i", input_path,
                "-vframes", "1",
                "-vf", scale_filter,
            ]

            # Format-specific options
            if format_ext in ("jpg", "jpeg"):
                cmd.extend(["-q:v", str(max(1, min(31, int(31 - (jpeg_quality / 100) * 30))))])
            elif format_ext == "png":
                cmd.extend(["-compression_level", "6"])

            cmd.append(output_path)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
                if result.returncode == 0:
                    output_files.append(output_path)
            except subprocess.TimeoutExpired:
                if status_callback:
                    status_callback(f"Warning: Frame extraction timed out at {ts}s")
            except FileNotFoundError:
                raise FFmpegNotFoundError(
                    "FFmpeg was not found. Please install FFmpeg and add it to PATH."
                )

            # Report progress
            if progress_callback:
                progress_callback(((i + 1) / total) * 100)

        return output_files
