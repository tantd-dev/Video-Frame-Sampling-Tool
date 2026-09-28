"""
Utility functions for filesystem operations.
"""

import os
import shutil
from pathlib import Path


def ensure_directory(path: str) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_export_base_dir(project_root: str, export_base: str, folder_name: str) -> Path:
    """
    Build the export base directory path.

    Returns: Path like <project_root>/exports/<folder_name>/
    """
    return Path(project_root) / export_base / folder_name


def is_directory_writable(path: str) -> bool:
    """Check if a directory is writable."""
    try:
        test_file = Path(path) / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


def sanitize_folder_name(name: str) -> str:
    """
    Sanitize a folder name by removing/replacing invalid characters.
    """
    # Replace common problematic characters
    invalid_chars = '<>:"/\\|?*'
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip().strip('.')
    # Default name if empty
    if not sanitized:
        sanitized = "output"
    return sanitized
