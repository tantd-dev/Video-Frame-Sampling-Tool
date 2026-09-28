"""
Application-wide configuration and default values.
"""

# Application info
APP_NAME = "Video Frame Sampling Tool"
APP_VERSION = "2.0.0"

# Supported video extensions
SUPPORTED_VIDEO_EXTENSIONS = [
    "mp4", "mov", "mkv", "avi", "webm", "m4v"
]

# Default sampling parameters
DEFAULT_SIMPLE_INTERVAL = 10          # seconds
DEFAULT_SCENE_THRESHOLD = 0.30
DEFAULT_SMART_CANDIDATE_INTERVAL = 10  # seconds
DEFAULT_SMART_SIMILARITY_THRESHOLD = 0.90
DEFAULT_HIERARCHICAL_COARSE_INTERVAL = 30  # seconds
DEFAULT_HIERARCHICAL_FINE_INTERVAL = 5     # seconds
DEFAULT_HIERARCHICAL_SCENE_THRESHOLD = 0.30

# Hybrid pipeline defaults
DEFAULT_HYBRID_MAX_FRAMES = 0              # 0 = unlimited
DEFAULT_HYBRID_SIMILARITY_THRESHOLD = 0.90
DEFAULT_HYBRID_DEDUP_GAP = 0.5             # seconds

# Default output settings
DEFAULT_OUTPUT_WIDTH = 1280
DEFAULT_OUTPUT_HEIGHT_AUTO = True  # maintain aspect ratio
DEFAULT_IMAGE_FORMAT = "JPEG"
DEFAULT_JPEG_QUALITY = 85
SUPPORTED_IMAGE_FORMATS = ["JPEG", "PNG", "BMP", "TIFF"]

# Export directory
EXPORT_BASE_DIR = "exports"
