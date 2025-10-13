"""
Data models for face swapping application.
"""
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ProcessingStatus(Enum):
    """Status of processing operation."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class FileType(Enum):
    """Supported file types."""
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class FaceDetectionResult:
    """Result of face detection."""
    faces: List[Any]  # FaceAnalysis results
    success: bool
    error_message: Optional[str] = None


@dataclass
class SwapResult:
    """Result of face swap operation."""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    original_name: Optional[str] = None
    file_type: FileType = FileType.IMAGE


@dataclass
class ProcessingTask:
    """A single file processing task."""
    id: str
    source_path: str
    target_path: str
    target_filename: str
    source_face_cache: Any
    status: ProcessingStatus = ProcessingStatus.IDLE

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class FileProgress:
    """Individual file processing progress."""
    filename: str
    progress_percentage: float = 0.0
    status: str = "pending"  # pending, processing, completed, error
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None
    file_type: str = "unknown"
    start_time: Optional[float] = None
    estimated_time: Optional[float] = None

    @property
    def is_image(self) -> bool:
        """Check if file is an image."""
        return self.file_type in ["image", "jpg", "jpeg", "png"]

    @property
    def is_video(self) -> bool:
        """Check if file is a video/GIF."""
        return self.file_type in ["video", "gif", "mp4", "avi", "mov", "mkv"]

    @property
    def progress_text(self) -> str:
        """Get progress text for display."""
        if self.status == "pending":
            return "Chờ xử lý..."
        elif self.status == "processing":
            if self.is_image:
                return ".1f"
            elif self.is_video and self.total_frames:
                if self.current_frame is not None:
                    return ".1f"
                else:
                    return ".1f"
        elif self.status == "completed":
            return "Hoàn thành"
        elif self.status == "error":
            return "Lỗi"
        return "Đang xử lý..."

@dataclass
class ProcessingProgress:
    """Progress tracking for processing operations."""
    total_files: int
    completed: int
    current_file: Optional[str] = None
    queue: List[str] = None
    start_time: Optional[float] = None
    speed: float = 0.0
    eta_seconds: float = 0.0
    results: List[SwapResult] = None
    video_progress: Optional[Dict[str, Any]] = None
    file_progress: Optional[Dict[str, FileProgress]] = None

    def __post_init__(self):
        if self.queue is None:
            self.queue = []
        if self.results is None:
            self.results = []
        if self.video_progress is None:
            self.video_progress = {
                "current_video_total_frames": 0,
                "current_video_processed_frames": 0,
                "current_video_progress_percentage": 0.0
            }
        if self.file_progress is None:
            self.file_progress = {}

    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        return (self.completed / self.total_files * 100) if self.total_files > 0 else 0.0

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.completed >= self.total_files


@dataclass
class Config:
    """Application configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    output_directory: str = "output"
    temp_directory: str = "/tmp"
    max_batch_size: int = 8
    face_detection_size: tuple = (640, 640)

    # GPU/CPU settings
    cuda_visible_devices: str = "0"
    omp_num_threads: int = 4
    mkl_num_threads: int = 4
