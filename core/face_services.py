"""
Face detection and swapping services.
"""
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from .models import FaceDetectionResult, SwapResult, FileType, Config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Service for face detection using InsightFace."""

    def __init__(self, config: Config):
        self.config = config
        self.app: Optional[FaceAnalysis] = None

    def initialize(self):
        """Initialize the face detection model."""
        try:
            logger.info("Loading face detection model...")
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0, det_size=self.config.face_detection_size)
            logger.info("Face detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load face detection model: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect faces in an image."""
        try:
            faces = self.app.get(image) if self.app else []
            success = len(faces) > 0
            return FaceDetectionResult(
                faces=faces,
                success=success,
                error_message=None if success else "No face detected in image"
            )
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return FaceDetectionResult(
                faces=[],
                success=False,
                error_message=str(e)
            )


class FaceSwapper:
    """Service for face swapping operations."""

    def __init__(self, config: Config):
        self.config = config
        self.swapper = None

    def initialize(self):
        """Initialize the face swapper model."""
        try:
            logger.info("Loading face swapper model...")
            self.swapper = get_model("models/inswapper_128.onnx", download=False)
            logger.info("Face swapper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load face swapper model: {e}")
            raise

    def swap_faces(self, source_face, target_image: np.ndarray, target_face) -> Optional[np.ndarray]:
        """Perform face swap between source and target."""
        try:
            # Validate inputs
            if target_image is None or target_image.size == 0:
                logger.warning("Target image is empty or None")
                return None

            if target_face is None:
                logger.warning("Target face is None")
                return None

            if source_face is None:
                logger.warning("Source face is None")
                return None

            result = self.swapper.get(target_image, target_face, source_face, paste_back=True)

            # Validate output
            if result is None or result.size == 0:
                logger.warning("Face swap returned empty or None result")
                return None

            return result

        except Exception as e:
            logger.error(f"Face swap error: {e}")
            return None


class FaceServices:
    """Combined face detection and swapping services."""

    def __init__(self, config: Config):
        self.config = config
        self.detector = FaceDetector(config)
        self.swapper = FaceSwapper(config)

    def initialize(self):
        """Initialize all face services."""
        self.detector.initialize()
        self.swapper.initialize()

    def swap_face(self, source_img: np.ndarray, target_img: np.ndarray) -> SwapResult:
        """Swap face from source to target image."""
        try:
            # Detect faces
            source_result = self.detector.detect_faces(source_img)
            target_result = self.detector.detect_faces(target_img)

            if not source_result.success:
                return SwapResult(
                    success=False,
                    error_message=source_result.error_message or "No source face detected"
                )

            if not target_result.success:
                return SwapResult(
                    success=False,
                    error_message=target_result.error_message or "No target face detected"
                )

            # Get first face for swapping
            source_face = source_result.faces[0]
            target_face = target_result.faces[0]

            # Perform swap
            swapped = self.swapper.swap_faces(source_face, target_img, target_face)

            return SwapResult(
                success=True,
                output_path=None,  # Will be set by caller
                file_type=FileType.IMAGE
            )

        except Exception as e:
            logger.error(f"Face swap operation error: {e}")
            return SwapResult(
                success=False,
                error_message=f"Swap error: {str(e)}"
            )
