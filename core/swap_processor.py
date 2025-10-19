"""
Main processing logic for face swapping operations.
"""
import logging
import uuid
import os
import cv2
import numpy as np
from typing import List, Optional, Callable, Any
import threading
import time
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from .models import (
    SwapResult, ProcessingProgress, ProcessingStatus, FileType,
    Config, ProcessingTask
)
from .face_services import FaceServices
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles file input/output and validation."""

    def __init__(self, config: Config):
        self.config = config

    def save_uploaded_file(self, file, filename: str, max_size_mb: float = 20000.0) -> str:
        """Save uploaded file to temporary location with size validation and streaming for large files."""
        # Validate file size before starting upload
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Seek back to beginning

        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(f"File size exceeds maximum limit of {max_size_mb}MB")

        file_uuid = str(uuid.uuid4())
        file_path = f"{self.config.temp_directory}/{file_uuid}_{filename}"

        # Use streaming approach for better memory efficiency with large files
        try:
            file_size_gb = file_size / (1024 * 1024 * 1024)
            if file_size > 100 * 1024 * 1024:  # Log progress for files > 100MB
                if file_size_gb >= 1:
                    logger.info(f"Saving large file ({file_size_gb:.1f}GB): {filename}")
                else:
                    logger.info(f"Saving large file ({file_size / 1024 / 1024:.1f}MB): {filename}")

            # For large files, always use chunked reading to prevent memory issues
            if file_size > 10 * 1024 * 1024:  # >10MB files use optimized chunked reading
                with open(file_path, 'wb') as out_file:
                    chunk_size = 128 * 1024  # 128KB chunks for better throughput
                    bytes_read = 0
                    last_progress_log = 0

                    while bytes_read < file_size:
                        remaining = file_size - bytes_read
                        read_size = min(chunk_size, remaining)
                        chunk = file.read(read_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        bytes_read += len(chunk)

                        # Log progress for very large files every 10%
                        if file_size > 1024 * 1024 * 1024:  # >1GB
                            progress_percentage = (bytes_read / file_size) * 100
                            progress_step = int(progress_percentage // 10) * 10
                            if progress_step > last_progress_log:
                                logger.info(f"Upload progress for {filename}: {progress_step}% complete")
                                last_progress_log = progress_step
            else:
                # For smaller files, use the standard method
                file.save(file_path)

            # Verify file was saved correctly
            if os.path.exists(file_path):
                saved_size = os.path.getsize(file_path)
                if saved_size != file_size:
                    logger.warning(f"File size mismatch: expected {file_size}, got {saved_size}")
                else:
                    if file_size_gb >= 1:
                        logger.info(f"File saved successfully: {file_path} ({file_size_gb:.1f}GB)")
                    else:
                        logger.info(f"File saved successfully: {file_path} ({saved_size / 1024 / 1024:.1f}MB)")
            else:
                raise IOError(f"Failed to save file: {file_path}")

            return file_path

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            raise e

    async def save_uploaded_file_async(self, file_stream, filename: str, file_size: int, max_size_mb: float = 500.0, progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
        """Asynchronously save uploaded file with progress tracking for better streaming uploads."""
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(f"File size exceeds maximum limit of {max_size_mb}MB")

        file_uuid = str(uuid.uuid4())
        file_path = f"{self.config.temp_directory}/{file_uuid}_{filename}"

        try:
            # Use async file writing for better performance
            async with aiofiles.open(file_path, 'wb') as out_file:
                bytes_written = 0
                chunk_size = 64 * 1024  # 64KB chunks for better throughput

                while bytes_written < file_size:
                    remaining = min(chunk_size, file_size - bytes_written)
                    chunk = file_stream.read(remaining)
                    if not chunk:
                        break

                    await out_file.write(chunk)
                    bytes_written += len(chunk)

                    # Report progress if callback provided
                    if progress_callback:
                        progress_callback(bytes_written, file_size)

            # Verify file was saved correctly
            if os.path.exists(file_path):
                saved_size = os.path.getsize(file_path)
                if saved_size != file_size:
                    logger.warning(f"File size mismatch: expected {file_size}, got {saved_size}")
                else:
                    logger.info(f"File saved asynchronously: {file_path} ({saved_size / 1024 / 1024:.1f}MB)")
            else:
                raise IOError(f"Failed to save file asynchronously: {file_path}")

            return file_path

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            raise e

    def validate_file(self, filename: str) -> bool:
        """Basic file validation."""
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.mov', '.mkv'}
        return any(filename.lower().endswith(ext) for ext in allowed_extensions)

    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files."""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {path}: {e}")


class SwapProcessor:
    """Main processor for face swapping operations."""

    def __init__(self, config: Config, face_services: FaceServices):
        self.config = config
        self.face_services = face_services
        self.video_processor = VideoProcessor(config.max_batch_size)
        self.file_processor = FileProcessor(config)
        self.progress_callback: Optional[Callable[[str, Any], None]] = None

    def set_progress_callback(self, callback: Callable[[str, Any], None]):
        """Set callback for updating individual file progress."""
        self.progress_callback = callback

    def process_image(
        self,
        source_face_cache,
        target_path: str,
        target_filename: str,
        swap_all_faces: bool = False
    ) -> SwapResult:
        """Process a single image file."""
        try:
            logger.info(f"Processing image: {target_filename}")

            target_img = cv2.imread(target_path)
            if target_img is None:
                logger.warning(f"Cannot read image {target_filename}, copying as original")
                # Copy original file to output if cannot process
                import shutil
                output_uuid = str(uuid.uuid4())
                output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                shutil.copy2(target_path, output_path)
                return SwapResult(
                    success=True,
                    output_path=f"/view/{output_path}",
                    original_name=target_filename,
                    file_type=FileType.IMAGE
                )

            # Detect faces in target image
            target_result = self.face_services.detector.detect_faces(target_img)
            if not target_result.success:
                logger.warning(f"No faces detected in {target_filename}, copying as original")
                # Copy original file to output if no faces detected
                import shutil
                output_uuid = str(uuid.uuid4())
                output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                shutil.copy2(target_path, output_path)
                return SwapResult(
                    success=True,
                    output_path=f"/view/{output_path}",
                    original_name=target_filename,
                    file_type=FileType.IMAGE
                )

            # Perform swap using the cached source face
            try:
                if swap_all_faces and len(target_result.faces) > 1:
                    # Swap all faces in the image
                    swapped = target_img.copy()
                    faces_swapped = 0

                    for target_face in target_result.faces:
                        swap_result = self.face_services.swapper.swap_faces(
                            source_face_cache,
                            swapped,
                            target_face
                        )
                        if swap_result is not None and swap_result.size > 0:
                            swapped = swap_result
                            faces_swapped += 1

                    logger.info(f"[SWAP] Image: {target_filename} - Swapped {faces_swapped}/{len(target_result.faces)} faces")

                    if faces_swapped == 0:
                        logger.warning(f"Face swap failed for all faces in {target_filename}, copying as original")
                        # Copy original file to output on swap failure
                        import shutil
                        output_uuid = str(uuid.uuid4())
                        output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                        shutil.copy2(target_path, output_path)
                        return SwapResult(
                            success=True,
                            output_path=f"/view/{output_path}",
                            original_name=target_filename,
                            file_type=FileType.IMAGE
                        )
                else:
                    # Swap only first face (legacy behavior)
                    swapped = self.face_services.swapper.swap_faces(
                        source_face_cache,
                        target_img,
                        target_result.faces[0]  # Use first detected face
                    )

                    # Log successful swap for image
                    logger.info(f"[SWAP] Image: {target_filename}")

                    if swapped is None or swapped.size == 0:
                        logger.warning(f"Face swap failed for {target_filename}, copying as original")
                        # Copy original file to output on swap failure
                        import shutil
                        output_uuid = str(uuid.uuid4())
                        output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                        shutil.copy2(target_path, output_path)
                        return SwapResult(
                            success=True,
                            output_path=f"/view/{output_path}",
                            original_name=target_filename,
                            file_type=FileType.IMAGE
                        )

                # Save result
                output_uuid = str(uuid.uuid4())
                output_path = f"{self.config.output_directory}/{output_uuid}.jpg"
                success = cv2.imwrite(output_path, swapped)

                if not success:
                    logger.warning(f"Failed to save swapped image for {target_filename}, copying as original")
                    # Copy original file to output on save failure
                    import shutil
                    output_uuid = str(uuid.uuid4())
                    output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                    shutil.copy2(target_path, output_path)
                    return SwapResult(
                        success=True,
                        output_path=f"/view/{output_path}",
                        original_name=target_filename,
                        file_type=FileType.IMAGE
                    )

                # Verify file was created
                if not os.path.exists(output_path):
                    logger.warning(f"Output file not created for {target_filename}, copying as original")
                    # Copy original file to output on file creation failure
                    import shutil
                    output_uuid = str(uuid.uuid4())
                    output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                    shutil.copy2(target_path, output_path)
                    return SwapResult(
                        success=True,
                        output_path=f"/view/{output_path}",
                        original_name=target_filename,
                        file_type=FileType.IMAGE
                    )

                logger.info(f"Image saved to: {output_path}")

            except Exception as e:
                logger.error(f"Face swap error for {target_filename}: {e}, copying as original")
                # Copy original file to output on any exception during swap
                import shutil
                output_uuid = str(uuid.uuid4())
                output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                shutil.copy2(target_path, output_path)
                return SwapResult(
                    success=True,
                    output_path=f"/view/{output_path}",
                    original_name=target_filename,
                    file_type=FileType.IMAGE
                )

            return SwapResult(
                success=True,
                output_path=f"/view/{output_path}",
                original_name=target_filename,
                file_type=FileType.IMAGE
            )

        except Exception as e:
            logger.error(f"Image processing error for {target_filename}: {e}, copying as original")
            # Copy original file to output on any processing exception
            try:
                import shutil
                output_uuid = str(uuid.uuid4())
                output_path = f"{self.config.output_directory}/{output_uuid}_{target_filename}"
                shutil.copy2(target_path, output_path)
                return SwapResult(
                    success=True,
                    output_path=f"/view/{output_path}",
                    original_name=target_filename,
                    file_type=FileType.IMAGE
                )
            except Exception as copy_e:
                logger.error(f"Failed to copy original file for {target_filename}: {copy_e}")
                return SwapResult(
                    success=False,
                    error_message=f"Processing failed and could not copy original: {str(e)}",
                    original_name=target_filename
                )

    def process_video(
        self,
        source_face_cache,
        target_path: str,
        target_filename: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SwapResult:
        """Process a single video file (legacy method - use process_video_with_audio instead)."""
        return self.process_video_with_audio(source_face_cache, target_path, target_filename, progress_callback)

    def process_video_with_audio(
        self,
        source_face_cache,
        target_path: str,
        target_filename: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancellation_callback: Optional[Callable[[], bool]] = None,
        swap_all_faces: bool = False
    ) -> SwapResult:
        """Process a single video file with audio preservation."""
        try:
            logger.info(f"Processing video with audio: {target_filename}")

            # Get total frames for logging
            total_frames_for_logging = None
            try:
                _, _, total_frames_for_logging, _ = self.video_processor.get_video_info(target_path)
            except Exception as e:
                logger.debug(f"Could not get frame count for logging: {e}")

            output_uuid = str(uuid.uuid4())

            # Special handling for GIF files - they have no audio
            if target_filename.lower().endswith('.gif'):
                logger.info("GIF file detected, processing without audio preservation")
                output_path = f"{self.config.output_directory}/{output_uuid}.gif"

                # Track frame number for logging
                frame_index = [0]

                # Define swap callback for GIF processing
                def swap_callback(frame):
                    # Detect faces in current frame
                    frame_result = self.face_services.detector.detect_faces(frame)
                    if not frame_result.success or not frame_result.faces:
                        logger.debug("No faces detected in frame, skipping swap")
                        return frame

                    try:
                        if swap_all_faces and len(frame_result.faces) > 1:
                            # Swap all faces in the frame
                            result_frame = frame.copy()
                            faces_swapped = 0

                            for target_face in frame_result.faces:
                                swap_result = self.face_services.swapper.swap_faces(
                                    source_face_cache,
                                    result_frame,
                                    target_face
                                )
                                if swap_result is not None and swap_result.size > 0:
                                    result_frame = swap_result
                                    faces_swapped += 1

                            frame_index[0] += 1
                            if faces_swapped > 0:
                                # Log successful swap for GIF frame
                                progress_text = f"Frame {frame_index[0]}"
                                if total_frames_for_logging:
                                    progress_text += f"/{total_frames_for_logging}"
                                logger.info(f"[SWAP] GIF: {target_filename} - {progress_text} - Swapped {faces_swapped} faces")
                                return result_frame
                            else:
                                logger.debug("Face swap failed for all faces in frame, using original frame")
                                return frame
                        else:
                            # Swap only first face (legacy behavior)
                            result = self.face_services.swapper.swap_faces(
                                source_face_cache,
                                frame,
                                frame_result.faces[0]  # Use first detected face
                            )

                            frame_index[0] += 1
                            if result is not None and result.size > 0:
                                # Log successful swap for GIF frame
                                progress_text = f"Frame {frame_index[0]}"
                                if total_frames_for_logging:
                                    progress_text += f"/{total_frames_for_logging}"
                                logger.info(f"[SWAP] GIF: {target_filename} - {progress_text}")
                                return result
                            else:
                                logger.debug("Face swap returned empty result, using original frame")
                                return frame

                    except Exception as e:
                        logger.debug(f"Frame swap error, using original frame: {e}")
                        return frame

                success = self.video_processor.process_video_to_gif(
                    target_path,
                    output_path,
                    swap_callback,
                    progress_callback,
                    cancellation_callback
                )
            else:
                # Regular video with audio preservation
                output_path = f"{self.config.output_directory}/{output_uuid}.mp4"

                # Track frame number for logging
                frame_index = [0]

                # Define swap callback for video processing
                def swap_callback(frame):
                    # Detect faces in current frame
                    frame_result = self.face_services.detector.detect_faces(frame)
                    if not frame_result.success or not frame_result.faces:
                        logger.debug("No faces detected in frame, skipping swap")
                        return frame

                    try:
                        if swap_all_faces and len(frame_result.faces) > 1:
                            # Swap all faces in the frame
                            result_frame = frame.copy()
                            faces_swapped = 0

                            for target_face in frame_result.faces:
                                swap_result = self.face_services.swapper.swap_faces(
                                    source_face_cache,
                                    result_frame,
                                    target_face
                                )
                                if swap_result is not None and swap_result.size > 0:
                                    result_frame = swap_result
                                    faces_swapped += 1

                            frame_index[0] += 1
                            if faces_swapped > 0:
                                # Log successful swap for video frame
                                progress_text = f"Frame {frame_index[0]}"
                                if total_frames_for_logging:
                                    progress_text += f"/{total_frames_for_logging}"
                                logger.info(f"[SWAP] Video: {target_filename} - {progress_text} - Swapped {faces_swapped} faces")
                                return result_frame
                            else:
                                logger.debug("Face swap failed for all faces in frame, using original frame")
                                return frame
                        else:
                            # Swap only first face (legacy behavior)
                            result = self.face_services.swapper.swap_faces(
                                source_face_cache,
                                frame,
                                frame_result.faces[0]  # Use first detected face
                            )

                            frame_index[0] += 1
                            if result is not None and result.size > 0:
                                # Log successful swap for video frame
                                progress_text = f"Frame {frame_index[0]}"
                                if total_frames_for_logging:
                                    progress_text += f"/{total_frames_for_logging}"
                                logger.info(f"[SWAP] Video: {target_filename} - {progress_text}")
                                return result
                            else:
                                logger.debug("Face swap returned empty result, using original frame")
                                return frame

                    except Exception as e:
                        logger.debug(f"Frame swap error, using original frame: {e}")
                        return frame

                success = self.video_processor.process_video_with_audio(
                    target_path,
                    output_path,
                    swap_callback,
                    progress_callback,
                    cancellation_callback
                )

            if success:
                return SwapResult(
                    success=True,
                    output_path=f"/view/{output_path}",
                    original_name=target_filename,
                    file_type=FileType.VIDEO
                )
            else:
                return SwapResult(
                    success=False,
                    error_message="Video processing failed",
                    original_name=target_filename
                )

        except Exception as e:
            logger.error(f"Video processing error for {target_filename}: {e}")
            return SwapResult(
                success=False,
                error_message=str(e),
                original_name=target_filename
            )

    def process_file(
        self,
        source_face_cache,
        target_path: str,
        target_filename: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SwapResult:
        """Process a single file (image or video)."""
        # First check file extension to determine type
        if target_filename.lower().endswith(('.gif', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
            # Video/GIF files
            return self.process_video(source_face_cache, target_path, target_filename, progress_callback)
        else:
            # Try to read as image
            target_img = cv2.imread(target_path)
            if target_img is not None:
                return self.process_image(source_face_cache, target_path, target_filename)
            else:
                return SwapResult(
                    success=False,
                    error_message=f"Cannot read file as image: {target_filename}",
                    original_name=target_filename
                )

    def process_batch(
        self,
        source_img_path: str,
        target_paths: List[str],
        target_filenames: List[str],
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> List[SwapResult]:
        """Process multiple target files with progress tracking."""
        try:
            # Load and cache source face
            source_img = cv2.imread(source_img_path)
            if source_img is None:
                return [SwapResult(
                    success=False,
                    error_message="Cannot read source image"
                ) for _ in target_filenames]

            source_result = self.swap_processor.face_services.detector.detect_faces(source_img)
            if not source_result.success:
                return [SwapResult(
                    success=False,
                    error_message=source_result.error_message or "No source face detected"
                ) for _ in target_filenames]

            source_face_cache = source_result.faces[0]
            results = []

            # Process each target
            for i, (target_path, target_filename) in enumerate(zip(target_paths, target_filenames)):
                logger.info(f"Processing {i+1}/{len(target_paths)}: {target_filename}")

                def video_progress_update(processed, total):
                    if progress_callback:
                        # Update video-specific progress
                        progress_callback(None)  # Will be handled in batch processor

                result = self.process_file(
                    source_face_cache,
                    target_path,
                    target_filename,
                    video_progress_update
                )
                results.append(result)

                if progress_callback:
                    progress = ProcessingProgress(
                        total_files=len(target_paths),
                        completed=i+1,
                        current_file=target_filename,
                        results=results
                    )
                    progress_callback(progress)

            return results

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [SwapResult(
                success=False,
                error_message=f"Processing error: {str(e)}"
            ) for _ in target_filenames]


class BatchProcessor:
    """Handles batch processing with background threading."""

    def __init__(self, config: Config, swap_processor: SwapProcessor):
        self.config = config
        self.swap_processor = swap_processor
        self.processing_progress: Optional[ProcessingProgress] = None
        self.cancel_flag = False
        self.processing_thread: Optional[threading.Thread] = None
        self.state_file_path = f"{config.output_directory}/processing_state.json"

        # Batch queue for sequential processing
        self.batch_queue: List[Dict[str, Any]] = []
        self.queue_lock = threading.Lock()

        # Load persisted state on initialization
        self.load_processing_state()

    def load_processing_state(self):
        """Load processing state from disk if it exists."""
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, 'r') as f:
                    state = json.load(f)

                # Restore state
                self.cancel_flag = state.get('cancel_flag', False)

                progress_data = state.get('progress', {})
                if progress_data:
                    from .models import ProcessingProgress

                    self.processing_progress = ProcessingProgress(
                        total_files=progress_data.get('total_files', 0),
                        completed=progress_data.get('completed', 0),
                        progress_percentage=progress_data.get('progress_percentage', 0.0),
                        current_file=progress_data.get('current_file'),
                        start_time=progress_data.get('start_time'),
                        queue=progress_data.get('queue', []),
                        results=[],
                    )

                    # Restore file progress
                    if 'file_progress' in state and state['file_progress']:
                        from .models import FileProgress
                        self.processing_progress.file_progress = {}
                        for filename, file_prog_data in state['file_progress'].items():
                            file_prog = FileProgress(
                                filename=file_prog_data.get('filename', filename),
                                status=file_prog_data.get('status', 'pending'),
                                file_type=file_prog_data.get('file_type', 'unknown'),
                                progress_percentage=file_prog_data.get('progress_percentage', 0.0),
                                current_frame=file_prog_data.get('current_frame'),
                                total_frames=file_prog_data.get('total_frames'),
                            )
                            file_prog.start_time = file_prog_data.get('start_time')
                            if 'estimated_time' in file_prog_data:
                                file_prog.estimated_time = file_prog_data['estimated_time']
                            if 'progress_text' in file_prog_data:
                                file_prog.progress_text = file_prog_data['progress_text']

                            self.processing_progress.file_progress[filename] = file_prog

                    # Restore results
                    if 'results' in state and state['results']:
                        self.processing_progress.results = []
                        from .models import SwapResult, FileType
                        for result_data in state['results']:
                            result = SwapResult(
                                success=result_data.get('success', False),
                                error_message=result_data.get('error_message'),
                                original_name=result_data.get('original_name'),
                                output_path=result_data.get('output_path'),
                            )
                            if result_data.get('file_type'):
                                result.file_type = FileType(result_data['file_type'])
                            self.processing_progress.results.append(result)

                    # Restore cancelled files
                    cancelled_files = state.get('cancelled_files', [])
                    self.processing_progress.cancelled_files = cancelled_files

                logger.info("Processing state loaded from disk")
                return True

        except Exception as e:
            logger.error(f"Error loading processing state: {e}")
            # Clean up corrupted state file
            try:
                if os.path.exists(self.state_file_path):
                    os.remove(self.state_file_path)
                    logger.info("Removed corrupted state file")
            except:
                pass

        return False

    def save_processing_state(self):
        """Save current processing state to disk."""
        try:
            if self.processing_progress:
                # Create serializable state
                state = {
                    'cancel_flag': self.cancel_flag,
                    'progress': {
                        'total_files': self.processing_progress.total_files,
                        'completed': self.processing_progress.completed,
                        'progress_percentage': self.processing_progress.progress_percentage,
                        'current_file': self.processing_progress.current_file,
                        'start_time': self.processing_progress.start_time,
                        'queue': self.processing_progress.queue or [],
                    },
                    'file_progress': {},
                    'results': [],
                    'cancelled_files': getattr(self.processing_progress, 'cancelled_files', [])
                }

                # Serialize file progress
                if self.processing_progress.file_progress:
                    for filename, file_prog in self.processing_progress.file_progress.items():
                        state['file_progress'][filename] = {
                            'filename': file_prog.filename,
                            'status': file_prog.status,
                            'file_type': file_prog.file_type,
                            'progress_percentage': file_prog.progress_percentage,
                            'current_frame': file_prog.current_frame,
                            'total_frames': file_prog.total_frames,
                            'start_time': file_prog.start_time,
                            'estimated_time': getattr(file_prog, 'estimated_time', None),
                            'progress_text': getattr(file_prog, 'progress_text', ''),
                        }

                # Serialize results
                if self.processing_progress.results:
                    for result in self.processing_progress.results:
                        state['results'].append({
                            'success': result.success,
                            'error_message': result.error_message,
                            'original_name': result.original_name,
                            'file_type': result.file_type.value if hasattr(result.file_type, 'value') else None,
                            'output_path': result.output_path,
                        })

                # Save to file
                with open(self.state_file_path, 'w') as f:
                    json.dump(state, f, indent=2, default=str)
                logger.debug("Processing state saved")

        except Exception as e:
            logger.error(f"Error saving processing state: {e}")

    def clear_processing_state(self):
        """Clear persisted processing state."""
        try:
            if os.path.exists(self.state_file_path):
                os.remove(self.state_file_path)
                logger.debug("Processing state cleared")
        except Exception as e:
            logger.error(f"Error clearing processing state: {e}")

    def is_cancelled(self) -> bool:
        """Check if processing has been cancelled."""
        return self.cancel_flag

    def get_progress(self) -> Optional[ProcessingProgress]:
        """Get current processing progress."""
        return self.processing_progress

    def get_queue_size(self) -> int:
        """Get the number of batches in the queue."""
        with self.queue_lock:
            return len(self.batch_queue)

    def cancel_processing(self) -> bool:
        """Cancel ongoing processing."""
        if self.processing_progress and not self.processing_progress.is_complete:
            logger.info("Setting cancel flag for ongoing processing")
            self.cancel_flag = True
            self.save_processing_state()  # Save state after setting cancel flag
            return True
        return False

    def add_batch_to_queue(
        self,
        source_path: str,
        target_paths: List[str],
        target_filenames: List[str],
        swap_all_faces: bool = False
    ) -> bool:
        """Add a batch to the processing queue."""
        batch = {
            'source_path': source_path,
            'target_paths': target_paths,
            'target_filenames': target_filenames,
            'swap_all_faces': swap_all_faces,
            'submitted_time': time.time()
        }

        with self.queue_lock:
            self.batch_queue.append(batch)
            logger.info(f"Added batch with {len(target_filenames)} files to queue. Queue size: {len(self.batch_queue)}")

        # Start processing if no current batch is running
        if not self.processing_progress or self.processing_progress.is_complete:
            self._start_next_batch_from_queue()

        return True

    def add_files_to_current_batch(
        self,
        source_path: str,
        additional_target_paths: List[str],
        additional_target_filenames: List[str],
        swap_all_faces: bool = False
    ) -> bool:
        """Add additional files to the currently running batch."""
        logger.info(f"🚀 CALLING add_files_to_current_batch with {len(additional_target_filenames)} files")

        if not self.processing_progress or self.processing_progress.is_complete:
            logger.warning("❌ No active batch to add files to, starting new batch")
            return self.add_batch_to_queue(source_path, additional_target_paths, additional_target_filenames, swap_all_faces)

        # For merge capability, allow different sources - just warn user
        logger.info(f"⚠️ Allowing merge with different source (this is intended for flexibility)")
        logger.info(f"📝 Current source: {getattr(self.processing_progress, 'source_path', 'None')[:50]}...")
        logger.info(f"📝 New source: {source_path[:50]}...")

        # Add files to current batch
        current_targets = getattr(self.processing_progress, 'target_paths', [])
        current_filenames = getattr(self.processing_progress, 'target_filenames', [])
        current_swap_all_faces = getattr(self.processing_progress, 'swap_all_faces', False)

        logger.info(f"📊 Before merge: {len(current_filenames)} files, adding {len(additional_target_filenames)} files")

        # Extend with new files
        new_target_paths = current_targets + additional_target_paths
        new_target_filenames = current_filenames + additional_target_filenames

        # Update batch data
        self.processing_progress.target_paths = new_target_paths
        self.processing_progress.target_filenames = new_target_filenames
        self.processing_progress.total_files = len(new_target_filenames)
        self.processing_progress.queue.extend(additional_target_filenames)

        logger.info(f"📈 After merge: total_files={self.processing_progress.total_files}, queue_length={len(self.processing_progress.queue)}")

        # Initialize progress for new files
        from .models import FileProgress
        for filename, target_path in zip(additional_target_filenames, additional_target_paths):
            ext = filename.lower().split('.')[-1]
            file_type = 'video' if ext in ['mp4', 'avi', 'mov', 'mkv', 'gif'] else 'image'

            # For videos, get frame count
            total_frames = None
            if file_type == 'video':
                try:
                    if os.path.exists(target_path):
                        _, _, total_frames, _ = self.swap_processor.video_processor.get_video_info(target_path)
                except Exception as e:
                    logger.debug(f"Could not get video info for {filename}: {e}")

            self.processing_progress.file_progress[filename] = FileProgress(
                filename=filename,
                status="queued",
                file_type=file_type,
                total_frames=total_frames,
                start_time=time.time()
            )
            logger.info(f"✅ Added file to progress: {filename}")

        # Save updated state after merging files
        logger.info("💾 Saving processing state after merge...")
        try:
            self.save_processing_state()
            logger.info("✅ Processing state saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save processing state after merge: {e}")

        logger.info(f"🎉 Successfully added {len(additional_target_filenames)} files to current batch. Final total: {len(new_target_filenames)}")
        return True

    def _start_next_batch_from_queue(self) -> bool:
        """Start processing the next batch from queue."""
        with self.queue_lock:
            if not self.batch_queue:
                return False
            batch = self.batch_queue.pop(0)

        source_path = batch['source_path']
        target_paths = batch['target_paths']
        target_filenames = batch['target_filenames']
        swap_all_faces = batch['swap_all_faces']

        self.processing_progress = ProcessingProgress(
            total_files=len(target_paths),
            completed=0,
            queue=target_filenames.copy(),
            start_time=time.time(),
            results=[],
            source_path=source_path,
            target_paths=target_paths,
            target_filenames=target_filenames,
            swap_all_faces=swap_all_faces
        )

        # Reset cancellation state for new batch
        self.cancel_flag = False

        # Save state immediately when starting
        self.save_processing_state()

        # Start processing thread
        thread = threading.Thread(
            target=self._process_batch_with_queue,
            args=(source_path, target_paths, target_filenames, swap_all_faces),
            daemon=True
        )
        thread.start()
        return True

    def start_background_processing(
        self,
        source_path: str,
        target_paths: List[str],
        target_filenames: List[str],
        swap_all_faces: bool = False
    ) -> bool:
        """Start background batch processing (adds to queue and processes immediately if possible)."""
        return self.add_batch_to_queue(source_path, target_paths, target_filenames, swap_all_faces)

    def _process_batch_with_queue(
        self,
        source_path: str,
        target_paths: List[str],
        target_filenames: List[str],
        swap_all_faces: bool = False
    ):
        """Background processing thread for queued batch."""
        try:
            # Initialize file progress tracking
            self._initialize_file_progress(target_filenames, target_paths)
            self.processing_progress.queue = target_filenames.copy()

            results = self._process_files_sequentially(source_path, target_paths, target_filenames, swap_all_faces)
            self.processing_progress.results = results
            self.processing_progress.completed = len(target_paths)

        except Exception as e:
            logger.error(f"Background processing error: {e}")
            # Mark processing as complete with error
            if self.processing_progress:
                self.processing_progress.results = [
                    SwapResult(success=False, error_message=str(e))
                    for _ in target_filenames
                ]

        finally:
            # Mark as complete and start next batch if available
            if self.processing_progress:
                # Setting completed to total_files makes is_complete property return True
                self.processing_progress.completed = self.processing_progress.total_files

            # Start next batch from queue
            self._start_next_batch_from_queue()

    def _process_batch(
        self,
        source_path: str,
        target_paths: List[str],
        target_filenames: List[str],
        swap_all_faces: bool = False
    ):
        """Background processing thread."""
        try:
            # Initialize file progress tracking
            self._initialize_file_progress(target_filenames, target_paths)
            self.processing_progress.queue = target_filenames.copy()

            results = self._process_files_sequentially(source_path, target_paths, target_filenames, swap_all_faces)
            self.processing_progress.results = results
            self.processing_progress.completed = len(target_paths)

        except Exception as e:
            logger.error(f"Background processing error: {e}")
            # Mark processing as complete with error
            if self.processing_progress:
                self.processing_progress.results = [
                    SwapResult(success=False, error_message=str(e))
                    for _ in target_filenames
                ]

        finally:
            # Mark as complete and save final state
            if self.processing_progress:
                # Setting completed to total_files makes is_complete property return True
                self.processing_progress.completed = self.processing_progress.total_files

            # Save final state (or clear if successful)
            if self.processing_progress and self.processing_progress.results:
                # Only keep failed results for potential retry, but for now just clear
                self.clear_processing_state()
            else:
                # Clear state file when done
                self.clear_processing_state()

    def _initialize_file_progress(self, target_filenames: List[str], target_paths: List[str]):
        """Initialize progress tracking for each file."""
        from .models import FileProgress

        self.processing_progress.file_progress = {}
        for filename, target_path in zip(target_filenames, target_paths):
            ext = filename.lower().split('.')[-1]
            file_type = 'video' if ext in ['mp4', 'avi', 'mov', 'mkv', 'gif'] else 'image'

            # For videos, get frame count
            total_frames = None
            if file_type == 'video':
                try:
                    if os.path.exists(target_path):
                        _, _, total_frames, _ = self.swap_processor.video_processor.get_video_info(target_path)
                except Exception as e:
                    logger.debug(f"Could not get video info for {filename}: {e}")

            self.processing_progress.file_progress[filename] = FileProgress(
                filename=filename,
                status="pending",
                file_type=file_type,
                total_frames=total_frames
            )

    def _process_files_sequentially(self, source_path: str, target_paths: List[str], target_filenames: List[str], swap_all_faces: bool = False) -> List[SwapResult]:
        """Process files sequentially with individual progress tracking."""
        try:
            # Initialize cancelled_files if not exists
            if not hasattr(self.processing_progress, 'cancelled_files'):
                self.processing_progress.cancelled_files = []

            # Load and cache source face
            source_img = cv2.imread(source_path)
            if source_img is None:
                return [SwapResult(
                    success=False,
                    error_message="Cannot read source image"
                ) for _ in target_filenames]

            source_result = self.swap_processor.face_services.detector.detect_faces(source_img)
            if not source_result.success:
                return [SwapResult(
                    success=False,
                    error_message=source_result.error_message or "No source face detected"
                ) for _ in target_filenames]

            source_face_cache = source_result.faces[0]
            results = []
            processed_count = 0

            # Process each target
            for i, (target_path, target_filename) in enumerate(zip(target_paths, target_filenames)):
                # Check for full cancellation before processing each file
                if self.cancel_flag:
                    logger.info(f"Full cancellation detected before processing {target_filename}, stopping...")
                    # Mark remaining files as cancelled
                    for j in range(i, len(target_filenames)):
                        remaining_filename = target_filenames[j]
                        if remaining_filename not in self.processing_progress.cancelled_files:
                            self._update_file_progress(remaining_filename, "cancelled")
                            results.append(SwapResult(
                                success=False,
                                error_message="Processing cancelled by user",
                                original_name=remaining_filename
                            ))
                    break

                # Check if this specific file was cancelled
                if target_filename in self.processing_progress.cancelled_files:
                    logger.info(f"Skipping cancelled file: {target_filename}")
                    self._update_file_progress(target_filename, "cancelled", progress_percentage=0.0)
                    results.append(SwapResult(
                        success=False,
                        error_message="File cancelled by user",
                        original_name=target_filename
                    ))
                    continue

                logger.info(f"Processing {processed_count+1}/{len(target_paths)}: {target_filename}")

                # Update progress: mark as processing
                self._update_file_progress(target_filename, "processing", start_time=time.time())

                # Process the file with progress callback
                result = self._process_single_file_with_progress(
                    source_face_cache, target_path, target_filename, swap_all_faces
                )
                results.append(result)
                processed_count += 1

                # Mark as completed
                self._update_file_progress(
                    target_filename,
                    "completed" if result.success else "error",
                    progress_percentage=100.0
                )

                # Update overall progress
                self.processing_progress.completed = processed_count
                self.processing_progress.current_file = target_filename

            return results

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [SwapResult(
                success=False,
                error_message=f"Processing error: {str(e)}"
            ) for _ in target_filenames]

    def _fallback_to_copy(self, target_path: str, target_filename: str) -> SwapResult:
        """Fallback: copy original file to output when processing fails."""
        logger.warning(f"Falling back to copy original file for {target_filename}")
        try:
            import shutil
            output_uuid = str(uuid.uuid4())
            if VideoProcessor.is_video_file(target_filename):
                output_path = f"{self.swap_processor.config.output_directory}/{output_uuid}_{target_filename}"
            else:
                output_path = f"{self.swap_processor.config.output_directory}/{output_uuid}_{target_filename}"

            shutil.copy2(target_path, output_path)
            logger.info(f"Original file copied to: {output_path}")

            file_type = FileType.VIDEO if VideoProcessor.is_video_file(target_filename) else FileType.IMAGE
            return SwapResult(
                success=True,
                output_path=f"/view/{output_path}",
                original_name=target_filename,
                file_type=file_type
            )
        except Exception as copy_e:
            logger.error(f"Failed to copy original file for {target_filename}: {copy_e}")
            return SwapResult(
                success=False,
                error_message=f"Processing failed and could not copy original file",
                original_name=target_filename
            )

    def _process_single_file_with_progress(self, source_face_cache, target_path: str, target_filename: str, swap_all_faces: bool = False) -> SwapResult:
        """Process a single file with progress tracking."""
        logger.info(f"Starting processing of file: {target_filename}")
        target_img = cv2.imread(target_path)

        if target_img is not None:
            # Process image
            logger.debug(f"Processing image: {target_filename}")
            self._update_file_progress(target_filename, "processing", progress_percentage=50.0)
            result = self.swap_processor.process_image(source_face_cache, target_path, target_filename, swap_all_faces)
            if result.success:
                logger.info(f"Image processing completed: {target_filename}")
            else:
                logger.error(f"Image processing failed: {target_filename} - {result.error_message}")
            return result
        elif VideoProcessor.is_video_file(target_filename):
            # Initialize video progress
            self._update_file_progress(target_filename, "processing", progress_percentage=0.0)

            # Check if video is readable before processing
            try:
                cap_test = cv2.VideoCapture(target_path)
                if not cap_test.isOpened():
                    logger.error(f"Cannot open video file {target_filename} for reading")
                    cap_test.release()
                    return self._fallback_to_copy(target_path, target_filename)
                cap_test.release()
            except Exception as e:
                logger.error(f"Video file validation failed for {target_filename}: {e}")
                return self._fallback_to_copy(target_path, target_filename)

            try:
                fps, width, height, total_frames = self.swap_processor.video_processor.get_video_info(target_path)
                logger.info(f"Video info for {target_filename}: {width}x{height}, {total_frames} frames, {fps} FPS")
                if total_frames:
                    self._update_file_progress(
                        target_filename,
                        "processing",
                        total_frames=total_frames
                    )
            except Exception as e:
                logger.warning(f"Could not get video info for {target_filename}: {e}, trying to process anyway")

            # Process video with frame-by-frame progress
            def video_progress_callback(processed_frames, total_frames):
                percentage = (processed_frames / total_frames * 100) if total_frames > 0 else 0
                logger.debug(f"Video {target_filename}: {processed_frames}/{total_frames} frames ({percentage:.1f}%)")
                self._update_file_progress(
                    target_filename,
                    "processing",
                    progress_percentage=percentage,
                    current_frame=processed_frames,
                    total_frames=total_frames
                )

            logger.info(f"Starting video processing: {target_filename}")
            result = self.swap_processor.process_video_with_audio(
                source_face_cache, target_path, target_filename, video_progress_callback, self.is_cancelled, swap_all_faces
            )

            if result.success:
                logger.info(f"Video processing completed: {target_filename}")
            else:
                logger.error(f"Video processing failed: {target_filename} - {result.error_message}")
                # Try to copy original on video processing failure
                try:
                    return self._fallback_to_copy(target_path, target_filename)
                except:
                    pass

            return result
        else:
            logger.warning(f"Unsupported file type: {target_filename}")
            return SwapResult(
                success=False,
                error_message=f"Unsupported file type: {target_filename}",
                original_name=target_filename
            )

    def _update_file_progress(self, filename: str, status: str = None, progress_percentage: float = None,
                            current_frame: int = None, total_frames: int = None, start_time: float = None):
        """Update progress for a specific file."""
        if filename in self.processing_progress.file_progress:
            file_prog = self.processing_progress.file_progress[filename]
            if status is not None:
                file_prog.status = status
            if progress_percentage is not None:
                file_prog.progress_percentage = progress_percentage
            if current_frame is not None:
                file_prog.current_frame = current_frame
            if total_frames is not None:
                file_prog.total_frames = total_frames
            if start_time is not None:
                file_prog.start_time = start_time

    def update_video_progress(self, current_video_total_frames: int, current_video_processed_frames: int):
        """Update video processing progress."""
        if self.processing_progress:
            self.processing_progress.video_progress = {
                "current_video_total_frames": current_video_total_frames,
                "current_video_processed_frames": current_video_processed_frames,
                "current_video_progress_percentage": (
                    (current_video_processed_frames / current_video_total_frames * 100)
                    if current_video_total_frames > 0 else 0
                )
            }
