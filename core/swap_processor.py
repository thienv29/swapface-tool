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
        target_filename: str
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
        cancellation_callback: Optional[Callable[[], bool]] = None
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

                # Define swap callback for video processing
                def swap_callback(frame):
                    # Detect faces in current frame
                    frame_result = self.face_services.detector.detect_faces(frame)
                    if not frame_result.success or not frame_result.faces:
                        logger.debug("No faces detected in frame, skipping swap")
                        return frame

                    try:
                        # Perform swap
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
                        # Perform swap
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

            source_result = self.face_services.detector.detect_faces(source_img)
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

    def is_cancelled(self) -> bool:
        """Check if processing has been cancelled."""
        return self.cancel_flag

    def get_progress(self) -> Optional[ProcessingProgress]:
        """Get current processing progress."""
        return self.processing_progress

    def cancel_processing(self) -> bool:
        """Cancel ongoing processing."""
        if self.processing_progress and not self.processing_progress.is_complete:
            logger.info("Setting cancel flag for ongoing processing")
            self.cancel_flag = True
            return True
        return False

    def start_background_processing(
        self,
        source_path: str,
        target_paths: List[str],
        target_filenames: List[str]
    ) -> bool:
        """Start background batch processing."""
        if self.processing_progress and not self.processing_progress.is_complete:
            return False  # Already processing

        self.processing_progress = ProcessingProgress(
            total_files=len(target_paths),
            completed=0,
            queue=target_filenames.copy(),
            start_time=time.time(),
            results=[]
        )

        # Add processing lock to prevent concurrent uploads
        import threading
        self.processing_lock = threading.Lock()
        with self.processing_lock:
            thread = threading.Thread(
                target=self._process_batch,
                args=(source_path, target_paths, target_filenames),
                daemon=True
            )
            thread.start()
            return True

    def _process_batch(
        self,
        source_path: str,
        target_paths: List[str],
        target_filenames: List[str]
    ):
        """Background processing thread."""
        try:
            # Initialize file progress tracking
            self._initialize_file_progress(target_filenames, target_paths)
            self.processing_progress.queue = target_filenames.copy()

            results = self._process_files_sequentially(source_path, target_paths, target_filenames)
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
            # Clear queue and mark as complete
            if self.processing_progress:
                self.processing_progress.queue = []

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

    def _process_files_sequentially(self, source_path: str, target_paths: List[str], target_filenames: List[str]) -> List[SwapResult]:
        """Process files sequentially with individual progress tracking."""
        try:
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

            # Process each target
            for i, (target_path, target_filename) in enumerate(zip(target_paths, target_filenames)):
                # Check for cancellation before processing each file
                if self.cancel_flag:
                    logger.info(f"Cancellation detected before processing {target_filename}, stopping...")
                    # Mark remaining files as cancelled
                    for j in range(i, len(target_filenames)):
                        remaining_filename = target_filenames[j]
                        self._update_file_progress(remaining_filename, "cancelled")
                        results.append(SwapResult(
                            success=False,
                            error_message="Processing cancelled by user",
                            original_name=remaining_filename
                        ))
                    break

                logger.info(f"Processing {i+1}/{len(target_paths)}: {target_filename}")

                # Update progress: mark as processing
                self._update_file_progress(target_filename, "processing", start_time=time.time())

                # Process the file with progress callback
                result = self._process_single_file_with_progress(
                    source_face_cache, target_path, target_filename
                )
                results.append(result)

                # Mark as completed
                self._update_file_progress(
                    target_filename,
                    "completed" if result.success else "error",
                    progress_percentage=100.0
                )

                # Update overall progress
                self.processing_progress.completed = i + 1
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

    def _process_single_file_with_progress(self, source_face_cache, target_path: str, target_filename: str) -> SwapResult:
        """Process a single file with progress tracking."""
        logger.info(f"Starting processing of file: {target_filename}")
        target_img = cv2.imread(target_path)

        if target_img is not None:
            # Process image
            logger.debug(f"Processing image: {target_filename}")
            self._update_file_progress(target_filename, "processing", progress_percentage=50.0)
            result = self.swap_processor.process_image(source_face_cache, target_path, target_filename)
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
                source_face_cache, target_path, target_filename, video_progress_callback, self.is_cancelled
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
