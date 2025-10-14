"""
Video processing utilities for face swapping.
"""
import logging
import os
from typing import Callable, Tuple, Optional
import cv2
import numpy as np
from .models import SwapResult, FileType

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Utility class for video processing operations."""

    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size

    def get_video_info(self, video_path: str) -> Tuple[int, int, int, int]:
        """Get basic video information."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        return fps, frame_width, frame_height, total_frames

    def create_video_writer(self, output_path: str, fps: int, width: int, height: int) -> cv2.VideoWriter:
        """Create video writer with appropriate codec (fallback strategy)."""
        # Try multiple codecs in order of preference
        codecs_to_try = [
            ('avc1', 'H.264'),
            ('mp4v', 'MPEG-4'),
            ('xvid', 'XVID'),
            ('X264', 'H.264 alternative'),
            ('MJPG', 'Motion JPEG'),
            ('DIVX', 'DIVX'),
            ('WMV1', 'Windows Media Video'),
            ('WMV2', 'Windows Media Video 2')
        ]

        last_error = None
        working_writer = None

        for fourcc_code, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                # Use .avi extension for better codec support
                test_path = output_path.replace('.mp4', '.avi')

                writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))

                # Test if writer opened successfully by writing a test frame
                if writer.isOpened():
                    # Create a test frame and write it
                    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    writer.write(test_frame)

                    # Try to release and reopen to test persistence
                    writer.release()
                    writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))

                    if writer.isOpened():
                        logger.info(f"Successfully created video writer with codec: {codec_name} ({fourcc_code})")
                        # Clean up test file
                        if os.path.exists(test_path):
                            os.remove(test_path)

                        # Recreate writer for actual output path
                        actual_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        if actual_writer.isOpened():
                            return actual_writer
                        else:
                            actual_writer.release()

                    # Clean up if not successful
                    writer.release()
                    if os.path.exists(test_path):
                        os.remove(test_path)

                last_error = f"Codec {codec_name} ({fourcc_code}) not available"

            except Exception as e:
                last_error = f"Error with codec {codec_name} ({fourcc_code}): {str(e)}"
                logger.debug(f"Codec {codec_name} failed: {e}")
                continue

        # Final fallback: try to create with no codec specified (let OpenCV choose)
        try:
            logger.warning("Trying fallback video writer without codec specification")
            writer = cv2.VideoWriter(output_path, -1, fps, (width, height))
            if writer.isOpened():
                logger.info("Fallback video writer created successfully")
                return writer
            writer.release()
        except Exception as e:
            logger.error(f"Fallback video writer also failed: {e}")

        # If all attempts fail, raise error with detailed diagnostics
        raise RuntimeError(f"All video codecs failed. Last error: {last_error}. "
                          f"Available codecs may be limited on this system. "
                          f"Consider installing additional video codecs or ffmpeg.")

    def process_video_batch(
        self,
        video_path: str,
        output_path: str,
        face_swap_callback: Callable[[np.ndarray], np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """Process video with batched face swapping."""
        try:
            logger.info(f"Processing video: {video_path}")

            # Get video info
            fps, frame_width, frame_height, total_frames = self.get_video_info(video_path)

            # Initialize video capture and writer with codec validation
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file for processing: {video_path}")
                return False

            try:
                out = self.create_video_writer(output_path, fps, frame_width, frame_height)
            except RuntimeError as e:
                logger.error(f"Video writer creation failed: {e}")
                cap.release()
                return False

            frames = []
            frame_count = 0
            processed_frames_count = 0

            logger.info(f"Video info - FPS: {fps}, Size: {frame_width}x{frame_height}, Frames: {total_frames}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)
                frame_count += 1

                # Process batch when enough frames or at end
                if len(frames) >= self.max_batch_size or frame_count == total_frames:
                    logger.debug(f"Processing batch of {len(frames)} frames")

                    processed_frames = []
                    batch_frame_count = len(frames)
                    for i, frame in enumerate(frames):
                        try:
                            processed_frame = face_swap_callback(frame)
                            # Check if we got a valid result
                            if processed_frame is not None and processed_frame.size > 0:
                                processed_frames.append(processed_frame)
                                processed_frames_count += 1
                                logger.debug(f"Processed video frame {frame_count - batch_frame_count + i + 1}/{total_frames}")
                            else:
                                logger.debug(f"Face swap returned empty result on frame {frame_count - batch_frame_count + i + 1}, using original frame")
                                processed_frames.append(frame)
                        except Exception as e:
                            logger.debug(f"Frame processing error on frame {frame_count - batch_frame_count + i + 1} (using original): {e}")
                            processed_frames.append(frame)  # Use original frame on error

                    # Write processed frames
                    for processed_frame in processed_frames:
                        out.write(processed_frame)

                    # Update progress with actual processed frames count
                    if progress_callback:
                        progress_callback(frame_count, total_frames)

                    frames.clear()  # Reset batch

            # Cleanup
            cap.release()
            out.release()

            logger.info(f"Video processing completed: {output_path}, processed {processed_frames_count} frames")
            return True

        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return False

    @staticmethod
    def is_video_file(filename: str) -> bool:
        """Check if file is a video based on extension."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return any(filename.lower().endswith(ext) for ext in video_extensions)
