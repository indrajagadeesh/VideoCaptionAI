import os
import ffmpeg
import subprocess
import logging
from ..configs.app_config import DEVICE

logger = logging.getLogger(__name__)

def extract_audio(input_video: str, output_audio: str):
    """
    ffmpeg -i input_video.mp4 -vn -ar 16000 -ac 1 -c:a mp3 -b:a 32k output.mp3
    """
    logger.info(f"Starting audio extraction from {input_video} to {output_audio}")
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream,
                           output_audio,
                           acodec='mp3',
                           ac=1,
                           ar='16k',
                           ab='32k',
                           vn=None)
    ffmpeg.run(stream, overwrite_output=True)
    logger.info(f"Audio extraction completed for {input_video}")

def check_cuda_availability():
    """Check if CUDA is available in ffmpeg"""
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], 
                              capture_output=True, text=True)
        return 'cuda' in result.stdout.lower()
    except Exception as e:
        logger.warning(f"Failed to check CUDA availability: {e}")
        return False


def add_subtitles_to_video(input_video: str, ass_file: str, output_video: str):
    """
    Add subtitles to video using FFmpeg, with CUDA acceleration if available.
    Ensures audio stream is properly preserved.
    """
    logger.info(f"Starting subtitle addition for {input_video} using subtitles file {ass_file}")
    
    try:
        # Check for CUDA availability
        has_cuda = check_cuda_availability()
        
        # Create input stream
        input_stream = ffmpeg.input(input_video)
        
        if has_cuda:
            logger.info("Using CUDA hardware acceleration")
            # Hardware-accelerated version
            video_stream = (
                input_stream.video
                .filter('subtitles', ass_file)
                .output(output_video,
                    vcodec='h264_nvenc',  # Use NVIDIA encoder
                    preset='p7',          # NVENC preset
                    **{'b:v': '8M'}       # Video bitrate
                )
            )
        else:
            logger.info("CUDA not available, using CPU encoding")
            # CPU version
            video_stream = (
                input_stream.video
                .filter('subtitles', ass_file)
                .output(output_video,
                    vcodec='libx264',     # CPU encoder
                    preset='medium',      # x264 preset
                    **{'b:v': '8M'}       # Video bitrate
                )
            )
        
        # Combine video and audio streams
        stream = (
            ffmpeg
            .output(
                input_stream.video.filter('subtitles', ass_file),
                input_stream.audio,       # Add audio stream
                output_video,
                acodec='copy',            # Copy audio without re-encoding
                vcodec='h264_nvenc' if has_cuda else 'libx264',
                preset='p7' if has_cuda else 'medium',
                **{'b:v': '8M'}
            )
        )
        
        # Run the FFmpeg command
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        logger.info(f"Subtitle addition completed successfully for {output_video}")
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error occurred: {error_message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise

# Example usage with error handling
def process_video_safely(input_video: str, ass_file: str, output_video: str):
    try:
        add_subtitles_to_video(input_video, ass_file, output_video)
    except Exception as e:
        logger.error(f"Failed to process video {input_video}: {str(e)}")
        # Handle the error appropriately (e.g., retry, skip, etc.)


