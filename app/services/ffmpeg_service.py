import os
import ffmpeg
import subprocess
import logging
import glob
from ..configs.app_config import DEVICE
import time
from typing import Dict, Optional, Tuple, List
from functools import lru_cache
from contextlib import contextmanager

def configure_logging(level=logging.DEBUG):
    """
    Configure logging to show output in terminal (stdout).
    Adjust the format and level as needed.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # If the root logger has no handlers, set up the basics
    if not root_logger.handlers:
        # You can change the log format here if you want
        log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        
        # Configure logging with a StreamHandler
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )

# Call configure_logging once at the module level (or from your app entry point).
configure_logging(logging.INFO)

logger = logging.getLogger(__name__)

class FFmpegSettings:
    """Constants for FFmpeg settings"""
    CUDA_SETTINGS = {
        'vcodec': 'h264_nvenc',
        'preset': 'p7',
        'rc': 'vbr',
        'b:v': '8M',
        'maxrate': '10M',
        'bufsize': '20M'
    }
    
    CPU_SETTINGS = {
        'vcodec': 'libx264',
        'preset': 'medium',
        'b:v': '8M',
        'maxrate': '10M',
        'bufsize': '20M',
        'threads': 'auto'
    }
    
    COMMON_SETTINGS = {
        'acodec': 'copy',
        'movflags': '+faststart',
        'loglevel': 'error'
    }
    
    AUDIO_SETTINGS = {
        'acodec': 'mp3',
        'ac': 1,
        'ar': '16k',
        'ab': '32k',
        'vn': None,
        'loglevel': 'error'
    }
    
    # Updated: Set your custom font directory here
    FONT_DIR = "/app/assets/fonts"
    FONT_EXTENSIONS = ['.ttf', '.otf', '.TTF', '.OTF']

@contextmanager
def timer(operation: str, context: Optional[str] = None):
    """Context manager for timing operations with detailed logging."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if context:
            logger.info(f"{operation} completed in {duration:.2f}s - {context}")
        else:
            logger.info(f"{operation} completed in {duration:.2f}s")

@lru_cache(maxsize=1)
def get_ffmpeg_info() -> Dict[str, str]:
    """Get FFmpeg version and capabilities information."""
    try:
        version_info = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True
        ).stdout.split(os.linesep)[0]
        
        hwaccels = subprocess.run(
            ['ffmpeg', '-hide_banner', '-hwaccels'], 
            capture_output=True, 
            text=True
        ).stdout.strip().split(os.linesep)
        
        return {
            'version': version_info,
            'hwaccels': hwaccels
        }
    except Exception as e:
        logger.error(f"Failed to get FFmpeg information: {str(e)}")
        return {'error': str(e)}

def is_cuda_available() -> bool:
    """Check if CUDA is available based on config DEVICE."""
    return DEVICE.lower() == 'cuda'

def get_nvidia_info() -> Optional[str]:
    """Get NVIDIA GPU information if available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        logger.warning("nvidia-smi not found")
        return None
    except Exception as e:
        logger.error(f"Error getting NVIDIA info: {str(e)}")
        return None

def get_stream_info(file_path: str, context: str = "") -> Optional[Dict]:
    """Get media file information using ffprobe with detailed error handling."""
    with timer(f"Media probe{f' for {context}' if context else ''}", file_path):
        try:
            probe = ffmpeg.probe(file_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            info = {
                'codec': video_stream.get('codec_name', 'unknown'),
                'width': video_stream.get('width', '?'),
                'height': video_stream.get('height', '?'),
                'fps': video_stream.get('r_frame_rate', '?'),
                'duration': float(video_stream.get('duration', 0)),
                'bitrate': int(video_stream.get('bit_rate', 0)) // 1000
            }
            
            logger.info(
                f"Media info for {os.path.basename(file_path)}: "
                f"{info['codec']} {info['width']}x{info['height']} "
                f"@ {info['fps']}fps, {info['duration']:.2f}s, {info['bitrate']}kbps"
            )
            return info
            
        except Exception as e:
            logger.error(f"Failed to probe media file {file_path}: {str(e)}")
            return None

def extract_audio(input_video: str, output_audio: str) -> bool:
    """Extract audio from video using FFmpeg with optimized settings."""
    logger.info(f"Starting audio extraction: {input_video} -> {output_audio}")
    
    with timer("Audio extraction", input_video):
        try:
            stream = (
                ffmpeg
                .input(input_video)
                .output(output_audio, **FFmpegSettings.AUDIO_SETTINGS)
                .overwrite_output()
            )
            
            cmd = ffmpeg.get_args(stream)
            logger.info(f"FFmpeg audio extraction command: ffmpeg {' '.join(cmd)}")
            
            stream.run(capture_stdout=True, capture_stderr=True)
            return True
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}", exc_info=True)
            return False

def get_available_fonts() -> List[str]:
    """Get list of available fonts in the fonts directory."""
    try:
        if not os.path.exists(FFmpegSettings.FONT_DIR):
            os.makedirs(FFmpegSettings.FONT_DIR)
            logger.info(f"Created fonts directory: {FFmpegSettings.FONT_DIR}")
            return []
        
        fonts = []
        for ext in FFmpegSettings.FONT_EXTENSIONS:
            fonts.extend(glob.glob(os.path.join(FFmpegSettings.FONT_DIR, f"*{ext}")))
        
        font_names = [os.path.basename(f) for f in fonts]
        logger.info(f"Found {len(font_names)} fonts in {FFmpegSettings.FONT_DIR}: {font_names}")
        return fonts
    except Exception as e:
        logger.error(f"Error scanning fonts directory: {str(e)}")
        return []

def create_fonts_conf(fonts: List[str]) -> str:
    """
    Create (or overwrite) a fonts.conf file that maps each
    font-family to the corresponding font file path. This allows
    FFmpeg/fontconfig to locate and use custom fonts in ASS subtitles.
    """
    fonts_conf_content = """<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.conf">
<fontconfig>
    <dir>{font_dir}</dir>
    {font_patterns}
</fontconfig>
"""
    font_pattern = """
    <match target="pattern">
        <test qual="any" name="family">
            <string>{font_family}</string>
        </test>
        <edit name="file" mode="assign">
            <string>{font_path}</string>
        </edit>
    </match>"""
    
    patterns = []
    for font_path in fonts:
        font_family = os.path.splitext(os.path.basename(font_path))[0]
        patterns.append(font_pattern.format(
            font_family=font_family,
            font_path=font_path
        ))
    
    conf_path = os.path.join(FFmpegSettings.FONT_DIR, "fonts.conf")
    try:
        with open(conf_path, 'w', encoding='utf-8') as f:
            f.write(fonts_conf_content.format(
                font_dir=FFmpegSettings.FONT_DIR,
                font_patterns='\n'.join(patterns)
            ))
        logger.info(f"Created fonts configuration file: {conf_path}")
        return conf_path
    except Exception as e:
        logger.error(f"Failed to create fonts.conf: {str(e)}")
        return ""

def add_subtitles_to_video(input_video: str, ass_file: str, output_video: str) -> bool:
    """
    Add subtitles to video with optimized encoding settings and font support.
    All font styles in /app/assets/font will be recognized and used if your ASS
    file references them by the font name (family).
    """
    logger.info(f"Starting video processing pipeline for {os.path.basename(input_video)}")
    
    process_start = time.time()
    try:
        # 1. Gather all font files in /app/assets/font
        fonts = get_available_fonts()
        # 2. Create (or update) a fonts.conf
        fonts_conf = create_fonts_conf(fonts)
        
        # 3. Detect whether CUDA is available (from your app_config.py / DEVICE)
        has_cuda = is_cuda_available()
        logger.info(f"Using {'CUDA' if has_cuda else 'CPU'} for video processing")
        
        # 4. Get info about the input video
        input_info = get_stream_info(input_video, "input")
        if not input_info:
            logger.error("Failed to get input video information")
            return False
        
        # 5. Prepare output (encoding) settings
        output_settings = FFmpegSettings.COMMON_SETTINGS.copy()
        output_settings.update(
            FFmpegSettings.CUDA_SETTINGS if has_cuda else FFmpegSettings.CPU_SETTINGS
        )
        
        # 6. Tell FFmpeg/fontconfig to use our custom fonts.conf
        if fonts_conf:
            os.environ['FONTCONFIG_FILE'] = fonts_conf
            logger.info(f"Using custom fonts configuration: {fonts_conf}")
        
        logger.debug(f"Encoding settings: {output_settings}")
        
        # 7. Build the FFmpeg filter for subtitles, including fontsdir
        filter_params = {
            'filename': ass_file,
            'fontsdir': FFmpegSettings.FONT_DIR
        }
        
        with timer("Video processing"):
            input_stream = ffmpeg.input(input_video)
            
            # Subtitles filter: passes in 'filename=<ASS>' and 'fontsdir=<FONT_DIR>'
            stream = (
                ffmpeg
                .output(
                    input_stream.video.filter('subtitles', **filter_params),
                    input_stream.audio,
                    output_video,
                    **output_settings
                )
                .overwrite_output()
            )
            
            cmd = ffmpeg.get_args(stream)
            logger.info(f"FFmpeg command for subtitles: ffmpeg {' '.join(cmd)}")
            
            stream.run(capture_stdout=True, capture_stderr=True)
        
        # 8. Check resulting output
        if output_info := get_stream_info(output_video, "output"):
            logger.info(
                f"Output video quality: {output_info['codec']} "
                f"{output_info['width']}x{output_info['height']} @ {output_info['fps']}fps"
            )
        
        total_duration = time.time() - process_start
        logger.info(f"Total processing pipeline completed in {total_duration:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}", exc_info=True)
        if 'cmd' in locals():
            logger.error(f"Failed command: ffmpeg {' '.join(cmd)}")
        return False
    finally:
        # Clean up environment variable so it doesn't affect other processes
        if 'FONTCONFIG_FILE' in os.environ:
            del os.environ['FONTCONFIG_FILE']

def process_video_safely(input_video: str, ass_file: str, output_video: str):
    """
    High-level entry point to run the subtitle-adding pipeline with error handling.
    """
    try:
        success = add_subtitles_to_video(input_video, ass_file, output_video)
        if not success:
            logger.error(f"Failed to process {input_video} -> {output_video}")
    except Exception as e:
        logger.error(f"Exception during process_video_safely: {str(e)}", exc_info=True)
