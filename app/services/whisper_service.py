import torch
import json
import logging
from typing import Dict
from ..configs.app_config import whisper_model, DEVICE

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path: str, language: str = "en") -> Dict:
    """
    Transcribe with Whisper using word-level timestamps.
    """
    # Log the start of transcription
    logger.info(f"Starting transcription of audio file: {audio_path} using device: {DEVICE}")
    
    # The model is loaded from config as `whisper_model`.
    # It's already moved to `DEVICE` (cpu or cuda).
    result = whisper_model.transcribe(audio_path, language=language, word_timestamps=True)
    
    # Log the amount of segments received from transcription
    num_segments = len(result.get("segments", []))
    logger.info(f"Transcription complete for {audio_path}. {num_segments} segments found.")
    
    subtitle_json = {
        "segments": [{
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "words": s.get("words", [])
        } for s in result["segments"]]
    }
    return subtitle_json
