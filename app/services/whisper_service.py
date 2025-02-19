import torch
import json
import logging
from typing import Dict
from ..configs.app_config import whisper_model, DEVICE

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path: str, language: str = "en", max_words_per_segment: int = 10) -> Dict:
    """
    Transcribe with Whisper using word-level timestamps, then limit each segment to
    a specified number of words (max_words_per_segment).
    """

    # Log the start of transcription
    logger.info(f"Starting transcription of audio file: {audio_path} using device: {DEVICE}")

    # The model is loaded from config as `whisper_model`.
    # It's already moved to `DEVICE` (cpu or cuda).
    result = whisper_model.transcribe(
        audio_path, 
        language=language, 
        word_timestamps=True
    )

    # Log the amount of segments received from transcription
    num_segments = len(result.get("segments", []))
    logger.info(f"Transcription complete for {audio_path}. {num_segments} segments found.")

    limited_segments = []
    for segment in result["segments"]:
        words = segment.get("words", [])
        
        # If there are no word-level timestamps, just keep the segment as-is
        if not words:
            limited_segments.append(segment)
            continue
        
        # Split words into smaller chunks based on max_words_per_segment
        for i in range(0, len(words), max_words_per_segment):
            chunk = words[i : i + max_words_per_segment]
            
            # Build the sub-segment's text from the chunk
            chunk_text = "".join([w["word"] for w in chunk]).strip()

            # Start time from the first word, end time from the last word
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]

            # Create a new segment for this chunk
            new_seg = {
                "start": chunk_start,
                "end": chunk_end,
                "text": chunk_text,
                "words": chunk,
            }
            
            limited_segments.append(new_seg)

    # Build the final JSON with our new limited segments
    subtitle_json = {
        "segments": limited_segments
    }

    return subtitle_json
