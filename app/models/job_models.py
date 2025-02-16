from pydantic import BaseModel
from typing import Optional, List

class JobResponse(BaseModel):
    jobId: str
    recordId: Optional[str]
    status: str
    message: str

class WordInfo(BaseModel):
    word: str
    start: float
    end: float
    confidence: float

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    words: List[WordInfo]

class TranscriptionResponse(BaseModel):
    jobId: str
    recordId: Optional[str]
    status: str
    message: str
    segments: List[TranscriptionSegment]
