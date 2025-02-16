from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
from .ass_settings import AssSettings

class VideoRequest(BaseModel):
    video_url: Optional[HttpUrl] = None
    video_path: Optional[str] = None
    recordId: Optional[str] = None
    language: str = Field(default="en", description="Language code for Whisper.")
    webhook_url: Optional[HttpUrl] = None
    ass_settings: AssSettings = AssSettings()

    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "recordId": "abc123",
                "language": "en",
                "webhook_url": "https://example.com/mywebhook",
                "ass_settings": {
                    "Fontname": "Arial",
                    "Fontsize": 40
                }
            }
        }
