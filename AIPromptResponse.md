Below is a **restructured** example project that follows a more **single responsibility** approach and uses:

- **FastAPI** for routing.
- A **local Python queue** (with a thread pool) for async job processing instead of Celery.
- **Config** and **environment variables** to manage the Whisper model and concurrency.
- **CUDA**/GPU check for Whisper.
- A simple **webhook** callback if `webhook_url` is provided; otherwise, a **status endpoint** is provided for polling.

> **Disclaimer**: This sample is for demonstration. You’ll likely expand, refine, or secure it further. Some design choices (e.g., in-memory storage for job states, ephemeral file paths) may need to be replaced with more robust solutions in production.

---

## Directory Structure

```
my_caption_app/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app (entry point)
│   ├── configs/
│   │   └── app_config.py       # Config / env variables, concurrency, etc.
│   ├── models/
│   │   ├── ass_settings.py     # Pydantic model for ASS style
│   │   ├── job_models.py       # Job request/response & transcription response
│   │   └── video_request.py    # The main request body for the endpoints
│   ├── routes/
│   │   └── video_routes.py     # All /v1/video/* endpoints
│   ├── services/
│   │   ├── ffmpeg_service.py   # Functions for audio/video processing
│   │   ├── whisper_service.py  # Whisper load and transcription
│   │   ├── ass_service.py      # Builds .ass from JSON
│   │   └── job_manager.py      # Local queue (thread pool) + job storage
│   └── utils/
│       └── file_utils.py       # Utility for file path building, etc.
├── temp/                       # Temporary files (auto-created)
├── storage/                    # Final processed files (auto-created)
├── requirements.txt
└── README.md
```

We'll walk through each component below.

---

## 1. `app/configs/app_config.py`

Manages environment variables, concurrency, default directories, and how we load the Whisper model.

```python
import os
import torch
import whisper

# Directories
TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")

# Make sure they exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Concurrency
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

# Whisper model configuration
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# Check for CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy load the Whisper model
# (Alternatively, load on first use in whisper_service)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME).to(DEVICE)
```

- `WHISPER_MODEL_NAME` is read from the environment (default `"base"`).  
- We detect if `cuda` is available, load Whisper onto GPU if possible.

---

## 2. `app/models/ass_settings.py`

Pydantic model for ASS style configuration (karaoke style by default).

```python
from pydantic import BaseModel

class AssSettings(BaseModel):
    Name: str = "Karaoke"
    Fontname: str = "Arial"
    Fontsize: int = 50
    PrimaryColour: str = "&H00FFFFFF"
    SecondaryColour: str = "&H000000FF"
    OutlineColour: str = "&H00000000"
    BackColour: str = "&H64000000"
    Bold: int = 0
    Italic: int = 0
    Underline: int = 0
    StrikeOut: int = 0
    ScaleX: int = 100
    ScaleY: int = 100
    Spacing: int = 0
    Angle: int = 0
    BorderStyle: int = 1
    Outline: int = 2
    Shadow: int = 0
    Alignment: int = 2
    MarginL: int = 10
    MarginR: int = 10
    MarginV: int = 10
    Encoding: int = 1
```

---

## 3. `app/models/video_request.py`

Main request body for the `/v1/video/caption` and `/v1/video/transcribe` endpoints.

```python
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
        schema_extra = {
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
```

- Either `video_url` or `video_path` must be provided by the caller.  
- `ass_settings` is used only by the caption endpoint.

---

## 4. `app/models/job_models.py`

Defines the job request/response and transcription response structures.

```python
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
```

---

## 5. `app/services/whisper_service.py`

Manages the actual call to the loaded Whisper model.  
We handle the **CUDA** device and model selection here, or we can do it once in config.

```python
import torch
import json
from typing import Dict
from ..configs.app_config import whisper_model, DEVICE

def transcribe_audio(audio_path: str, language: str = "en") -> Dict:
    """
    Transcribe with Whisper using word-level timestamps.
    """
    # The model is loaded from config as `whisper_model`.
    # It's already moved to `DEVICE` (cpu or cuda).
    result = whisper_model.transcribe(audio_path, language=language, word_timestamps=True)

    subtitle_json = {
        "segments": [{
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "words": s.get("words", [])
        } for s in result["segments"]]
    }
    return subtitle_json
```

---

## 6. `app/services/ffmpeg_service.py`

Handles audio extraction and final video creation with FFmpeg.

```python
import os
import ffmpeg

def extract_audio(input_video: str, output_audio: str):
    """
    ffmpeg -i input_video.mp4 -vn -ar 16000 -ac 1 -c:a mp3 -b:a 32k output.mp3
    """
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream,
                           output_audio,
                           acodec='mp3',
                           ac=1,
                           ar='16k',
                           ab='32k',
                           vn=None)
    ffmpeg.run(stream, overwrite_output=True)

def add_subtitles_to_video(input_video: str, ass_file: str, output_video: str):
    """
    ffmpeg -i input.mp4 -vf "ass=subs.ass" -c:a copy output.mp4
    """
    stream = ffmpeg.input(input_video)
    video = ffmpeg.filter_(stream, 'ass', ass_file)
    out = ffmpeg.output(video, output_video, c_a='copy')
    ffmpeg.run(out, overwrite_output=True)
```

---

## 7. `app/services/ass_service.py`

Utility to convert JSON transcripts into .ass karaoke subtitles.

```python
from typing import Dict
from ..models.ass_settings import AssSettings

def convert_json_to_ass(data: Dict, ass_settings: AssSettings) -> str:
    """
    Convert JSON (with segments & words) to an ASS karaoke file.
    """
    def format_time(sec: float) -> str:
        # H:MM:SS.cs
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        s_int = int(s)
        cs = int(round((s - s_int)*100))  # centiseconds
        return f"{h}:{m:02}:{s_int:02}.{cs:02}"

    lines = []
    # [Script Info]
    lines.append("[Script Info]")
    lines.append("Title: Karaoke Subtitles")
    lines.append("ScriptType: v4.00+")
    lines.append("PlayResX: 1280")
    lines.append("PlayResY: 720")
    lines.append("Timer: 100.0")
    lines.append("")

    # [V4+ Styles]
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
                 " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
                 " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
                 " Alignment, MarginL, MarginR, MarginV, Encoding")

    style_str = (
        f"Style: {ass_settings.Name},{ass_settings.Fontname},{ass_settings.Fontsize},"
        f"{ass_settings.PrimaryColour},{ass_settings.SecondaryColour},{ass_settings.OutlineColour},"
        f"{ass_settings.BackColour},{ass_settings.Bold},{ass_settings.Italic},{ass_settings.Underline},"
        f"{ass_settings.StrikeOut},{ass_settings.ScaleX},{ass_settings.ScaleY},{ass_settings.Spacing},"
        f"{ass_settings.Angle},{ass_settings.BorderStyle},{ass_settings.Outline},{ass_settings.Shadow},"
        f"{ass_settings.Alignment},{ass_settings.MarginL},{ass_settings.MarginR},{ass_settings.MarginV},"
        f"{ass_settings.Encoding}"
    )
    lines.append(style_str)
    lines.append("")

    # [Events]
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    # Build lines
    for seg in data["segments"]:
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        karaoke_parts = []
        for w in seg["words"]:
            duration_cs = int(round((w["end"] - w["start"])*100))
            karaoke_parts.append(f"{{\\k{duration_cs}}}{w['word']}")
        text_line = " ".join(karaoke_parts)
        line_str = f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,{text_line}"
        lines.append(line_str)

    return "\n".join(lines)
```

---

## 8. `app/services/job_manager.py`

Implements a **local in-memory** queue with **thread pool** concurrency. We track jobs in a dictionary, process them asynchronously, and can poll the status from the dictionary.

```python
import threading
import uuid
import time
from typing import Callable, Any, Dict
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from ..configs.app_config import MAX_WORKERS

# Store job statuses in-memory
JOBS: Dict[str, dict] = {}
TASK_QUEUE = Queue()

# Thread pool for concurrency
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def worker_loop():
    while True:
        try:
            job_id, func, args = TASK_QUEUE.get()
            job_data = JOBS[job_id]
            job_data["status"] = "processing"
            try:
                func(*args)
                # If everything is fine, status is completed
                if job_data["status"] != "failed":  # in case we set it to failed inside func
                    job_data["status"] = "completed"
            except Exception as e:
                job_data["status"] = "failed"
                job_data["message"] = str(e)
            finally:
                TASK_QUEUE.task_done()
        except Empty:
            time.sleep(0.1)

# Start a single thread that pulls tasks from the queue
threading.Thread(target=worker_loop, daemon=True).start()

def add_job(func: Callable, *args) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "message": "",
        "result": None,
    }
    TASK_QUEUE.put((job_id, func, args))
    return job_id

def get_job_status(job_id: str) -> dict:
    return JOBS.get(job_id, {"status": "not_found", "message": "Invalid job ID"})
```

**How it works**:  
- **`add_job`**: Creates a unique job ID, stores a job record in `JOBS`, then enqueues `(job_id, function, args)` onto `TASK_QUEUE`.  
- A **single worker thread** runs `worker_loop` which uses a **thread pool** or simply calls `func(*args)` in the same worker thread. (Here we do it directly, but you could also do `EXECUTOR.submit(func, *args)` to run in the pool. We keep it straightforward.)  
- `get_job_status` returns the dictionary for that job.

---

## 9. `app/utils/file_utils.py`

Utility to handle dynamic file path naming.

```python
import os
from ..configs.app_config import TEMP_DIR, STORAGE_DIR

def temp_file_path(stage: str, record_id: str, job_id: str, ext: str) -> str:
    """
    stage_{recordId or jobId}.{ext}
    """
    base = record_id if record_id else job_id
    filename = f"{stage}_{base}.{ext}"
    return os.path.join(TEMP_DIR, filename)

def storage_file_path(prefix: str, record_id: str, job_id: str, ext: str) -> str:
    """
    prefix_recordId or jobId.{ext}
    """
    base = record_id if record_id else job_id
    filename = f"{prefix}_{base}.{ext}"
    return os.path.join(STORAGE_DIR, filename)
```

---

## 10. `app/routes/video_routes.py`

Where we define our **FastAPI** endpoints. This file uses the `job_manager` to queue tasks, and provides a status route.

```python
from fastapi import APIRouter, HTTPException
from typing import Optional
from ..models.video_request import VideoRequest
from ..models.job_models import JobResponse, TranscriptionResponse
from ..services.job_manager import add_job, get_job_status
from ..services import ffmpeg_service, whisper_service
from ..services import ass_service
from ..utils.file_utils import temp_file_path, storage_file_path
import json
import requests

router = APIRouter(prefix="/v1/video")

@router.post("/caption", response_model=JobResponse)
def caption_video(req: VideoRequest):
    """
    Enqueue a job to generate a captioned video with embedded .ass subtitles.
    """
    def process_caption_job(job_id: str, request_data: VideoRequest):
        record_id = request_data.recordId or ""
        # Mark job as processing in manager
        job_info = get_job_status(job_id)
        job_info["status"] = "processing"

        # 1) Download or get local path
        from ..utils.file_utils import temp_file_path, storage_file_path
        from ..services.ffmpeg_service import extract_audio, add_subtitles_to_video
        from ..services.whisper_service import transcribe_audio
        from ..services.ass_service import convert_json_to_ass

        if request_data.video_path:
            video_path = request_data.video_path
        else:
            # Download
            import httpx
            video_path = temp_file_path("input", record_id, job_id, "mp4")
            with httpx.stream("GET", request_data.video_url, follow_redirects=True) as r:
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=8192):
                        f.write(chunk)

        # 2) Extract audio
        audio_path = temp_file_path("extract", record_id, job_id, "mp3")
        extract_audio(video_path, audio_path)

        # 3) Transcribe
        transcription = transcribe_audio(audio_path, request_data.language)
        transcript_json_path = temp_file_path("transcript", record_id, job_id, "json")
        with open(transcript_json_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2)

        # 4) Build ASS
        ass_data = convert_json_to_ass(transcription, request_data.ass_settings)
        ass_path = temp_file_path("ass", record_id, job_id, "ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_data)

        # 5) Add subtitles to video
        final_video_path = storage_file_path("captioned", record_id, job_id, "mp4")
        add_subtitles_to_video(video_path, ass_path, final_video_path)

        # Provide a shareable link (placeholder)
        download_url = f"http://localhost:8000/v1/video/share/{final_video_path.split('/')[-1]}"

        job_info["message"] = download_url

        # If webhook_url is present, POST final result
        if request_data.webhook_url:
            requests.post(
                request_data.webhook_url,
                json={
                    "jobId": job_id,
                    "recordId": record_id,
                    "status": "completed",
                    "download_url": download_url
                }
            )

    # Add job to the queue
    from ..services.job_manager import add_job
    job_id = add_job(process_caption_job, req, req)
    return JobResponse(
        jobId=job_id,
        recordId=req.recordId,
        status="queued",
        message="Caption job enqueued."
    )

@router.post("/transcribe", response_model=JobResponse)
def transcribe_video(req: VideoRequest):
    """
    Enqueue a job to transcribe the video (JSON with word-level timestamps).
    Only performs the 1) extract audio + 2) whisper transcribe steps.
    """
    def process_transcribe_job(job_id: str, request_data: VideoRequest):
        job_info = get_job_status(job_id)
        job_info["status"] = "processing"

        record_id = request_data.recordId or ""

        # 1) Download or get local path
        from ..utils.file_utils import temp_file_path
        from ..services.ffmpeg_service import extract_audio
        from ..services.whisper_service import transcribe_audio

        if request_data.video_path:
            video_path = request_data.video_path
        else:
            # Download
            import httpx
            video_path = temp_file_path("input", record_id, job_id, "mp4")
            with httpx.stream("GET", request_data.video_url, follow_redirects=True) as r:
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=8192):
                        f.write(chunk)

        # 2) Extract audio
        audio_path = temp_file_path("extract", record_id, job_id, "mp3")
        extract_audio(video_path, audio_path)

        # 3) Transcribe
        transcription = transcribe_audio(audio_path, request_data.language)
        job_info["result"] = transcription

        # If webhook_url is present, send results
        if request_data.webhook_url:
            import requests
            requests.post(
                request_data.webhook_url,
                json={
                    "jobId": job_id,
                    "recordId": record_id,
                    "status": "completed",
                    "transcription": transcription
                }
            )

    job_id = add_job(process_transcribe_job, req, req)
    return JobResponse(
        jobId=job_id,
        recordId=req.recordId,
        status="queued",
        message="Transcription job enqueued."
    )

@router.get("/status/{job_id}", response_model=JobResponse)
def get_job_status_endpoint(job_id: str):
    """
    Poll job status.
    """
    job_info = get_job_status(job_id)
    if job_info["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse(
        jobId=job_id,
        recordId="",
        status=job_info["status"],
        message=job_info["message"]
    )

@router.get("/transcriptresult/{job_id}", response_model=TranscriptionResponse)
def get_transcript_result(job_id: str):
    """
    After a transcribe job completes, retrieve the JSON transcripts.
    """
    job_info = get_job_status(job_id)
    if job_info["status"] == "completed":
        data = job_info.get("result")
        if not data:
            raise HTTPException(status_code=400, detail="No transcription data available.")
        # Build the response model
        return TranscriptionResponse(
            jobId=job_id,
            recordId="",
            status="completed",
            message=job_info["message"],
            segments=[
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "words": s.get("words", [])
                }
                for s in data["segments"]
            ]
        )
    elif job_info["status"] in ("queued", "processing"):
        raise HTTPException(status_code=400, detail="Job not completed yet.")
    else:
        raise HTTPException(status_code=400, detail="Job failed or not found.")

@router.get("/share/{filename}")
def share_file(filename: str):
    """
    Return a downloadable or shareable link for the final video.
    In production, you'd do actual file-serving or pre-signed URLs.
    """
    import os
    from fastapi.responses import JSONResponse
    from ..configs.app_config import STORAGE_DIR

    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    # Return a JSON with a pretend link
    return JSONResponse({"download_url": f"http://localhost:8000/v1/video/share/{filename}"})
```

Key points:
- `caption_video`: 5-step pipeline.  
- `transcribe_video`: 2-step pipeline.  
- We do a local function (`process_caption_job` or `process_transcribe_job`) that is queued with `add_job(...)`.
- We store intermediate results in `job_info["result"]` or `job_info["message"]`.
- Webhook support: If provided, we `POST` to that URL upon completion.

---

## 11. `app/main.py`

The entry point that creates the FastAPI app and includes our router.

```python
from fastapi import FastAPI
from .routes.video_routes import router as video_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Video Captioning and Transcription",
        version="1.0.0",
        description="Async local queue with concurrency for video captioning/transcription."
    )
    # Include routers
    app.include_router(video_router)
    return app

app = create_app()
```

---

## 12. `requirements.txt`

Minimal requirements (adjust versions as needed):

```txt
fastapi==0.95.2
uvicorn==0.22.0
pydantic==1.10.7
ffmpeg-python==0.2.0
torch>=1.9.0
whisper==1.0
httpx==0.24.0
```

> Also remember you need `ffmpeg` installed at the OS level.

---

## 13. Running Locally

1. **Install** the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run** FastAPI:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   This automatically starts the background `worker_loop` that processes jobs up to `MAX_WORKERS` concurrency.

3. **Test** with cURL or a REST client:

   - **Caption** a video:
     ```bash
     curl -X POST http://localhost:8000/v1/video/caption \
       -H "Content-Type: application/json" \
       -d '{
         "video_url": "https://example.com/video.mp4",
         "recordId": "123xyz",
         "language": "en"
       }'
     ```

   - **Check Status**:
     ```bash
     curl -X GET http://localhost:8000/v1/video/status/<jobId>
     ```

   - **Get Transcription** (if you used `/transcribe`):
     ```bash
     curl -X GET http://localhost:8000/v1/video/transcriptresult/<jobId>
     ```

---

## Highlights & Next Steps

1. **Single Responsibility Principle**:
   - Separate modules for config, routes, models, services, and utility code.
2. **Local Queue**:
   - Simple in-memory `Queue` + a background thread that processes tasks using concurrency up to `MAX_WORKERS`.
   - Sufficient for a single-machine scenario.
3. **CUDA**:
   - The code checks for `torch.cuda.is_available()` and loads Whisper on GPU if present.
4. **Whisper Model by Environment**:
   - `WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")`
   - e.g., set `WHISPER_MODEL=small` or `large-v2` to load different models.
5. **JSON I/O**:
   - All endpoints produce and consume JSON.  
6. **Webhook Support**:
   - If `webhook_url` is provided, an HTTP POST with final results is sent upon completion. Otherwise, clients can poll the status endpoint.

> In a real production environment, you might add:

- **Persistent storage** for job states (database).
- **File-serving** with robust security or short-lived signed URLs (S3, local server, Nginx).
- **Proper error handling** and logging (e.g., Sentry).
- **Cleanup** of old temp files (cron job or auto-clean policy).
- **Authentication** & **authorization** to protect your APIs.

This completes a **single-machine** asynchronous processing system for video **captioning** and **transcription** using **FastAPI**, **FFmpeg**, and **OpenAI’s Whisper**.