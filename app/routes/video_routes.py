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
        ass_data = convert_json_to_ass(transcription, request_data.ass_settings, request_data.subtitle_style)
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
    job_id = add_job(process_caption_job, req)
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

    job_id = add_job(process_transcribe_job, req)
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
