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
