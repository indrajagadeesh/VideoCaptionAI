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
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")

# Check for CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy load the Whisper model
# (Alternatively, load on first use in whisper_service)

print(f"Loading Whisper model {WHISPER_MODEL_NAME} on {DEVICE}")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME).to(DEVICE)
