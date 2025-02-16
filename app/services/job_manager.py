import threading
import uuid
import time
import logging                     # added
from typing import Callable, Any, Dict
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from ..configs.app_config import MAX_WORKERS

logger = logging.getLogger(__name__)  # added logger

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
            logger.info(f"Job {job_id} started processing using function: {func.__name__}")  # added log
            try:
                func(job_id, *args)
                # If everything is fine, status is completed
                if job_data["status"] != "failed":  # in case it was set to failed inside func
                    job_data["status"] = "completed"
                logger.info(f"Job {job_id} completed successfully with status: {job_data['status']}")  # added log
            except Exception as e:
                job_data["status"] = "failed"
                job_data["message"] = str(e)
                logger.error(f"Job {job_id} failed with error: {str(e)}")  # added log
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
    logger.info(f"Job {job_id} added to the queue with function: {func.__name__}")  # added log
    return job_id

def get_job_status(job_id: str) -> dict:
    return JOBS.get(job_id, {"status": "not_found", "message": "Invalid job ID"})
