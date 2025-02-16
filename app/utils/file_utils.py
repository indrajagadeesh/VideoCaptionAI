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
