from typing import Dict

from pydantic import BaseModel


class JobSubmission(BaseModel):
    blend_file: str
    frame_start: int
    frame_end: int
    output_root: str
    project: str
    user_email: str
    user_name: str
    chunk_size: int = 3
    format: str = "PNG"
    fps: int = 24
    has_previews: bool = False
    image_file_extension: str = ".png"
    priority: int = 50
    submitter_platform: str = "linux"
    job_type: str = "simple-blender-render"


def prepare_payload(job: JobSubmission) -> Dict[str, any]:
    return {
        "metadata": {
            "project": job.project,
            "user.email": job.user_email,
            "user.name": job.user_name
        },
        "name": "Blender Render Job",
        "priority": job.priority,
        "settings": {
            "blendfile": job.blend_file,
            "chunk_size": job.chunk_size,
            "format": job.format,
            "fps": job.fps,
            "frames": f"{job.frame_start}-{job.frame_end}",
            "has_previews": job.has_previews,
            "image_file_extension": job.image_file_extension,
            "render_output_path": f"{job.output_path}/output",
            "render_output_root": job.output_path
        },
        "submitter_platform": job.submitter_platform,
        "type": job.job_type
    }