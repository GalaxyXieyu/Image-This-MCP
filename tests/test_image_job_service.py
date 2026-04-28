from pathlib import Path

from image_this_mcp.services.image_job_service import ImageJobService


def test_image_job_service_submit_and_get(monkeypatch, tmp_path):
    monkeypatch.setattr(ImageJobService, "_worker_loop", lambda self: None)
    service = ImageJobService(db_path=str(tmp_path / "jobs.db"))
    try:
        job_id = service.submit_job({"prompt": "test prompt", "provider": "openai", "mode": "generate"})
        job = service.get_job(job_id)
        assert job is not None
        assert job["job_id"] == job_id
        assert job["status"] == "queued"
        assert job["payload"]["prompt"] == "test prompt"
    finally:
        service.stop()
