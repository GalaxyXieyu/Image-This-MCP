"""Asynchronous image job queue for remote deployments."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, Optional

from ..core.exceptions import ValidationError
from ..models import ModelRegistry
from .. import services
from ..config.settings import ModelTier
from ..tools.generate_image.gemini_handler import handle_gemini_request
from ..tools.generate_image.jimeng_handler import handle_jimeng_request
from ..tools.generate_image.openai_handler import handle_openai_request


class ImageJobService:
    """SQLite-backed background queue for async image generation jobs."""

    def __init__(self, db_path: str, poll_interval_seconds: int = 2):
        self.db_path = db_path
        self.poll_interval_seconds = poll_interval_seconds
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="image-job-worker")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._worker_thread.start()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_jobs_status_created_at ON image_jobs(status, created_at)")
            conn.commit()

    def submit_job(self, payload: Dict[str, Any]) -> str:
        """Insert a new queued job."""
        self._validate_payload(payload)
        now = time.time()
        job_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_jobs (job_id, status, payload, created_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (job_id, json.dumps(payload), now, now),
            )
            conn.commit()
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM image_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def list_jobs(self, status: Optional[str] = None, limit: int = 20) -> list[Dict[str, Any]]:
        query = "SELECT * FROM image_jobs"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        params += (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def stop(self):
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            job = self._claim_next_job()
            if not job:
                time.sleep(self.poll_interval_seconds)
                continue
            self._execute_job(job)

    def _claim_next_job(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM image_jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            now = time.time()
            updated = conn.execute(
                """
                UPDATE image_jobs
                SET status = 'running', started_at = ?, updated_at = ?
                WHERE job_id = ? AND status = 'queued'
                """,
                (now, now, row["job_id"]),
            )
            conn.commit()
            if updated.rowcount == 0:
                return None
            claimed = dict(row)
            claimed["status"] = "running"
            return claimed

    def _execute_job(self, job: Dict[str, Any]):
        payload = json.loads(job["payload"])
        try:
            result = self._run_generate(payload)
            now = time.time()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE image_jobs
                    SET status = 'completed', result = ?, updated_at = ?, completed_at = ?
                    WHERE job_id = ?
                    """,
                    (json.dumps(result), now, now, job["job_id"]),
                )
                conn.commit()
        except Exception as exc:
            self.logger.error(f"Async image job failed {job['job_id']}: {exc}")
            now = time.time()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE image_jobs
                    SET status = 'failed', error = ?, updated_at = ?, completed_at = ?
                    WHERE job_id = ?
                    """,
                    (str(exc), now, now, job["job_id"]),
                )
                conn.commit()

    def _validate_payload(self, payload: Dict[str, Any]):
        prompt = payload.get("prompt", "").strip()
        if not prompt:
            raise ValidationError("Async job requires a non-empty prompt")
        if payload.get("mode") not in (None, "generate", "auto"):
            raise ValidationError("Async queue currently supports generate mode only")

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "payload": json.loads(row["payload"]),
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }

    def _resolve_provider_and_model(self, payload: Dict[str, Any]) -> tuple[str, Optional[str]]:
        provider = (payload.get("provider") or "auto").lower()
        model = payload.get("model")

        if provider == "auto":
            env_provider = os.getenv("IMAGE_PROVIDER")
            if env_provider:
                provider = env_provider.lower()
            else:
                available = services.list_initialized_providers()
                provider = available[0] if available else "gemini"

        if model:
            selected_model = ModelRegistry.get(model)
            if not selected_model:
                raise ValidationError(f"Unknown model '{model}'. Use list_models first.")
            model_provider = selected_model.provider
            if provider == "jimeng" and model_provider in {"jimeng", "jimeng45"}:
                provider = "jimeng45" if model_provider == "jimeng45" else "jimeng"
            elif provider in {"auto", model_provider}:
                provider = model_provider
            elif provider == "openai" and model_provider == "openai":
                provider = "openai"
            elif provider == "gemini" and model_provider == "gemini":
                provider = "gemini"
            else:
                raise ValidationError(
                    f"Model '{model}' belongs to provider '{model_provider}', not '{provider}'."
                )
        return provider, model

    def _run_generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        provider, model = self._resolve_provider_and_model(payload)
        limiter = services.get_request_limiter()

        with limiter.limit(provider):
            if provider == "gemini":
                _thumbs, metadata, selected_tier, model_info = handle_gemini_request(
                    prompt=payload["prompt"],
                    n=payload.get("n", 1),
                    negative_prompt=payload.get("negative_prompt"),
                    system_instruction=payload.get("system_instruction"),
                    input_image_paths=None,
                    file_id=None,
                    aspect_ratio=payload.get("aspect_ratio"),
                    model_tier=payload.get("model_tier", ModelTier.FLASH.value),
                    thinking_level=payload.get("thinking_level", "high"),
                    resolution=payload.get("resolution", "high"),
                    enable_grounding=payload.get("enable_grounding", True),
                    output_dir=payload.get("output_dir"),
                    detected_mode="generate",
                )
                return {
                    "provider": "gemini",
                    "model_id": model_info["model_id"],
                    "model_tier": selected_tier,
                    "images": metadata,
                    "returned": len(metadata),
                }

            if provider == "openai":
                _thumbs, metadata = handle_openai_request(
                    prompt=payload["prompt"],
                    n=payload.get("n", 1),
                    negative_prompt=payload.get("negative_prompt"),
                    input_image_paths=None,
                    file_id=None,
                    aspect_ratio=payload.get("aspect_ratio"),
                    output_dir=payload.get("output_dir"),
                    detected_mode="generate",
                    model=model,
                )
                return {
                    "provider": "openai",
                    "model_id": model or (metadata[0].get("model") if metadata else None),
                    "images": metadata,
                    "returned": len(metadata),
                }

            _thumbs, metadata = handle_jimeng_request(
                prompt=payload["prompt"],
                n=payload.get("n", 1),
                negative_prompt=payload.get("negative_prompt"),
                input_image_paths=None,
                file_id=None,
                aspect_ratio=payload.get("aspect_ratio"),
                output_dir=payload.get("output_dir"),
                detected_mode="generate",
                provider_name=provider,
                model=model,
            )
            return {
                "provider": "jimeng",
                "model_id": model or (metadata[0].get("model") if metadata else None),
                "images": metadata,
                "returned": len(metadata),
            }
