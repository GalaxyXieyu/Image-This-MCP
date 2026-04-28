"""Artifact publishing service for remote image delivery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import logging
import mimetypes
from pathlib import Path
import time
import uuid
from typing import Any, Dict, Optional

from minio import Minio

from ..config.settings import MinioConfig


@dataclass
class PublishedArtifact:
    """Information about a published remote artifact."""

    object_name: str
    bucket: str
    artifact_url: str
    content_type: str


class ArtifactService:
    """Uploads generated artifacts to MinIO and returns client-consumable URLs."""

    def __init__(self, config: MinioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client: Optional[Minio] = None

    @property
    def client(self) -> Minio:
        """Lazy init MinIO client."""
        if self._client is None:
            self._client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region,
            )
        return self._client

    def validate_config(self) -> bool:
        """Check whether artifact publishing is configured."""
        return self.config.validate_credentials()

    def ensure_bucket(self) -> None:
        """Create the target bucket if it does not exist."""
        if not self.client.bucket_exists(self.config.bucket):
            self.client.make_bucket(self.config.bucket)
            self.logger.info(f"Created MinIO bucket: {self.config.bucket}")

    def publish_file(
        self,
        local_path: str,
        content_type: Optional[str] = None,
    ) -> PublishedArtifact:
        """Upload a local file and return a stable artifact URL."""
        if not self.validate_config():
            raise ValueError("ArtifactService is not configured")

        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(local_path)

        self.ensure_bucket()

        guessed_type, _ = mimetypes.guess_type(path.name)
        effective_type = content_type or guessed_type or "application/octet-stream"
        object_name = self._build_object_name(path)

        self.client.fput_object(
            bucket_name=self.config.bucket,
            object_name=object_name,
            file_path=str(path),
            content_type=effective_type,
        )

        artifact_url = self._build_artifact_url(object_name)
        self.logger.info(f"Published artifact to MinIO: {artifact_url}")
        return PublishedArtifact(
            object_name=object_name,
            bucket=self.config.bucket,
            artifact_url=artifact_url,
            content_type=effective_type,
        )

    def _build_object_name(self, path: Path) -> str:
        """Create a collision-resistant object key."""
        timestamp = time.strftime("%Y/%m/%d")
        suffix = uuid.uuid4().hex[:12]
        prefix = self.config.key_prefix.strip("/ ")
        return f"{prefix}/{timestamp}/{suffix}-{path.name}" if prefix else f"{timestamp}/{suffix}-{path.name}"

    def _build_artifact_url(self, object_name: str) -> str:
        """Prefer stable public URL, fallback to a long-lived presigned URL."""
        if self.config.public_base_url:
            return f"{self.config.public_base_url.rstrip('/')}/{self.config.bucket}/{object_name}"
        return self.client.presigned_get_object(
            self.config.bucket,
            object_name,
            expires=timedelta(seconds=self.config.presign_expiry_seconds),
        )

    def build_metadata(self, published: PublishedArtifact) -> Dict[str, Any]:
        """Convert published artifact info to response metadata."""
        return {
            "artifact_url": published.artifact_url,
            "artifact_bucket": published.bucket,
            "artifact_object_name": published.object_name,
            "artifact_content_type": published.content_type,
            "storage_provider": "minio",
        }
