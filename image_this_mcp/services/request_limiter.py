"""Global concurrency controls for provider-backed requests."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import threading
import time
from typing import Dict, Iterator

from ..config.settings import ServerConfig
from ..core.exceptions import ValidationError


class RequestLimiter:
    """Process-wide semaphore-based throttling for image generation requests."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._global = threading.BoundedSemaphore(max(1, config.global_max_concurrent_requests))
        self._provider_limits: Dict[str, threading.BoundedSemaphore] = {
            "gemini": threading.BoundedSemaphore(max(1, config.gemini_max_concurrent_requests)),
            "openai": threading.BoundedSemaphore(max(1, config.openai_max_concurrent_requests)),
        }

    @contextmanager
    def limit(self, provider: str) -> Iterator[None]:
        """Acquire global and provider-specific capacity for a request."""
        started = time.time()
        provider_sem = self._provider_limits.get(provider)

        if not self._global.acquire(timeout=self.config.queue_wait_timeout_seconds):
            raise ValidationError("Timed out waiting for global request capacity")

        provider_acquired = False
        try:
            if provider_sem is not None:
                provider_acquired = provider_sem.acquire(timeout=self.config.queue_wait_timeout_seconds)
                if not provider_acquired:
                    raise ValidationError(f"Timed out waiting for {provider} request capacity")

            wait_seconds = round(time.time() - started, 3)
            if wait_seconds > 0:
                self.logger.info(f"Queued {provider} request for {wait_seconds}s before execution")
            yield
        finally:
            if provider_sem is not None and provider_acquired:
                provider_sem.release()
            self._global.release()
