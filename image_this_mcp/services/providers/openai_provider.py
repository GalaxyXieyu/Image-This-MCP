"""
OpenAI image generation provider (DALL-E 3, gpt-image-2, etc.).

Calls the OpenAI Images API (POST /v1/images/generations) via a configurable
base URL (e.g. https://yunwu.ai or https://api.openai.com).
"""

import base64
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import httpx
from fastmcp.utilities.types import Image as MCPImage

from .base import BaseImageProvider
from ...config.settings import OpenAIConfig
from ...core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseImageProvider):
    """
    OpenAI image generation provider.

    Supports any OpenAI-compatible images API endpoint, including:
    - OpenAI official API (api.openai.com)
    - Third-party proxies (e.g. yunwu.ai)
    - Self-hosted compatible gateways
    """

    provider_name = "openai"
    provider_version = "1.0.0"

    # Supported aspect ratio -> size mappings
    ASPECT_RATIO_MAP = {
        "1:1": "1024x1024",
        "3:4": "1024x1792",
        "4:3": "1792x1024",
        "9:16": "1024x1792",
        "16:9": "1792x1024",
        "2:3": "1024x1792",
        "3:2": "1792x1024",
    }

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy init of persistent HTTP client."""
        if self._client is None:
            timeout = httpx.Timeout(self.config.request_timeout, connect=10.0)
            self._client = httpx.Client(timeout=timeout)
        return self._client

    def generate_images(
        self,
        prompt: str,
        n: int = 1,
        negative_prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[MCPImage], List[Dict[str, Any]]]:
        """
        Generate images via OpenAI Images API.

        Args:
            prompt: Text description of desired image.
            n: Number of images (OpenAI DALL-E 3 supports n=1 only).
            negative_prompt: Ignored by OpenAI Images API.
            aspect_ratio: Maps to OpenAI size strings.
            **kwargs: Extra params (quality, style, model, size, etc.).

        Returns:
            Tuple of (images, metadata).
        """
        if not self.validate_config():
            raise ValidationError("OpenAI provider credentials not configured")

        # Resolve size from aspect_ratio or kwargs
        size = kwargs.get("size")
        if not size and aspect_ratio:
            size = self._map_aspect_ratio_to_size(aspect_ratio)
        if not size:
            size = self.config.default_size

        # Resolve model, quality, style
        model = kwargs.get("model") or self.config.default_model
        quality = kwargs.get("quality") or self.config.default_quality
        style = kwargs.get("style") or self.config.default_style

        # OpenAI DALL-E 3 supports n=1 only; gpt-image-2 may support more.
        # We loop for compatibility.
        all_images: List[MCPImage] = []
        all_metadata: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None

        for i in range(n):
            try:
                req_body = self._build_request(
                    prompt=prompt,
                    model=model,
                    size=size,
                    quality=quality,
                    style=style,
                    n=1,
                )
                self.logger.info(
                    f"Generating image {i + 1}/{n} with OpenAI model={model} size={size}"
                )

                response = self._make_request_with_retry(req_body)
                data_list = response.get("data", [])
                if not data_list:
                    raise ValueError("Empty response from OpenAI Images API")

                item = data_list[0]
                image_bytes = None
                if item.get("b64_json"):
                    image_bytes = base64.b64decode(item["b64_json"])
                elif item.get("url"):
                    image_bytes = self._fetch_image_from_url(item["url"])

                if not image_bytes:
                    raise ValueError("No image data in OpenAI response")

                mcp_image = MCPImage(data=image_bytes, format="png")
                all_images.append(mcp_image)

                metadata = {
                    "provider": "openai",
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "index": i + 1,
                    "mime_type": "image/png",
                    "revised_prompt": item.get("revised_prompt"),
                    "created": response.get("created"),
                }
                all_metadata.append(metadata)
                self.logger.info(
                    f"OpenAI image {i + 1}/{n} generated ({len(image_bytes)} bytes)"
                )

            except Exception as e:
                last_error = e
                self.logger.error(f"Failed to generate OpenAI image {i + 1}/{n}: {e}")
                continue

        if not all_images:
            raise last_error or ValueError("OpenAI Images API returned no images")

        return all_images, all_metadata

    def edit_image(
        self,
        instruction: str,
        image_data: bytes,
        mime_type: str = "image/png",
        **kwargs,
    ) -> Tuple[List[MCPImage], int]:
        """
        Image editing is not supported by the standard OpenAI Images API.

        Returns NotImplementedError with a helpful message.
        """
        raise NotImplementedError(
            "OpenAI image editing is not supported via the Images API. "
            "Use provider='gemini' for image editing."
        )

    def validate_config(self) -> bool:
        """Validate that API key and base URL are present."""
        return bool(self.config.api_key and self.config.base_url)

    def _map_aspect_ratio_to_size(self, aspect_ratio: str) -> str:
        """Map aspect ratio to OpenAI size string."""
        return self.ASPECT_RATIO_MAP.get(aspect_ratio, self.config.default_size)

    def _build_request(
        self,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        style: str,
        n: int = 1,
    ) -> Dict[str, Any]:
        """Build the JSON body for /v1/images/generations."""
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }
        # Only include quality/style if the model supports them
        if quality:
            body["quality"] = quality
        if style:
            body["style"] = style
        return body

    def _make_request_with_retry(self, req_body: Dict[str, Any]) -> Dict[str, Any]:
        """POST to /v1/images/generations with exponential backoff retries."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        url = f"{self.config.base_url.rstrip('/')}/v1/images/generations"

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay * (2 ** (attempt - 1))
                    self.logger.info(
                        f"OpenAI retry {attempt + 1}/{self.config.max_retries} after {delay}s"
                    )
                    time.sleep(delay)

                response = self.client.post(url, headers=headers, json=req_body)
                response.raise_for_status()
                result = response.json()

                if result.get("error"):
                    msg = result["error"].get("message", "Unknown error")
                    raise ValueError(f"OpenAI API error: {msg}")

                return result

            except httpx.HTTPStatusError as e:
                last_error = e
                self.logger.warning(f"OpenAI HTTP error attempt {attempt + 1}: {e}")
                if e.response.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.config.max_retries - 1:
                        continue
                break

            except Exception as e:
                last_error = e
                self.logger.warning(f"OpenAI error attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    continue
                break

        raise last_error or RuntimeError("All OpenAI retry attempts failed")

    def _fetch_image_from_url(self, url: str, timeout: int = 60) -> bytes:
        """Fetch image bytes from a URL."""
        try:
            resp = self.client.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            self.logger.error(f"Failed to fetch image from URL: {e}")
            raise

    def _get_capabilities(self) -> Dict[str, Any]:
        """Return OpenAI provider capabilities."""
        return {
            "max_images_per_request": self.config.max_images_per_request,
            "supported_aspect_ratios": list(self.ASPECT_RATIO_MAP.keys()),
            "supported_sizes": list(set(self.ASPECT_RATIO_MAP.values())),
            "supports_editing": False,
            "supports_reference_images": False,
            "max_resolution": "1792",
            "default_size": self.config.default_size,
            "response_formats": ["b64_json", "url"],
            "retry_config": {
                "max_retries": self.config.max_retries,
                "retry_delay": self.config.retry_delay,
            },
            "features": {
                "quality_control": True,
                "style_control": True,
                "revised_prompt": True,
            },
        }

    def __del__(self):
        """Cleanup HTTP client."""
        if self._client:
            self._client.close()