"""
Jimeng 4.5 (Seedream 4.5 via Ark API) image generation provider.

This provider integrates with Volcengine's Ark API for Jimeng 4.5 (Seedream 4.5),
supporting text-to-image generation with reference images, improved quality,
and larger output resolutions.
"""

import base64
import logging
import threading
import time
from typing import List, Tuple, Dict, Any, Optional

import httpx

from .base import BaseImageProvider
from fastmcp.utilities.types import Image as MCPImage
from ...config.settings import Jimeng45Config
from ...core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class Jimeng45Provider(BaseImageProvider):
    """
    Jimeng 4.5 (Seedream 4.5) image generation provider.

    Features:
    - Text-to-image generation via Ark API
    - Support for 1-14 reference images
    - Bearer Token authentication (simpler than HMAC-SHA256)
    - Larger output resolutions (up to 16M pixels)
    - Usage tracking for billing
    - Automatic retries with exponential backoff
    - Serial request queue to avoid rate limiting
    """

    provider_name = "jimeng45"
    provider_version = "1.0.0"

    def __init__(self, config: Jimeng45Config):
        """
        Initialize Jimeng 4.5 provider.

        Args:
            config: Jimeng45Config instance with credentials and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Serial queue lock to avoid rate limiting
        self._lock = threading.Lock()
        self._request_queue = []

        # HTTP client (persistent connection)
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy initialization of HTTP client."""
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
        reference_images: Optional[List[str]] = None,
        size: Optional[str] = None,
        watermark: Optional[bool] = None,
        **kwargs
    ) -> Tuple[List[MCPImage], List[Dict[str, Any]]]:
        """
        Generate images using Jimeng 4.5 (Seedream 4.5).

        Args:
            prompt: Text description of desired image
            n: Number of images to generate (Jimeng 4.5 typically returns 1 per request)
            negative_prompt: Text describing what to avoid (may be ignored by API)
            aspect_ratio: Aspect ratio hint (e.g., "3:4", "16:9") - maps to size
            reference_images: List of image URLs/base64 for reference (1-14 images)
            size: Explicit size string (e.g., "1728x2304"), overrides aspect_ratio
            watermark: Whether to add watermark (overrides config default)
            **kwargs: Additional provider-specific parameters

        Returns:
            Tuple of (images, metadata)
        """
        if not self.validate_config():
            raise ValidationError("Jimeng 4.5 provider credentials not configured")

        # Map aspect_ratio to size if not explicitly provided
        if size is None and aspect_ratio:
            size = self._map_aspect_ratio_to_size(aspect_ratio)

        # Use serial queue to avoid rate limiting
        with self._lock:
            return self._generate_images_serial(
                prompt=prompt,
                n=n,
                negative_prompt=negative_prompt,
                size=size,
                reference_images=reference_images,
                watermark=watermark,
                **kwargs
            )

    def _generate_images_serial(
        self,
        prompt: str,
        n: int,
        negative_prompt: Optional[str],
        size: Optional[str],
        reference_images: Optional[List[str]],
        watermark: Optional[bool],
        **kwargs
    ) -> Tuple[List[MCPImage], List[Dict[str, Any]]]:
        """
        Internal method for serial image generation.

        This method is called within a lock to ensure requests are serialized.
        """
        all_images = []
        all_metadata = []
        last_error = None

        # Jimeng 4.5 typically generates 1 image per request, so we loop
        for i in range(n):
            try:
                # Build request body
                req_body = self._build_generation_request(
                    prompt=prompt,
                    size=size,
                    reference_images=reference_images,
                    watermark=watermark
                )

                self.logger.info(
                    f"Generating image {i + 1}/{n} with Jimeng 4.5 (Seedream 4.5)"
                )

                # Make API request with retry logic
                response = self._make_request_with_retry(req_body)

                # Extract image from response
                data_list = response.get("data", [])
                if not data_list:
                    raise ValueError("Empty response from Jimeng 4.5 API")

                # Find first valid image result
                item = next(
                    (entry for entry in data_list
                     if entry.get("b64_json") or entry.get("url") or entry.get("error")),
                    data_list[0]
                )

                # Check for image-level error
                if item.get("error"):
                    error_code = item["error"].get("code", "unknown")
                    error_msg = item["error"].get("message", "Unknown error")
                    raise ValueError(
                        f"Jimeng 4.5 generation error: code={error_code}, msg={error_msg}"
                    )

                # Get image data
                image_bytes = None
                if item.get("b64_json"):
                    image_bytes = base64.b64decode(item["b64_json"])
                elif item.get("url"):
                    # Fetch image from URL
                    image_bytes = self._fetch_image_from_url(item["url"])

                if not image_bytes:
                    raise ValueError("No image data in response from Jimeng 4.5 API")

                # Create MCP Image object
                mcp_image = MCPImage(data=image_bytes, format="jpeg")
                all_images.append(mcp_image)

                # Metadata
                metadata = {
                    "provider": "jimeng45",
                    "model": response.get("model", self.config.model),
                    "prompt": prompt,
                    "size": item.get("size", size or self.config.default_size),
                    "index": i + 1,
                    "mime_type": "image/jpeg",
                    "watermark": watermark if watermark is not None else self.config.watermark,
                    "usage": response.get("usage"),
                    "created": response.get("created"),
                }
                all_metadata.append(metadata)

                self.logger.info(
                    f"Successfully generated image {i + 1}/{n} "
                    f"({len(image_bytes)} bytes)"
                )

            except Exception as e:
                last_error = e
                self.logger.error(f"Failed to generate image {i + 1}/{n}: {e}")
                # Continue with next image rather than failing completely
                continue

        if not all_images:
            raise last_error or ValueError("Jimeng 4.5 API returned no images")

        return all_images, all_metadata

    def edit_image(
        self,
        instruction: str,
        image_data: bytes,
        mime_type: str = "image/jpeg",
        **kwargs
    ) -> Tuple[List[MCPImage], int]:
        """
        Edit an image using Jimeng 4.5.

        Jimeng 4.5 supports image editing through reference images.
        We convert the source image to base64 and pass it as a reference.

        Args:
            instruction: Natural language editing instruction
            image_data: Source image bytes
            mime_type: MIME type of source image
            **kwargs: Additional parameters

        Returns:
            Tuple of (edited_images, count)
        """
        self.logger.info(
            "Jimeng 4.5 edit_image uses reference image approach"
        )

        # Convert source image to base64
        base64_prefix = "data:image/jpeg;base64,"
        if mime_type == "image/png":
            base64_prefix = "data:image/png;base64,"

        image_base64 = base64.b64encode(image_data).decode("utf-8")
        reference_image = f"{base64_prefix}{image_base64}"

        # Generate new image with reference
        images, metadata = self.generate_images(
            prompt=instruction,
            n=1,
            reference_images=[reference_image],
        )

        return images, len(images)

    def validate_config(self) -> bool:
        """Validate Jimeng 4.5 configuration."""
        return self.config.validate_credentials()

    def _map_aspect_ratio_to_size(self, aspect_ratio: str) -> str:
        """
        Map aspect ratio string to size string.

        Args:
            aspect_ratio: Aspect ratio like "3:4", "16:9", "1:1"

        Returns:
            Size string like "1728x2304"
        """
        ratio_map = {
            "3:4": "1728x2304",   # Xiaohongshu optimized
            "4:3": "2304x1728",
            "1:1": "1024x1024",
            "16:9": "1920x1080",
            "9:16": "1080x1920",
            "2:3": "1024x1536",
        }
        return ratio_map.get(aspect_ratio, self.config.default_size)

    def _build_generation_request(
        self,
        prompt: str,
        size: Optional[str] = None,
        reference_images: Optional[List[str]] = None,
        watermark: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Build request body for Jimeng 4.5 (Ark API).

        Args:
            prompt: Generation prompt
            size: Image size string
            reference_images: List of reference image URLs/base64
            watermark: Whether to add watermark

        Returns:
            Request body dict
        """
        req_body = {
            "model": self.config.model,
            "prompt": prompt,
            "response_format": self.config.response_format,
            "sequential_image_generation": self.config.sequential_image_generation,
            "size": size or self.config.default_size,
            "stream": False,
            "watermark": watermark if watermark is not None else self.config.watermark,
        }

        # Add reference images if provided (Jimeng 4.5 supports 1-14 images)
        if reference_images:
            normalized_images = self._normalize_reference_images(reference_images)
            if len(normalized_images) == 1:
                req_body["image"] = normalized_images[0]
            else:
                req_body["image"] = normalized_images

        return req_body

    def _normalize_reference_images(self, images: List[str]) -> List[str]:
        """
        Normalize reference images to the format expected by Jimeng 4.5 API.

        Jimeng 4.5 accepts:
        - HTTP/HTTPS URLs
        - Base64 data URIs (data:image/...;base64,...)

        Args:
            images: List of image URLs or base64 strings

        Returns:
            Normalized list of image references
        """
        normalized = []
        for img in images:
            img_str = str(img).strip()
            if not img_str:
                continue

            # If already a data URI or URL, use as-is
            if img_str.startswith(("http://", "https://", "data:image/")):
                normalized.append(img_str)
            else:
                # Assume it's raw base64, add data URI prefix
                normalized.append(f"data:image/jpeg;base64,{img_str}")

        return normalized

    def _make_request_with_retry(
        self,
        req_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic.

        Implements exponential backoff for retries.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay * (2 ** (attempt - 1))
                    self.logger.info(
                        f"Retry attempt {attempt + 1}/{self.config.max_retries} "
                        f"after {delay}s delay"
                    )
                    time.sleep(delay)

                # Make request
                response = self.client.post(
                    self.config.api_endpoint,
                    headers=headers,
                    json=req_body
                )
                response.raise_for_status()

                result = response.json()

                # Check for API-level errors
                if result.get("error"):
                    error_msg = result["error"].get("message", "Unknown error")
                    raise ValueError(f"Jimeng 4.5 API error: {error_msg}")

                return result

            except httpx.HTTPStatusError as e:
                last_error = e
                self.logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")

                # Check if error is retryable
                if e.response.status_code in [429, 500, 502, 503, 504]:
                    if attempt < self.config.max_retries - 1:
                        continue  # Retry
                break  # Non-retryable error

            except Exception as e:
                last_error = e
                self.logger.warning(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    continue  # Retry
                break  # Final attempt failed

        # All retries exhausted
        raise last_error or RuntimeError("All retry attempts failed for Jimeng 4.5")

    def _fetch_image_from_url(self, url: str, timeout: int = 60) -> bytes:
        """
        Fetch image from URL (fallback when b64_json is not available).

        Args:
            url: Image URL
            timeout: Request timeout in seconds

        Returns:
            Image bytes
        """
        try:
            response = self.client.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            self.logger.error(f"Failed to fetch image from URL: {e}")
            raise

    def _get_capabilities(self) -> Dict[str, Any]:
        """Get Jimeng 4.5 provider capabilities."""
        return {
            "max_images_per_request": 1,
            "supported_aspect_ratios": ["3:4", "4:3", "1:1", "16:9", "9:16", "2:3"],
            "supported_sizes": self.config.SUPPORTED_SIZES,
            "supports_editing": True,
            "supports_reference_images": True,
            "max_reference_images": 14,  # Jimeng 4.5 supports up to 14 reference images
            "max_resolution": "4096x4096",  # Up to ~16M pixels
            "default_resolution": self.config.default_size,
            "response_formats": ["b64_json", "url"],
            "retry_config": {
                "max_retries": self.config.max_retries,
                "retry_delay": self.config.retry_delay,
            },
            "features": {
                "watermark": self.config.watermark,
                "usage_tracking": True,
                "custom_endpoints": True,  # Supports custom endpoint IDs
            }
        }

    def __del__(self):
        """Cleanup HTTP client on deletion."""
        if self._client:
            self._client.close()
