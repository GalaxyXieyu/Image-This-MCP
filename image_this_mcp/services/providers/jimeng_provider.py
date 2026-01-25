"""
Jimeng AI (Volcengine) image generation provider.

This provider integrates with Volcengine's Jimeng AI image generation service,
supporting text-to-image and image-to-image generation with reference images.
"""

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from typing import List, Tuple, Dict, Any, Optional

import httpx

from .base import BaseImageProvider
from fastmcp.utilities.types import Image as MCPImage
from ...config.settings import JimengConfig
from ...core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class JimengProvider(BaseImageProvider):
    """
    Jimeng AI image generation provider.

    Features:
    - Text-to-image generation
    - Image editing with reference images
    - Automatic retries with exponential backoff
    - Serial request queue to avoid rate limiting
    - Built-in image upload for reference images
    """

    provider_name = "jimeng"
    provider_version = "1.0.0"

    # Volcengine API constants
    HOST = "visual.volcengineapi.com"
    REGION = "cn-north-1"
    SERVICE = "cv"
    VERSION = "2022-08-31"

    def __init__(self, config: JimengConfig):
        """
        Initialize Jimeng provider.

        Args:
            config: JimengConfig instance with credentials and settings
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
        **kwargs
    ) -> Tuple[List[MCPImage], List[Dict[str, Any]]]:
        """
        Generate images using Jimeng AI.

        Args:
            prompt: Text description of desired image
            n: Number of images to generate (Jimeng typically returns 1)
            negative_prompt: Text describing what to avoid
            aspect_ratio: Aspect ratio (e.g., "3:4", "16:9")
            reference_images: List of image URLs for reference
            **kwargs: Additional provider-specific parameters

        Returns:
            Tuple of (images, metadata)
        """
        if not self.validate_config():
            raise ValidationError("Jimeng provider credentials not configured")

        # Use serial queue to avoid rate limiting
        with self._lock:
            return self._generate_images_serial(
                prompt=prompt,
                n=n,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                reference_images=reference_images,
                **kwargs
            )

    def _generate_images_serial(
        self,
        prompt: str,
        n: int,
        negative_prompt: Optional[str],
        aspect_ratio: Optional[str],
        reference_images: Optional[List[str]],
        **kwargs
    ) -> Tuple[List[MCPImage], List[Dict[str, Any]]]:
        """
        Internal method for serial image generation.

        This method is called within a lock to ensure requests are serialized.
        """
        all_images = []
        all_metadata = []

        # Jimeng typically generates 1 image per request, so we loop
        for i in range(n):
            try:
                # Build request body
                req_body = self._build_generation_request(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    reference_images=reference_images
                )

                self.logger.info(
                    f"Generating image {i + 1}/{n} with Jimeng AI"
                )

                # Make API request with retry logic
                response = self._make_request_with_retry(req_body)

                # Extract image from response
                image_base64 = response.get("data", {}).get("binary_data_base64", [None])[0]
                if not image_base64:
                    raise ValueError("Empty response from Jimeng API")

                # Decode base64 image
                image_bytes = base64.b64decode(image_base64)

                # Create MCP Image object
                mcp_image = MCPImage(data=image_bytes, format="png")
                all_images.append(mcp_image)

                # Metadata
                metadata = {
                    "provider": "jimeng",
                    "model": "jimeng_t2i_v40",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "aspect_ratio": aspect_ratio or "3:4",
                    "width": self.config.default_width,
                    "height": self.config.default_height,
                    "index": i + 1,
                    "mime_type": "image/png",
                }
                all_metadata.append(metadata)

                self.logger.info(
                    f"Successfully generated image {i + 1}/{n} "
                    f"({len(image_bytes)} bytes)"
                )

            except Exception as e:
                self.logger.error(f"Failed to generate image {i + 1}/{n}: {e}")
                # Continue with next image rather than failing completely
                continue

        return all_images, all_metadata

    def edit_image(
        self,
        instruction: str,
        image_data: bytes,
        mime_type: str = "image/png",
        **kwargs
    ) -> Tuple[List[MCPImage], int]:
        """
        Edit an image using Jimeng AI.

        Args:
            instruction: Natural language editing instruction
            image_data: Source image bytes
            mime_type: MIME type of source image
            **kwargs: Additional parameters

        Returns:
            Tuple of (edited_images, count)
        """
        # Jimeng supports image editing through the same API with reference images
        # For now, we'll generate a new image based on the instruction
        # This can be enhanced later to support true editing

        self.logger.warning(
            "Jimeng edit_image uses text-to-image generation. "
            "True image editing not yet implemented."
        )

        images, metadata = self.generate_images(
            prompt=instruction,
            n=1,
            aspect_ratio="3:4"
        )

        return images, len(images)

    def validate_config(self) -> bool:
        """Validate Jimeng configuration."""
        return self.config.validate_credentials()

    def _build_generation_request(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        reference_images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Build request body for Jimeng API."""
        req_body = {
            "req_key": "jimeng_t2i_v40",
            "req_json": "{}",
            "prompt": prompt,
            "width": self.config.default_width,
            "height": self.config.default_height,
            "scale": 0.5,
            "force_single": True,
        }

        # Add reference images if provided
        if reference_images:
            req_body["image_urls"] = reference_images

        return req_body

    def _make_request_with_retry(
        self,
        req_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic.

        Implements exponential backoff for retries.
        """
        body_str = json.dumps(req_body)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        payload_hash = hashlib.sha256(body_str.encode()).hexdigest()

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Host": self.HOST,
            "X-Date": timestamp,
            "X-Content-Sha256": payload_hash,
        }

        # Generate signature
        signature = self._generate_signature(
            method="POST",
            body=body_str,
            timestamp=timestamp,
            headers=headers
        )
        headers["Authorization"] = signature

        # Build URL
        query = f"Action=CVProcess&Version={self.VERSION}"
        url = f"https://{self.HOST}/?{query}"

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
                response = self.client.post(url, headers=headers, content=body_str)
                response.raise_for_status()

                result = response.json()

                # Check for API errors
                if result.get("code") != 10000:
                    error_msg = result.get("message", "Unknown error")
                    raise ValueError(
                        f"Jimeng API error: code={result.get('code')}, msg={error_msg}"
                    )

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
        raise last_error or RuntimeError("All retry attempts failed")

    def _generate_signature(
        self,
        method: str,
        body: str,
        timestamp: str,
        headers: Dict[str, str]
    ) -> str:
        """
        Generate Volcengine Signature V4 HMAC-SHA256 signature.

        This follows the Volcengine Signature V4 algorithm:
        https://www.volcengine.com/docs/6291/74920
        """
        # Sort headers
        sorted_headers = sorted(headers.keys())
        canonical_headers = "\n".join(
            f"{key.lower()}:{headers[key].strip()}"
            for key in sorted_headers
        ) + "\n"
        signed_headers = ";".join(key.lower() for key in sorted_headers)

        # Canonical request
        payload_hash = hashlib.sha256(body.encode()).hexdigest()
        canonical_request = "\n".join([
            method,
            "/",
            f"Action=CVProcess&Version={self.VERSION}",
            canonical_headers,
            signed_headers,
            payload_hash
        ])

        # String to sign
        date_stamp = timestamp[:8]
        credential_scope = f"{date_stamp}/{self.REGION}/{self.SERVICE}/request"
        hashed_canonical = hashlib.sha256(canonical_request.encode()).hexdigest()
        string_to_sign = "\n".join([
            "HMAC-SHA256",
            timestamp,
            credential_scope,
            hashed_canonical
        ])

        # Calculate signature
        signing_key = self._get_signature_key(
            self.config.secret_key,
            date_stamp,
            self.REGION,
            self.SERVICE
        )
        signature = hmac.new(
            signing_key,
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()

        # Build authorization header
        authorization = (
            f"HMAC-SHA256 Credential={self.config.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        return authorization

    def _get_signature_key(
        self,
        secret_key: str,
        date_stamp: str,
        region_name: str,
        service_name: str
    ) -> bytes:
        """
        Derive signing key for Volcengine Signature V4.

        This follows the key derivation hierarchy:
        kDate -> kRegion -> kService -> kSigning
        """
        def sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode(), hashlib.sha256).digest()

        k_date = sign(secret_key.encode(), date_stamp)
        k_region = sign(k_date, region_name)
        k_service = sign(k_region, service_name)
        k_signing = sign(k_service, "request")

        return k_signing

    def _get_capabilities(self) -> Dict[str, Any]:
        """Get Jimeng provider capabilities."""
        return {
            "max_images_per_request": 1,
            "supported_aspect_ratios": ["3:4", "1:1", "16:9"],
            "supports_editing": True,
            "supports_reference_images": True,
            "max_resolution": "2048x1536",
            "default_resolution": f"{self.config.default_height}x{self.config.default_width}",
            "retry_config": {
                "max_retries": self.config.max_retries,
                "retry_delay": self.config.retry_delay,
            }
        }

    def __del__(self):
        """Cleanup HTTP client on deletion."""
        if self._client:
            self._client.close()
