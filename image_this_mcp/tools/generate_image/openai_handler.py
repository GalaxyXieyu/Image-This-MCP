"""OpenAI provider handler for generate_image tool."""

import logging
from typing import List, Optional, Tuple

from fastmcp.utilities.types import Image as MCPImage

from ...core.exceptions import ValidationError
from ... import services

logger = logging.getLogger(__name__)


def handle_openai_request(
    prompt: str,
    n: int,
    negative_prompt: Optional[str],
    input_image_paths: Optional[List[str]],
    file_id: Optional[str],
    aspect_ratio: Optional[str],
    output_dir: Optional[str],
    detected_mode: str,
    model: Optional[str] = None,
) -> Tuple[List[MCPImage], List[dict]]:
    """
    Handle an OpenAI provider generation request.

    Returns:
        Tuple of (thumbnail_images, metadata)
    """
    if file_id:
        raise ValidationError("OpenAI provider does not support Files API file_id inputs")

    if detected_mode == "edit":
        raise ValidationError(
            "OpenAI provider does not support image editing. Use provider='gemini'."
        )

    if input_image_paths:
        raise ValidationError(
            "OpenAI provider does not support input_image_paths for conditioning yet"
        )

    provider_instance = services.get_provider("openai")
    if not provider_instance:
        raise ValidationError("Provider 'openai' not initialized")

    logger.info("Generate mode (OpenAI): creating new images")
    if aspect_ratio:
        logger.info(f"Using aspect ratio override: {aspect_ratio}")

    thumbnail_images, metadata = provider_instance.generate_images(
        prompt=prompt,
        n=n,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        model=model,
    )

    for i, img in enumerate(thumbnail_images):
        if i < len(metadata) and isinstance(metadata[i], dict):
            metadata[i].setdefault(
                "size_bytes", len(img.data) if hasattr(img, "data") else 0
            )

    # Persist images through file service
    file_service = services.get_file_image_service()
    saved_thumbnails: List[MCPImage] = []
    saved_metadata: List[dict] = []
    for i, img in enumerate(thumbnail_images):
        image_bytes = getattr(img, "data", None)
        if not image_bytes:
            logger.warning(f"OpenAI returned empty image data at index {i + 1}")
            continue

        base_meta = metadata[i] if i < len(metadata) and isinstance(metadata[i], dict) else {}
        mime_type = base_meta.get("mime_type") if base_meta else None
        thumbnail_image, file_meta = file_service.save_external_image(
            image_bytes=image_bytes,
            mime_type=mime_type or "image/png",
            metadata=base_meta or None,
            output_dir=output_dir,
        )
        saved_thumbnails.append(thumbnail_image)
        saved_metadata.append(file_meta)

    if not saved_metadata:
        raise ValidationError("OpenAI provider returned no images to save")

    return saved_thumbnails, saved_metadata
