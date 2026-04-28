"""Jimeng provider handler for generate_image tool."""

import logging
import mimetypes
from typing import List, Optional, Tuple

from fastmcp.utilities.types import Image as MCPImage

from ...core.exceptions import ValidationError
from ... import services

logger = logging.getLogger(__name__)


def handle_jimeng_request(
    prompt: str,
    n: int,
    negative_prompt: Optional[str],
    input_image_paths: Optional[List[str]],
    file_id: Optional[str],
    aspect_ratio: Optional[str],
    output_dir: Optional[str],
    detected_mode: str,
    provider_name: str = "jimeng",
    model: Optional[str] = None,
) -> Tuple[List[MCPImage], List[dict]]:
    """
    Handle a Jimeng provider generation or edit request.

    Returns:
        Tuple of (thumbnail_images, metadata)
    """
    if file_id:
        raise ValidationError(
            "Jimeng provider does not support Files API file_id inputs"
        )

    provider_instance = services.get_provider(provider_name)
    if not provider_instance:
        raise ValidationError(f"Provider '{provider_name}' not initialized")

    thumbnail_images: List[MCPImage] = []
    metadata: List[dict] = []

    if detected_mode == "edit":
        if not input_image_paths:
            raise ValidationError("Jimeng edit mode requires input_image_paths")
        if len(input_image_paths) != 1:
            raise ValidationError("Jimeng edit mode supports exactly one input image")

        edit_path = input_image_paths[0]
        logger.info(f"Edit mode (Jimeng): using file path {edit_path}")

        try:
            with open(edit_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            raise ValidationError(f"Failed to read input image {edit_path}: {e}") from e

        mime_type, _ = mimetypes.guess_type(edit_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"

        thumbnail_images, _ = provider_instance.edit_image(
            instruction=prompt,
            image_data=image_bytes,
            mime_type=mime_type,
        )
        metadata = [
            {
                "provider": "jimeng",
                "model": model or "jimeng_t2i_v40",
                "instruction": prompt,
                "edit_index": i + 1,
                "mime_type": "image/png",
                "size_bytes": len(img.data) if hasattr(img, "data") else 0,
            }
            for i, img in enumerate(thumbnail_images)
        ]
    else:
        if input_image_paths:
            raise ValidationError(
                "Jimeng provider does not support input_image_paths for conditioning yet"
            )

        logger.info("Generate mode (Jimeng): creating new images")
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
            logger.warning(f"Jimeng returned empty image data at index {i + 1}")
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
        raise ValidationError("Jimeng provider returned no images to save")

    return saved_thumbnails, saved_metadata
