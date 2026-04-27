"""Gemini provider handler for generate_image tool."""

import base64
import logging
import mimetypes
from typing import List, Optional, Tuple

from fastmcp.utilities.types import Image as MCPImage

from ...config.settings import ModelTier, ThinkingLevel
from ...core.exceptions import ValidationError
from ... import services
from .utils import get_enhanced_image_service

logger = logging.getLogger(__name__)


def handle_gemini_request(
    prompt: str,
    n: int,
    negative_prompt: Optional[str],
    system_instruction: Optional[str],
    input_image_paths: Optional[List[str]],
    file_id: Optional[str],
    aspect_ratio: Optional[str],
    model_tier: str,
    thinking_level: str,
    resolution: str,
    enable_grounding: bool,
    output_dir: Optional[str],
    detected_mode: str,
) -> Tuple[List[MCPImage], List[dict], str, dict]:
    """
    Handle a Gemini provider generation or edit request.

    Returns:
        Tuple of (thumbnail_images, metadata, selected_tier, model_info)
    """
    # Parse model tier
    try:
        tier = ModelTier(model_tier) if model_tier else ModelTier.AUTO
    except ValueError:
        logger.warning(f"Invalid model_tier '{model_tier}', defaulting to AUTO")
        tier = ModelTier.AUTO

    # Validate thinking level for Pro model
    try:
        if thinking_level:
            _ = ThinkingLevel(thinking_level)
    except ValueError:
        logger.warning(f"Invalid thinking_level '{thinking_level}', defaulting to HIGH")
        thinking_level = "high"

    # Get model selector to determine which model to use
    model_selector = services.get_model_selector()

    _, selected_tier = model_selector.select_model(
        prompt=prompt,
        requested_tier=tier,
        n=n,
        resolution=resolution,
        input_images=input_image_paths,
        thinking_level=thinking_level,
        enable_grounding=enable_grounding,
    )

    model_info = model_selector.get_model_info(selected_tier)
    logger.info(
        f"Selected {model_info['emoji']} {model_info['name']} "
        f"({selected_tier.value}) for this request"
    )

    enhanced_image_service = get_enhanced_image_service(selected_tier)

    thumbnail_images: List[MCPImage] = []
    metadata: List[dict] = []

    if detected_mode == "edit" and file_id:
        logger.info(f"Edit mode: using file_id {file_id}")
        thumbnail_images, metadata = enhanced_image_service.edit_image_by_file_id(
            file_id=file_id, edit_prompt=prompt, output_dir=output_dir
        )

    elif detected_mode == "edit" and input_image_paths and len(input_image_paths) == 1:
        logger.info(f"Edit mode: using file path {input_image_paths[0]}")
        thumbnail_images, metadata = enhanced_image_service.edit_image_by_path(
            instruction=prompt, file_path=input_image_paths[0], output_dir=output_dir
        )

    else:
        logger.info("Generate mode: creating new images")
        if aspect_ratio:
            logger.info(f"Using aspect ratio override: {aspect_ratio}")

        input_images = None
        if input_image_paths:
            input_images = _load_input_images(input_image_paths)
            logger.info(f"Loaded {len(input_images)} input images from file paths")

        thumbnail_images, metadata = enhanced_image_service.generate_images(
            prompt=prompt,
            n=n,
            negative_prompt=negative_prompt,
            system_instruction=system_instruction,
            input_images=input_images,
            aspect_ratio=aspect_ratio,
            output_dir=output_dir,
        )

    return thumbnail_images, metadata, selected_tier.value, model_info


def _load_input_images(paths: List[str]) -> List[Tuple[str, str]]:
    """Load images from file paths into base64 tuples."""
    input_images: List[Tuple[str, str]] = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                image_bytes = f.read()
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            input_images.append((base64_data, mime_type))
            logger.debug(f"Loaded input image: {path} ({mime_type})")
        except Exception as e:
            raise ValidationError(f"Failed to load input image {path}: {e}") from e
    return input_images
