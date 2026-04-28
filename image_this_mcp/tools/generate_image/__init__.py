"""Generate or edit images (Multi-Provider: Gemini, Jimeng).

This package splits the previous monolithic generate_image tool into:
- __init__.py: orchestration, validation, provider resolution
- gemini_handler.py: Gemini-specific generation/edit logic
- jimeng_handler.py: Jimeng-specific generation/edit logic
- response_builder.py: text summaries and structured content
- utils.py: helper functions
"""

import logging
import os
from typing import Annotated, List, Literal, Optional

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from pydantic import Field

from ...config.constants import MAX_INPUT_IMAGES
from ...config.settings import ModelTier
from ...core.exceptions import ValidationError
from ... import services

from .gemini_handler import handle_gemini_request
from .jimeng_handler import handle_jimeng_request
from .openai_handler import handle_openai_request
from .response_builder import build_gemini_response, build_jimeng_response, build_openai_response, build_empty_response


def register_generate_image_tool(server: FastMCP):
    """Register the generate_image tool with the FastMCP server."""

    @server.tool(
        annotations={
            "title": "Generate or edit images (Multi-Provider: Gemini, Jimeng, OpenAI)",
            "readOnlyHint": True,
            "openWorldHint": True,
        }
    )
    def generate_image(
        prompt: Annotated[
            str,
            Field(
                description="Clear, detailed image prompt. Include subject, composition, "
                "action, location, style, and any text to render. Use the aspect_ratio "
                "parameter to pin a specific canvas shape when needed.",
                min_length=1,
                max_length=8192,
            ),
        ],
        n: Annotated[
            int, Field(description="Requested image count (model may return fewer).", ge=1, le=4)
        ] = 1,
        negative_prompt: Annotated[
            Optional[str],
            Field(description="Things to avoid (style, objects, text).", max_length=1024),
        ] = None,
        system_instruction: Annotated[
            Optional[str], Field(description="Optional system tone/style guidance.", max_length=512)
        ] = None,
        input_image_path_1: Annotated[
            Optional[str],
            Field(description="Path to first input image for composition/conditioning"),
        ] = None,
        input_image_path_2: Annotated[
            Optional[str],
            Field(description="Path to second input image for composition/conditioning"),
        ] = None,
        input_image_path_3: Annotated[
            Optional[str],
            Field(description="Path to third input image for composition/conditioning"),
        ] = None,
        file_id: Annotated[
            Optional[str],
            Field(
                description="Files API file ID to use as input/edit source (e.g., 'files/abc123'). "
                "If provided, this takes precedence over input_image_path_* parameters for the primary input."
            ),
        ] = None,
        mode: Annotated[
            str,
            Field(
                description="Operation mode: 'generate' for new image creation, 'edit' for modifying existing images. "
                "Auto-detected based on input parameters if not specified."
            ),
        ] = "auto",
        model_tier: Annotated[
            Optional[str],
            Field(
                description="Model tier: 'flash' (speed, 1024px), 'pro' (quality, up to 4K), or 'auto' (smart selection). "
                "Default: 'flash' - use Gemini 3.1 Flash Image Preview for faster, cleaner iteration."
            ),
        ] = "flash",
        resolution: Annotated[
            Optional[str],
            Field(
                description="Output resolution: 'high', '4k', '2k', '1k'. "
                "4K and 2K only available with 'pro' model. Default: 'high'."
            ),
        ] = "high",
        thinking_level: Annotated[
            Optional[str],
            Field(
                description="Reasoning depth for Pro model: 'low' (faster), 'high' (better quality). "
                "Only applies to Pro model. Default: 'high'."
            ),
        ] = "high",
        enable_grounding: Annotated[
            bool,
            Field(
                description="Enable Google Search grounding for factual accuracy (Pro model only). "
                "Useful for real-world subjects. Default: true."
            ),
        ] = True,
        aspect_ratio: Annotated[
            Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"] | None,
            Field(
                description="Optional output aspect ratio (e.g., '16:9'). "
                "See docs for supported values: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9."
            ),
        ] = None,
        provider: Annotated[
            Literal["gemini", "jimeng", "openai", "auto"] | None,
            Field(
                description="Image generation provider: 'gemini' (Nano Banana - Flash/Pro models), "
                "'jimeng' (Volcengine Jimeng - Chinese-optimized), "
                "'openai' (DALL-E 3 / GPT Image 2), or 'auto' (use default from config). "
                "Default: 'auto' - uses IMAGE_PROVIDER environment variable or defaults to 'gemini'."
            ),
        ] = "auto",
        output_dir: Annotated[
            Optional[str],
            Field(
                description="Custom output directory for saving generated images. "
                "If not specified, uses the default directory (~/image-this). "
                "Path can be absolute or relative to the current working directory."
            ),
        ] = None,
        _ctx: Context = None,
    ) -> ToolResult:
        """
        Generate new images or edit existing images using natural language instructions.

        **Multi-Provider Support**:
        - **gemini**: Nano Banana (Flash & Pro models, 4K support, Google Search grounding)
        - **jimeng**: Volcengine Jimeng (Chinese-optimized, 3:4 portrait default, serial queue)
        - **openai**: OpenAI Images API (DALL-E 3 / GPT Image 2, photorealistic outputs)
        - **auto**: Automatically selects based on IMAGE_PROVIDER environment variable (default: gemini)

        Supports multiple input modes:
        1. Pure generation: Just provide a prompt to create new images
        2. Multi-image conditioning: Provide up to 3 input images using input_image_path_1/2/3 parameters
        3. File ID editing: Edit previously uploaded images using Files API ID
        4. File path editing: Edit local images by providing single input image path

        Automatically detects mode based on parameters or can be explicitly controlled.
        Input images are read from the local filesystem to avoid massive token usage.
        Returns both MCP image content blocks and structured JSON with metadata.
        """
        logger = logging.getLogger(__name__)

        try:
            # Resolve provider
            resolved_provider = provider or "auto"
            if resolved_provider == "auto":
                env_provider = os.getenv("IMAGE_PROVIDER")
                if env_provider:
                    resolved_provider = env_provider.lower()
                else:
                    available_providers = services.list_initialized_providers()
                    resolved_provider = available_providers[0] if available_providers else "gemini"
                logger.info(f"Auto-selected provider: {resolved_provider}")
            else:
                logger.info(f"Using specified provider: {resolved_provider}")
            provider = resolved_provider

            # Validate provider is available
            available = services.list_initialized_providers()
            if provider not in available:
                if available:
                    raise ValidationError(
                        f"Provider '{provider}' not available. "
                        f"Initialized providers: {', '.join(available)}"
                    )
                else:
                    raise ValidationError(
                        f"No providers initialized. Please check your configuration."
                    )

            # Construct input_image_paths list from individual parameters
            input_image_paths: List[str] = []
            for path in [input_image_path_1, input_image_path_2, input_image_path_3]:
                if path:
                    input_image_paths.append(path)
            if not input_image_paths:
                input_image_paths = None  # type: ignore[assignment]

            logger.info(
                f"Generate image request: prompt='{prompt[:50]}...', n={n}, "
                f"paths={input_image_paths}, provider={provider}, model_tier={model_tier}, aspect_ratio={aspect_ratio}"
            )

            # Auto-detect mode based on inputs
            detected_mode = mode
            if mode == "auto":
                if file_id or (input_image_paths and len(input_image_paths) == 1):
                    detected_mode = "edit"
                else:
                    detected_mode = "generate"

            # Validation
            if mode not in ["auto", "generate", "edit"]:
                raise ValidationError("Mode must be 'auto', 'generate', or 'edit'")

            if input_image_paths:
                if len(input_image_paths) > MAX_INPUT_IMAGES:
                    raise ValidationError(f"Maximum {MAX_INPUT_IMAGES} input images allowed")
                for i, path in enumerate(input_image_paths):
                    if not os.path.exists(path):
                        raise ValidationError(f"Input image {i + 1} not found: {path}")
                    if not os.path.isfile(path):
                        raise ValidationError(f"Input image {i + 1} is not a file: {path}")

            if detected_mode == "edit":
                if not file_id and not input_image_paths:
                    raise ValidationError("Edit mode requires either file_id or input_image_paths")
                if file_id and input_image_paths and len(input_image_paths) > 1:
                    raise ValidationError(
                        "Edit mode with file_id supports only additional input images, not multiple primary inputs"
                    )

            thumbnail_images = []
            metadata = []
            selected_tier = None
            model_info = None
            request_limiter = services.get_request_limiter()

            with request_limiter.limit(provider):
                if provider == "gemini":
                    thumbnail_images, metadata, selected_tier, model_info = handle_gemini_request(
                        prompt=prompt,
                        n=n,
                        negative_prompt=negative_prompt,
                        system_instruction=system_instruction,
                        input_image_paths=input_image_paths,  # type: ignore[arg-type]
                        file_id=file_id,
                        aspect_ratio=aspect_ratio,
                        model_tier=model_tier,
                        thinking_level=thinking_level,
                        resolution=resolution,
                        enable_grounding=enable_grounding,
                        output_dir=output_dir,
                        detected_mode=detected_mode,
                    )
                elif provider == "openai":
                    thumbnail_images, metadata = handle_openai_request(
                        prompt=prompt,
                        n=n,
                        negative_prompt=negative_prompt,
                        input_image_paths=input_image_paths,  # type: ignore[arg-type]
                        file_id=file_id,
                        aspect_ratio=aspect_ratio,
                        output_dir=output_dir,
                        detected_mode=detected_mode,
                    )
                else:
                    thumbnail_images, metadata = handle_jimeng_request(
                        prompt=prompt,
                        n=n,
                        negative_prompt=negative_prompt,
                        input_image_paths=input_image_paths,  # type: ignore[arg-type]
                        file_id=file_id,
                        aspect_ratio=aspect_ratio,
                        output_dir=output_dir,
                        detected_mode=detected_mode,
                    )

            # Build response
            if metadata:
                metadata = [m for m in metadata if m is not None and isinstance(m, dict)]
                if not metadata:
                    summary = f"❌ Failed to {detected_mode} image(s): {prompt[:50]}... No valid results returned."
                    content = [__import__("mcp.types", fromlist=["TextContent"]).TextContent(type="text", text=summary)]
                    structured_content = {
                        "error": "no_valid_metadata",
                        "message": summary,
                        "mode": detected_mode,
                    }
                    return ToolResult(content=content, structured_content=structured_content)

                if provider == "gemini":
                    return build_gemini_response(
                        thumbnail_images=thumbnail_images,
                        metadata=metadata,
                        detected_mode=detected_mode,
                        selected_tier=selected_tier,
                        model_info=model_info,
                        thinking_level=thinking_level,
                        resolution=resolution,
                        enable_grounding=enable_grounding,
                        input_image_paths=input_image_paths,  # type: ignore[arg-type]
                        file_id=file_id,
                        aspect_ratio=aspect_ratio,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        n=n,
                    )
                elif provider == "openai":
                    return build_openai_response(
                        thumbnail_images=thumbnail_images,
                        metadata=metadata,
                        detected_mode=detected_mode,
                        input_image_paths=input_image_paths,  # type: ignore[arg-type]
                        aspect_ratio=aspect_ratio,
                        prompt=prompt,
                        n=n,
                    )
                else:
                    return build_jimeng_response(
                        thumbnail_images=thumbnail_images,
                        metadata=metadata,
                        detected_mode=detected_mode,
                        input_image_paths=input_image_paths,  # type: ignore[arg-type]
                        aspect_ratio=aspect_ratio,
                        prompt=prompt,
                        n=n,
                    )
            else:
                return build_empty_response(detected_mode, prompt)

        except ValidationError:
            logger.error("Validation error in generate_image", exc_info=True)
            raise
        except Exception:
            logger.error("Unexpected error in generate_image", exc_info=True)
            raise
