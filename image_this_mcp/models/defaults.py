"""Default model definitions for built-in providers.

These are registered automatically when the server starts.
Users can override or extend them via environment variables or config files.
"""

from .model_info import ModelCapability, ModelInfo, ModelTier
from .registry import ModelRegistry


def register_default_models() -> None:
    """Register all built-in models. Safe to call multiple times."""
    # Gemini Flash
    Registry = ModelRegistry
    Registry.register(
        ModelInfo(
            id="gemini-3.1-flash-image-preview",
            name="Gemini 3.1 Flash Image Preview",
            provider="gemini",
            tier=ModelTier.FLASH,
            model_name="gemini-3.1-flash-image-preview",
            max_resolution=1024,
            default_resolution="1024",
            request_timeout=60,
            capabilities=ModelCapability(
                editing=True,
                reference_images=True,
                aspect_ratio_control=True,
            ),
            description="Google's fast image generation model. Great for rapid prototyping and quick iterations.",
            emoji="⚡",
            best_for="Rapid prototyping, quick iterations, high-volume generation",
            default_config={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
            },
        )
    )

    # Gemini Pro
    Registry.register(
        ModelInfo(
            id="gemini-3-pro-image-preview",
            name="Gemini 3 Pro Image",
            provider="gemini",
            tier=ModelTier.PRO,
            model_name="gemini-3-pro-image-preview",
            max_resolution=3840,
            default_resolution="high",
            request_timeout=90,
            capabilities=ModelCapability(
                thinking=True,
                grounding=True,
                media_resolution_control=True,
                text_rendering=True,
                editing=True,
                reference_images=True,
                multi_image_conditioning=True,
                search_grounding=True,
                high_resolution=True,
                aspect_ratio_control=True,
                system_instruction=True,
            ),
            description="Google's most advanced image generation model. Supports 4K, grounding, and advanced reasoning.",
            emoji="🏆",
            best_for="Production assets, professional photography, high-fidelity outputs, images with text",
            default_config={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
                "thinking_level": "HIGH",
                "media_resolution": "HIGH",
                "enable_search_grounding": True,
            },
        )
    )

    # Jimeng (legacy Volcengine visual API)
    Registry.register(
        ModelInfo(
            id="jimeng",
            name="Jimeng AI",
            provider="jimeng",
            tier=ModelTier.STANDARD,
            model_name="jimeng",
            max_resolution=2048,
            default_resolution="1536x2048",
            supported_aspect_ratios=["3:4", "4:3", "1:1", "16:9", "9:16"],
            request_timeout=120,
            capabilities=ModelCapability(
                editing=False,
                reference_images=True,
                aspect_ratio_control=True,
                watermark_optional=True,
            ),
            description="Volcengine Jimeng AI. Chinese-optimized image generation with serial queue protection.",
            emoji="🎨",
            best_for="Chinese-language prompts, portrait-oriented images",
            default_config={
                "width": 1536,
                "height": 2048,
                "max_retries": 3,
                "retry_delay": 5,
            },
        )
    )

    # Jimeng 4.5 (Seedream via Ark API)
    Registry.register(
        ModelInfo(
            id="doubao-seedream-4-5-251128",
            name="Jimeng 4.5 (Seedream)",
            provider="jimeng45",
            tier=ModelTier.STANDARD,
            model_name="doubao-seedream-4-5-251128",
            max_resolution=2304,
            default_resolution="1728x2304",
            supported_aspect_ratios=[
                "1:1", "3:4", "2:3", "4:3", "16:9", "9:16"
            ],
            request_timeout=120,
            capabilities=ModelCapability(
                editing=False,
                reference_images=True,
                aspect_ratio_control=True,
                watermark_optional=True,
                high_resolution=True,
            ),
            description="ByteDance Seedream 4.5 via Volcengine Ark API. High-quality Chinese image generation.",
            emoji="🎨",
            best_for="Xiaohongshu-style portraits, high-quality Chinese aesthetics",
            default_config={
                "size": "1728x2304",
                "response_format": "b64_json",
                "watermark": False,
                "max_retries": 3,
                "retry_delay": 5,
            },
        )
    )

    # OpenAI gpt-image-2
    Registry.register(
        ModelInfo(
            id="gpt-image-2",
            name="OpenAI GPT Image 2",
            provider="openai",
            tier=ModelTier.PRO,
            model_name="gpt-image-2",
            max_resolution=1792,
            default_resolution="1024x1024",
            supported_aspect_ratios=["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"],
            max_images_per_request=1,
            request_timeout=120,
            capabilities=ModelCapability(
                editing=False,
                reference_images=False,
                aspect_ratio_control=True,
                high_resolution=True,
                text_rendering=True,
            ),
            description="OpenAI's latest image generation model (gpt-image-2). High-quality photorealistic outputs via OpenAI Images API.",
            emoji="🖼️",
            best_for="Photorealistic images, artistic illustrations, DALL-E quality outputs",
            default_config={
                "quality": "standard",
                "size": "1024x1024",
                "response_format": "b64_json",
                "max_retries": 3,
                "retry_delay": 5,
            },
        )
    )

    # OpenAI dall-e-3
    Registry.register(
        ModelInfo(
            id="dall-e-3",
            name="OpenAI DALL-E 3",
            provider="openai",
            tier=ModelTier.STANDARD,
            model_name="dall-e-3",
            max_resolution=1792,
            default_resolution="1024x1024",
            supported_aspect_ratios=["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"],
            max_images_per_request=1,
            request_timeout=120,
            capabilities=ModelCapability(
                editing=False,
                reference_images=False,
                aspect_ratio_control=True,
                high_resolution=True,
                text_rendering=True,
            ),
            description="OpenAI DALL-E 3. Reliable text-to-image generation with style and quality controls.",
            emoji="🎨",
            best_for="Creative illustrations, conceptual art, reliable text rendering",
            default_config={
                "quality": "standard",
                "style": "vivid",
                "size": "1024x1024",
                "response_format": "b64_json",
                "max_retries": 3,
                "retry_delay": 5,
            },
        )
    )

    # Set provider defaults
    Registry.set_provider_default("gemini", "gemini-3.1-flash-image-preview")
    Registry.set_provider_default("jimeng", "jimeng")
    Registry.set_provider_default("jimeng45", "doubao-seedream-4-5-251128")
    Registry.set_provider_default("openai", "gpt-image-2")
