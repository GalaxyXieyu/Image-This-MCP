"""Model metadata and capability definitions."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelTier(str, Enum):
    """Model performance/quality tier."""
    FLASH = "flash"      # Speed-optimized
    PRO = "pro"          # Quality-optimized
    STANDARD = "standard"  # Balanced
    CUSTOM = "custom"    # User-defined or third-party


@dataclass
class ModelCapability:
    """Fine-grained capability flags for a model."""
    thinking: bool = False
    grounding: bool = False
    media_resolution_control: bool = False
    text_rendering: bool = False
    editing: bool = True
    reference_images: bool = True
    multi_image_conditioning: bool = False
    search_grounding: bool = False
    high_resolution: bool = False
    watermark_optional: bool = False
    aspect_ratio_control: bool = True
    negative_prompt: bool = False
    system_instruction: bool = False


@dataclass
class ModelInfo:
    """Complete metadata for an image generation model.

    All fields have sensible defaults so third-party models can be registered
    with minimal boilerplate.
    """
    # Identity
    id: str                                    # Unique identifier, e.g. "gemini-3.1-flash-image-preview"
    name: str                                  # Human-readable name
    provider: str                              # Provider key, e.g. "gemini", "jimeng", "jimeng45"
    tier: ModelTier = ModelTier.STANDARD       # Performance tier

    # Technical specs
    model_name: str = ""                       # API-facing model name (falls back to id if empty)
    max_resolution: int = 1024                 # Max output dimension in pixels
    default_resolution: str = "1024"           # Default resolution string
    supported_aspect_ratios: List[str] = field(default_factory=lambda: [
        "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
    ])
    max_images_per_request: int = 4
    max_inline_image_size: int = 20 * 1024 * 1024  # 20 MB
    default_image_format: str = "png"
    request_timeout: int = 60                  # seconds

    # Capabilities
    capabilities: ModelCapability = field(default_factory=ModelCapability)

    # Provider-specific defaults
    default_config: Dict[str, Any] = field(default_factory=dict)

    # Presentation
    description: str = ""
    emoji: str = ""
    best_for: str = ""

    def __post_init__(self):
        if not self.model_name:
            self.model_name = self.id
        if isinstance(self.tier, str):
            self.tier = ModelTier(self.tier)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (suitable for JSON/tool output)."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "tier": self.tier.value,
            "model_name": self.model_name,
            "max_resolution": self.max_resolution,
            "default_resolution": self.default_resolution,
            "supported_aspect_ratios": self.supported_aspect_ratios,
            "max_images_per_request": self.max_images_per_request,
            "capabilities": {
                k: v for k, v in self.capabilities.__dict__.items()
            },
            "description": self.description,
            "emoji": self.emoji,
            "best_for": self.best_for,
        }

    def supports(self, capability: str) -> bool:
        """Check if this model supports a given capability flag."""
        return getattr(self.capabilities, capability, False)
