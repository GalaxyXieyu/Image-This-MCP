"""Dynamic model registry for MCP image generation providers.

This module provides a centralized, extensible registry for AI image generation models.
Models are no longer hard-coded; they can be registered at runtime from config files,
environment variables, or API responses.
"""

from .model_info import ModelCapability, ModelInfo, ModelTier
from .registry import ModelRegistry
from .defaults import register_default_models

__all__ = [
    "ModelCapability",
    "ModelInfo",
    "ModelTier",
    "ModelRegistry",
    "register_default_models",
]
