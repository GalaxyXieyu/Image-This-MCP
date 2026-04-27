"""Centralized model registry with dynamic registration and filtering."""

import logging
from typing import Dict, List, Optional, Callable

from .model_info import ModelInfo, ModelTier

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for all image generation models.

    Supports runtime registration from code, config files, or API responses.
    No model is hard-coded inside this class.
    """

    _models: Dict[str, ModelInfo] = {}
    _provider_defaults: Dict[str, str] = {}  # provider -> default model_id

    @classmethod
    def register(cls, model: ModelInfo) -> None:
        """Register a model. Overwrites existing entries with the same id."""
        cls._models[model.id] = model
        logger.info(f"Registered model: {model.id} ({model.name})")

    @classmethod
    def register_from_dict(cls, data: dict) -> None:
        """Convenience: register a model from a plain dict (e.g. JSON)."""
        caps = data.pop("capabilities", {})
        model = ModelInfo(
            **data,
            capabilities=ModelInfo.__dataclass_fields__["capabilities"].type(**caps),
        )
        cls.register(model)

    @classmethod
    def unregister(cls, model_id: str) -> None:
        """Remove a model from the registry."""
        cls._models.pop(model_id, None)
        for provider, default_id in list(cls._provider_defaults.items()):
            if default_id == model_id:
                cls._provider_defaults.pop(provider, None)

    @classmethod
    def get(cls, model_id: str) -> Optional[ModelInfo]:
        """Get a model by its unique id."""
        return cls._models.get(model_id)

    @classmethod
    def list_all(cls) -> List[ModelInfo]:
        """Return all registered models."""
        return list(cls._models.values())

    @classmethod
    def list_ids(cls) -> List[str]:
        """Return all registered model ids."""
        return list(cls._models.keys())

    @classmethod
    def filter(
        cls,
        provider: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        capability: Optional[str] = None,
        predicate: Optional[Callable[[ModelInfo], bool]] = None,
    ) -> List[ModelInfo]:
        """Filter models by criteria.

        Args:
            provider: Match provider key (e.g. "gemini", "jimeng").
            tier: Match tier enum.
            capability: Require a capability flag (e.g. "grounding").
            predicate: Optional custom filter function.
        """
        results = cls.list_all()
        if provider:
            results = [m for m in results if m.provider == provider]
        if tier:
            results = [m for m in results if m.tier == tier]
        if capability:
            results = [m for m in results if m.supports(capability)]
        if predicate:
            results = [m for m in results if predicate(m)]
        return results

    @classmethod
    def set_provider_default(cls, provider: str, model_id: str) -> None:
        """Set the default model for a provider."""
        cls._provider_defaults[provider] = model_id
        logger.info(f"Default model for {provider}: {model_id}")

    @classmethod
    def get_provider_default(cls, provider: str) -> Optional[ModelInfo]:
        """Get the default model for a provider."""
        model_id = cls._provider_defaults.get(provider)
        if model_id:
            return cls.get(model_id)
        # Fallback: first model for that provider
        models = cls.filter(provider=provider)
        return models[0] if models else None

    @classmethod
    def get_default_provider_model(cls) -> Optional[ModelInfo]:
        """Get the global default model (first available)."""
        if cls._models:
            return next(iter(cls._models.values()))
        return None

    @classmethod
    def reset(cls) -> None:
        """Clear all registered models (useful for testing)."""
        cls._models.clear()
        cls._provider_defaults.clear()
        logger.info("ModelRegistry reset")
