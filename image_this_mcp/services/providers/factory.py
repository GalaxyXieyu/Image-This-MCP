"""
Provider factory for multi-provider image generation.

This module provides a factory pattern for creating and managing
image generation provider instances.
"""

import logging
from typing import Dict, Type, Optional, List

from .base import BaseImageProvider
from .gemini_provider import GeminiProvider
from .jimeng_provider import JimengProvider
from ...config.settings import ServerConfig, GeminiConfig, JimengConfig
from ..gemini_client import GeminiClient
from ..image_storage_service import ImageStorageService

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating and managing image generation providers.

    This class implements the factory pattern to:
    - Register new provider classes
    - Create provider instances with configuration
    - Manage provider lifecycle
    - Provide access to initialized providers
    """

    _providers: Dict[str, BaseImageProvider] = {}
    _provider_classes: Dict[str, Type[BaseImageProvider]] = {
        "gemini": GeminiProvider,
        "jimeng": JimengProvider,
    }
    _storage_service: Optional[ImageStorageService] = None

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[BaseImageProvider]
    ):
        """
        Register a new provider class.

        Args:
            name: Unique provider identifier (e.g., 'gemini', 'jimeng')
            provider_class: Provider class (must inherit from BaseImageProvider)

        Example:
            ProviderFactory.register_provider("custom", CustomProvider)
        """
        cls._provider_classes[name] = provider_class
        logger.info(f"Registered provider class: {name}")

    @classmethod
    def create_provider(
        cls,
        name: str,
        **config
    ) -> BaseImageProvider:
        """
        Create a provider instance.

        Args:
            name: Provider identifier
            **config: Provider-specific configuration

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If provider not found or config is invalid
        """
        if name not in cls._provider_classes:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Available providers: {list(cls._provider_classes.keys())}"
            )

        provider_class = cls._provider_classes[name]

        try:
            provider = provider_class(**config)

            if not provider.validate_config():
                raise ValueError(
                    f"Invalid configuration for provider: {name}. "
                    f"Please check your credentials and settings."
                )

            logger.info(f"Created provider instance: {name}")
            return provider

        except Exception as e:
            logger.error(f"Failed to create provider {name}: {e}")
            raise

    @classmethod
    def get_provider(cls, name: str) -> Optional[BaseImageProvider]:
        """
        Get an initialized provider instance.

        Args:
            name: Provider identifier

        Returns:
            Provider instance or None if not initialized
        """
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List of provider identifiers
        """
        return list(cls._provider_classes.keys())

    @classmethod
    def list_initialized_providers(cls) -> List[str]:
        """
        List all initialized provider names.

        Returns:
            List of initialized provider identifiers
        """
        return list(cls._providers.keys())

    @classmethod
    def set_storage_service(cls, storage_service: ImageStorageService):
        """
        Set the shared image storage service.

        Args:
            storage_service: ImageStorageService instance
        """
        cls._storage_service = storage_service
        logger.info("Storage service registered with ProviderFactory")

    @classmethod
    def initialize_all_providers(
        cls,
        server_config: ServerConfig,
        gemini_config: Optional[GeminiConfig] = None,
        jimeng_config: Optional[JimengConfig] = None
    ):
        """
        Initialize all available providers.

        This method creates provider instances based on available configuration.
        Providers with missing/invalid configuration are skipped with a warning.

        Args:
            server_config: Server configuration
            gemini_config: Optional Gemini configuration
            jimeng_config: Optional Jimeng configuration
        """
        logger.info("Initializing providers...")

        # Initialize Gemini provider
        if gemini_config:
            try:
                gemini_client = GeminiClient(server_config, gemini_config)
                gemini_provider = cls.create_provider(
                    "gemini",
                    client=gemini_client,
                    config=gemini_config,
                    storage_service=cls._storage_service
                )
                cls._providers["gemini"] = gemini_provider
                logger.info("✓ Gemini provider initialized")

            except Exception as e:
                logger.warning(f"✗ Gemini provider initialization failed: {e}")
                logger.warning("  Gemini features will be unavailable")

        # Initialize Jimeng provider
        if jimeng_config and jimeng_config.validate_credentials():
            try:
                jimeng_provider = cls.create_provider(
                    "jimeng",
                    config=jimeng_config
                )
                cls._providers["jimeng"] = jimeng_provider
                logger.info("✓ Jimeng provider initialized")

            except Exception as e:
                logger.warning(f"✗ Jimeng provider initialization failed: {e}")
                logger.warning("  Jimeng features will be unavailable")
        elif jimeng_config:
            logger.warning("✗ Jimeng provider skipped: missing credentials")

        # Summary
        initialized = list(cls._providers.keys())
        if initialized:
            logger.info(f"Initialized providers: {', '.join(initialized)}")
        else:
            logger.error("No providers initialized! At least one provider is required.")

    @classmethod
    def get_default_provider(cls) -> Optional[BaseImageProvider]:
        """
        Get the default provider (Gemini).

        Returns:
            Default provider instance or None if not initialized
        """
        return cls._providers.get("gemini")

    @classmethod
    def get_provider_info(cls, name: str) -> Optional[Dict]:
        """
        Get provider information and capabilities.

        Args:
            name: Provider identifier

        Returns:
            Provider info dict or None if provider not found
        """
        provider = cls.get_provider(name)
        if not provider:
            return None

        return provider.get_provider_info()

    @classmethod
    def get_all_providers_info(cls) -> Dict[str, Dict]:
        """
        Get information for all initialized providers.

        Returns:
            Dict mapping provider names to their info
        """
        return {
            name: cls.get_provider_info(name)
            for name in cls.list_initialized_providers()
        }

    @classmethod
    def reset(cls):
        """
        Reset the factory (clear all providers).

        This is primarily useful for testing.
        """
        cls._providers.clear()
        cls._storage_service = None
        logger.info("ProviderFactory reset")
