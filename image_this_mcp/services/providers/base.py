"""
Abstract base class for image generation providers.

All providers must implement the BaseImageProvider interface to ensure
consistent behavior across different image generation services.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from fastmcp.utilities.types import Image as MCPImage


class BaseImageProvider(ABC):
    """
    Abstract base class: Image generation provider unified interface.

    All image generation providers (Gemini, Jimeng, etc.) must inherit
    from this class and implement all abstract methods.
    """

    provider_name: str
    """Unique identifier for this provider (e.g., 'gemini', 'jimeng')"""

    provider_version: str = "1.0.0"
    """Provider implementation version"""

    @abstractmethod
    def generate_images(
        self,
        prompt: str,
        n: int = 1,
        negative_prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[MCPImage], List[Dict[str, Any]]]:
        """
        Generate images based on text prompt.

        Args:
            prompt: Main generation prompt (text description)
            n: Number of images to generate (default: 1)
            negative_prompt: Optional text describing what to avoid
            aspect_ratio: Optional aspect ratio (e.g., "16:9", "3:4")
            **kwargs: Provider-specific parameters

        Returns:
            Tuple of (images, metadata) where:
            - images: List of MCP Image objects
            - metadata: List of dicts with generation info (model, params, etc.)

        Raises:
            ValidationError: If input parameters are invalid
            ProviderError: If generation fails
        """
        pass

    @abstractmethod
    def edit_image(
        self,
        instruction: str,
        image_data: bytes,
        mime_type: str = "image/png",
        **kwargs
    ) -> Tuple[List[MCPImage], int]:
        """
        Edit an existing image based on natural language instruction.

        Args:
            instruction: Natural language editing instruction
            image_data: Source image bytes
            mime_type: MIME type of source image (default: "image/png")
            **kwargs: Provider-specific parameters

        Returns:
            Tuple of (edited_images, count) where:
            - edited_images: List of edited MCP Image objects
            - count: Number of images generated

        Raises:
            ValidationError: If input image is invalid
            ProviderError: If editing fails
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate provider configuration.

        Returns:
            True if configuration is valid and provider is ready to use

        This should check:
        - Required credentials (API keys, tokens)
        - Network connectivity (optional)
        - Basic API access (optional, may make a test call)
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider metadata and capabilities.

        Returns:
            Dict with provider information:
            - name: Provider name
            - version: Provider version
            - capabilities: Dict of supported features
            - default_params: Default parameter values
        """
        return {
            "name": self.provider_name,
            "version": self.provider_version,
            "capabilities": self._get_capabilities(),
        }

    def _get_capabilities(self) -> Dict[str, Any]:
        """
        Get provider-specific capabilities.

        Override this method to declare provider capabilities.

        Returns:
            Dict with capability flags and limits
        """
        return {
            "max_images_per_request": 1,
            "supported_aspect_ratios": [],
            "supports_editing": False,
            "supports_reference_images": False,
            "max_resolution": None,
        }
