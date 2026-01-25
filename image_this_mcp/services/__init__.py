"""Service registry for dependency injection."""

import os
from typing import Optional, List

from ..config.settings import (
    FlashImageConfig,
    GeminiConfig,
    JimengConfig,
    ModelSelectionConfig,
    ProImageConfig,
    ServerConfig,
)
from .enhanced_image_service import EnhancedImageService
from .file_image_service import FileImageService
from .file_service import FileService
from .files_api_service import FilesAPIService
from .gemini_client import GeminiClient
from .image_database_service import ImageDatabaseService
from .image_storage_service import ImageStorageService
from .maintenance_service import MaintenanceService
from .model_selector import ModelSelector
from .pro_image_service import ProImageService
from .providers import ProviderFactory

# Global service instances (initialized by the server)
_gemini_client: Optional[GeminiClient] = None
_file_image_service: Optional[FileImageService] = None
_file_service: Optional[FileService] = None
_enhanced_image_service: Optional[EnhancedImageService] = None
_files_api_service: Optional[FilesAPIService] = None
_image_database_service: Optional[ImageDatabaseService] = None
_image_storage_service: Optional[ImageStorageService] = None
_maintenance_service: Optional[MaintenanceService] = None

# Multi-model support services
_flash_gemini_client: Optional[GeminiClient] = None
_pro_gemini_client: Optional[GeminiClient] = None
_pro_image_service: Optional[ProImageService] = None
_model_selector: Optional[ModelSelector] = None


def initialize_services(server_config: ServerConfig, gemini_config: GeminiConfig):
    """Initialize all services with configurations (legacy + multi-model)."""
    global \
        _gemini_client, \
        _file_image_service, \
        _file_service, \
        _enhanced_image_service, \
        _files_api_service, \
        _image_database_service, \
        _image_storage_service, \
        _maintenance_service, \
        _flash_gemini_client, \
        _pro_gemini_client, \
        _pro_image_service, \
        _model_selector

    # Initialize core services (legacy compatibility)
    _gemini_client = GeminiClient(server_config, gemini_config)
    _file_image_service = FileImageService(_gemini_client, gemini_config, server_config)
    _file_service = FileService(_gemini_client)

    # Initialize enhanced services for workflows.md implementation
    out_dir = server_config.image_output_dir
    _image_database_service = ImageDatabaseService(db_path=os.path.join(out_dir, "images.db"))
    # Use a subdirectory within the configured output directory for temp images
    temp_images_dir = os.path.join(out_dir, "temp_images")
    _image_storage_service = ImageStorageService(gemini_config, temp_images_dir)
    _files_api_service = FilesAPIService(_gemini_client, _image_database_service)
    _enhanced_image_service = EnhancedImageService(
        _gemini_client, _files_api_service, _image_database_service, gemini_config, out_dir
    )
    _maintenance_service = MaintenanceService(_files_api_service, _image_database_service, out_dir)

    # Initialize multi-model support services
    flash_config = FlashImageConfig()
    pro_config = ProImageConfig()
    selection_config = ModelSelectionConfig.from_env()

    # Create separate Gemini clients for each model
    _flash_gemini_client = GeminiClient(server_config, flash_config)
    _pro_gemini_client = GeminiClient(server_config, pro_config)

    # Create Pro image service (Flash uses existing _file_image_service)
    _pro_image_service = ProImageService(
        _pro_gemini_client,
        pro_config,
        _image_storage_service
    )

    # Create model selector
    _model_selector = ModelSelector(
        _file_image_service,  # Flash service
        _pro_image_service,   # Pro service
        selection_config
    )

    # Initialize multi-provider support
    jimeng_config = JimengConfig.from_env()

    # Register storage service with factory
    ProviderFactory.set_storage_service(_image_storage_service)

    # Initialize all providers (Gemini and Jimeng)
    ProviderFactory.initialize_all_providers(
        server_config=server_config,
        gemini_config=flash_config,  # Use Flash config as default
        jimeng_config=jimeng_config
    )


def get_image_service() -> FileImageService:
    """Get the image service instance."""
    if _file_image_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _file_image_service


def get_file_service() -> FileService:
    """Get the file service instance."""
    if _file_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _file_service


def get_gemini_client() -> GeminiClient:
    """Get the Gemini client instance."""
    if _gemini_client is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _gemini_client


def get_file_image_service() -> FileImageService:
    """Get the file image service instance."""
    if _file_image_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _file_image_service


def get_enhanced_image_service() -> EnhancedImageService:
    """Get the enhanced image service instance (workflows.md implementation)."""
    if _enhanced_image_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _enhanced_image_service


def get_files_api_service() -> FilesAPIService:
    """Get the Files API service instance."""
    if _files_api_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _files_api_service


def get_image_database_service() -> ImageDatabaseService:
    """Get the image database service instance."""
    if _image_database_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _image_database_service


def get_maintenance_service() -> MaintenanceService:
    """Get the maintenance service instance."""
    if _maintenance_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _maintenance_service


def get_image_storage_service() -> ImageStorageService:
    """Get the image storage service instance."""
    if _image_storage_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _image_storage_service


def get_pro_image_service() -> ProImageService:
    """Get the Pro image service instance."""
    if _pro_image_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _pro_image_service


def get_model_selector() -> ModelSelector:
    """Get the model selector instance."""
    if _model_selector is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _model_selector


# Multi-provider support functions

def get_provider_factory() -> type[ProviderFactory]:
    """Get the provider factory class."""
    return ProviderFactory


def get_provider(name: str):
    """Get a specific provider instance."""
    return ProviderFactory.get_provider(name)


def get_default_provider():
    """Get the default provider (Gemini)."""
    return ProviderFactory.get_default_provider()


def list_providers() -> List[str]:
    """List all available provider names."""
    return ProviderFactory.list_providers()


def list_initialized_providers() -> List[str]:
    """List all initialized provider names."""
    return ProviderFactory.list_initialized_providers()
