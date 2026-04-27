"""Utility helpers for the generate_image tool."""

from ...config.settings import ModelTier


def get_enhanced_image_service(selected_tier: ModelTier | None = None):
    """Get the enhanced image service instance."""
    from ...services import get_enhanced_image_service, get_pro_enhanced_image_service

    if selected_tier == ModelTier.PRO:
        return get_pro_enhanced_image_service()
    return get_enhanced_image_service()
