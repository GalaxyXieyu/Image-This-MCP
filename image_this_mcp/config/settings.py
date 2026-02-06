from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

_ENV_LOADED = False


def load_env() -> None:
    """加载 .env（优先使用显式路径，其次尝试项目根目录）。"""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    dotenv_path = os.getenv("NANOBANANA_ENV_PATH") or os.getenv("DOTENV_PATH")
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
    else:
        load_dotenv()
        repo_env = Path(__file__).resolve().parents[2] / ".env"
        if repo_env.exists():
            load_dotenv(dotenv_path=repo_env)

    _ENV_LOADED = True

from ..core.exceptions import ADCConfigurationError
from .constants import AUTH_ERROR_MESSAGES


class ModelTier(str, Enum):
    """Model selection options."""
    FLASH = "flash"  # Speed-optimized (Gemini 2.5 Flash)
    PRO = "pro"      # Quality-optimized (Gemini 3 Pro)
    AUTO = "auto"    # Automatic selection


class AuthMethod(Enum):
    """Authentication method options."""
    API_KEY = "api_key"      # Developer API + API Key
    VERTEX_AI = "vertex_ai"  # Vertex AI API + ADC
    AUTO = "auto"            # Auto-detect


class ThinkingLevel(str, Enum):
    """Gemini 3 thinking levels for advanced reasoning."""
    LOW = "low"      # Minimal latency, less reasoning
    HIGH = "high"    # Maximum reasoning (default for Pro)


class MediaResolution(str, Enum):
    """Media resolution for vision processing."""
    LOW = "low"      # Faster, less detail
    MEDIUM = "medium"  # Balanced
    HIGH = "high"    # Maximum detail


@dataclass
class ServerConfig:
    """Server configuration settings."""

    gemini_api_key: Optional[str] = None
    server_name: str = "image-this-mcp"
    transport: str = "stdio"  # stdio or http
    host: str = "127.0.0.1"
    port: int = 9000
    mask_error_details: bool = False
    max_concurrent_requests: int = 10
    image_output_dir: str = ""
    auth_method: AuthMethod = AuthMethod.AUTO
    gcp_project_id: Optional[str] = None
    gcp_region: str = "us-central1"
    api_base_url: Optional[str] = None  # Custom API base URL for third-party Banana API
    default_provider: str = "gemini"  # Default image provider: 'gemini' or 'jimeng'

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        load_env()

        # Auth method
        auth_method_str = os.getenv("NANOBANANA_AUTH_METHOD", "auto").lower()
        try:
            auth_method = AuthMethod(auth_method_str)
        except ValueError:
            auth_method = AuthMethod.AUTO

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gcp_project = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_region = os.getenv("GCP_REGION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
        api_base_url = os.getenv("GEMINI_API_BASE_URL") or os.getenv("BANANA_API_BASE_URL")

        # Validation logic
        if auth_method == AuthMethod.API_KEY:
            if not api_key:
                raise ValueError(AUTH_ERROR_MESSAGES["api_key_required"])
        
        elif auth_method == AuthMethod.VERTEX_AI:
            if not gcp_project:
                raise ADCConfigurationError(AUTH_ERROR_MESSAGES["vertex_ai_project_required"])
        
        else:  # AUTO
            if not api_key:
                if not gcp_project:
                    raise ValueError(AUTH_ERROR_MESSAGES["no_auth_configured"])
                auth_method = AuthMethod.VERTEX_AI
            else:
                auth_method = AuthMethod.API_KEY

        # Handle image output directory
        output_dir = os.getenv("IMAGE_OUTPUT_DIR", "").strip()
        if not output_dir:
            # Default to ~/image-this in user's home directory for better compatibility
            output_dir = str(Path.home() / "image-this")

        # Convert to absolute path and ensure it exists
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        return cls(
            gemini_api_key=api_key,
            auth_method=auth_method,
            gcp_project_id=gcp_project,
            gcp_region=gcp_region,
            transport=os.getenv("FASTMCP_TRANSPORT", "stdio"),
            host=os.getenv("FASTMCP_HOST", "127.0.0.1"),
            port=int(os.getenv("FASTMCP_PORT", "9000")),
            mask_error_details=os.getenv("FASTMCP_MASK_ERRORS", "false").lower() == "true",
            image_output_dir=str(output_path),
            api_base_url=api_base_url,
            default_provider=os.getenv("IMAGE_PROVIDER", "gemini"),
        )


@dataclass
class BaseModelConfig:
    """Shared base configuration for all models."""
    max_images_per_request: int = 4
    max_inline_image_size: int = 20 * 1024 * 1024  # 20MB
    default_image_format: str = "png"
    request_timeout: int = 60  # seconds


@dataclass
class FlashImageConfig(BaseModelConfig):
    """Gemini 2.5 Flash Image configuration (speed-optimized)."""
    model_name: str = "gemini-2.5-flash-image"
    max_resolution: int = 1024
    supports_thinking: bool = False
    supports_grounding: bool = False
    supports_media_resolution: bool = False


@dataclass
class ProImageConfig(BaseModelConfig):
    """Gemini 3 Pro Image configuration (quality-optimized)."""
    model_name: str = "gemini-3-pro-image-preview"
    max_resolution: int = 3840  # 4K
    default_resolution: str = "high"  # low/medium/high
    default_thinking_level: ThinkingLevel = ThinkingLevel.HIGH
    default_media_resolution: MediaResolution = MediaResolution.HIGH
    supports_thinking: bool = True
    supports_grounding: bool = True
    supports_media_resolution: bool = True
    enable_search_grounding: bool = True
    request_timeout: int = 90  # Pro model needs more time for 4K


@dataclass
class ModelSelectionConfig:
    """Configuration for intelligent model selection."""
    default_tier: ModelTier = ModelTier.PRO
    auto_quality_keywords: List[str] = field(default_factory=lambda: [
        "4k", "high quality", "professional", "production",
        "high-res", "high resolution", "detailed", "sharp", "crisp",
        "hd", "ultra", "premium", "magazine", "print"
    ])
    auto_speed_keywords: List[str] = field(default_factory=lambda: [
        "quick", "fast", "draft", "prototype", "sketch",
        "rapid", "rough", "temporary", "test"
    ])

    @classmethod
    def from_env(cls) -> "ModelSelectionConfig":
        """Load model selection config from environment."""
        load_env()

        model_tier_str = os.getenv("NANOBANANA_MODEL", "pro").lower()
        try:
            default_tier = ModelTier(model_tier_str)
        except ValueError:
            default_tier = ModelTier.AUTO

        return cls(default_tier=default_tier)


@dataclass
class GeminiConfig:
    """Legacy Gemini API configuration (backward compatibility)."""
    model_name: str = "gemini-2.5-flash-image"
    max_images_per_request: int = 4
    max_inline_image_size: int = 20 * 1024 * 1024  # 20MB
    default_image_format: str = "png"
    request_timeout: int = 60  # seconds - increased for image generation


@dataclass
class JimengConfig:
    """Jimeng AI (Volcengine) configuration."""

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    api_endpoint: str = "visual.volcengineapi.com"
    region: str = "cn-north-1"
    service: str = "cv"
    version: str = "2022-08-31"
    default_width: int = 1536  # 3:4 portrait ratio
    default_height: int = 2048
    request_timeout: int = 120  # seconds - Jimeng may be slower
    max_retries: int = 3  # Number of retries for failed requests
    retry_delay: int = 5  # Initial retry delay in seconds

    @classmethod
    def from_env(cls) -> "JimengConfig":
        """Load configuration from environment variables."""
        load_env()

        return cls(
            access_key=os.getenv("JIMENG_ACCESS_KEY"),
            secret_key=os.getenv("JIMENG_SECRET_KEY"),
            request_timeout=int(os.getenv("JIMENG_TIMEOUT", "120")),
        )

    def validate_credentials(self) -> bool:
        """Validate that required credentials are present."""
        return bool(self.access_key and self.secret_key)


@dataclass
class Jimeng45Config:
    """Jimeng 4.5 (Seedream 4.5 via Ark API) configuration."""

    api_key: Optional[str] = None
    api_endpoint: str = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    model: str = "doubao-seedream-4.5"  # or custom endpoint ID like 'ep-xxxx'
    default_size: str = "1728x2304"  # 3:4 portrait ratio for Xiaohongshu
    response_format: str = "b64_json"  # 'b64_json' or 'url'
    request_timeout: int = 120  # seconds
    max_retries: int = 3  # Number of retries for failed requests
    retry_delay: int = 5  # Initial retry delay in seconds
    watermark: bool = False  # Add AI-generated watermark
    sequential_image_generation: str = "disabled"  # 'disabled' or 'auto'

    # Supported sizes (format: "WIDTHxHEIGHT")
    # Total pixels must be between [3686400, 16777216] for Jimeng 4.5
    SUPPORTED_SIZES = [
        "1024x1024",  # 1:1
        "1024x1365",  # 3:4
        "1024x1536",  # 2:3
        "1728x2304",  # 3:4 (Xiaohongshu optimized)
        "1536x2048",  # 3:4
        "2048x2731",  # 3:4
        "2304x1728",  # 4:3
        "2048x1536",  # 4:3
        "1920x1080",  # 16:9
        "1080x1920",  # 9:16
    ]

    @classmethod
    def from_env(cls) -> "Jimeng45Config":
        """Load configuration from environment variables."""
        load_env()

        return cls(
            api_key=os.getenv("JIMENG45_API_KEY") or os.getenv("ARK_API_KEY"),
            api_endpoint=os.getenv(
                "JIMENG45_API_ENDPOINT",
                "https://ark.cn-beijing.volces.com/api/v3/images/generations"
            ),
            model=os.getenv("JIMENG45_MODEL", "doubao-seedream-4.5"),
            default_size=os.getenv("JIMENG45_SIZE", "1728x2304"),
            response_format=os.getenv("JIMENG45_RESPONSE_FORMAT", "b64_json"),
            request_timeout=int(os.getenv("JIMENG45_TIMEOUT", "120")),
            max_retries=int(os.getenv("JIMENG45_MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("JIMENG45_RETRY_DELAY", "5")),
            watermark=os.getenv("JIMENG45_WATERMARK", "false").lower() == "true",
            sequential_image_generation=os.getenv("JIMENG45_SEQUENTIAL", "disabled"),
        )

    def validate_credentials(self) -> bool:
        """Validate that required credentials are present."""
        return bool(self.api_key)

    def validate_size(self, size: str) -> bool:
        """Validate that the size is supported."""
        if size in self.SUPPORTED_SIZES:
            return True

        # Check pixel count
        try:
            width, height = map(int, size.lower().split("x"))
            total_pixels = width * height
            return 3686400 <= total_pixels <= 16777216
        except (ValueError, AttributeError):
            return False
