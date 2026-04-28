from io import BytesIO
from unittest.mock import Mock

from PIL import Image as PILImage

from image_this_mcp.config.settings import GeminiConfig, ServerConfig
from image_this_mcp.services.file_image_service import FileImageService


def _png_bytes() -> bytes:
    image = PILImage.new("RGB", (16, 16), color="red")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_save_external_image_attaches_artifact_metadata(tmp_path):
    artifact_service = Mock()
    artifact_service.validate_config.return_value = True
    artifact_service.publish_file.return_value = Mock()
    artifact_service.build_metadata.return_value = {
        "artifact_url": "https://minio.example.com/image-this/test.png",
        "storage_provider": "minio",
    }

    service = FileImageService(
        gemini_client=Mock(),
        gemini_config=GeminiConfig(),
        server_config=ServerConfig(image_output_dir=str(tmp_path)),
        artifact_service=artifact_service,
    )

    _thumb, metadata = service.save_external_image(_png_bytes(), mime_type="image/png")

    assert metadata["artifact_url"] == "https://minio.example.com/image-this/test.png"
    assert metadata["storage_provider"] == "minio"
    artifact_service.publish_file.assert_called_once()
