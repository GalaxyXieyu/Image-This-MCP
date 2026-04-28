import httpx

from image_this_mcp.config.settings import OpenAIConfig
from image_this_mcp.services.providers.openai_provider import OpenAIProvider


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`\x00\x00"
    b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_openai_provider_handles_task_based_generation():
    """ToAPIs-style generation.task responses should be polled to completion."""
    task_id = "tsk_img_test_123"
    call_counts = {"poll": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/images/generations" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "id": task_id,
                    "object": "generation.task",
                    "model": "gpt-image-2",
                    "status": "queued",
                    "progress": 0,
                },
            )

        if request.url.path == f"/v1/images/generations/{task_id}" and request.method == "GET":
            call_counts["poll"] += 1
            if call_counts["poll"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "id": task_id,
                        "object": "generation.task",
                        "model": "gpt-image-2",
                        "status": "in_progress",
                        "progress": 10,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": task_id,
                    "object": "generation.task",
                    "model": "gpt-image-2",
                    "status": "completed",
                    "progress": 100,
                    "result": {
                        "data": [
                            {
                                "url": "https://files.toapis.com/generated/test.png",
                                "size": "1:1",
                            }
                        ]
                    },
                },
            )

        if request.url == httpx.URL("https://files.toapis.com/generated/test.png"):
            return httpx.Response(200, content=PNG_BYTES, headers={"Content-Type": "image/png"})

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    config = OpenAIConfig(
        api_key="test-key",
        base_url="https://toapis.com/v1",
        default_model="gpt-image-2",
        poll_interval=0,
        max_poll_seconds=5,
        user_agent="pytest-agent",
    )
    provider = OpenAIProvider(config)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))

    images, metadata = provider.generate_images(
        prompt="A test image",
        n=1,
        aspect_ratio="1:1",
    )

    assert len(images) == 1
    assert images[0].data == PNG_BYTES
    assert len(metadata) == 1
    assert metadata[0]["task_id"] == task_id
    assert metadata[0]["status"] == "completed"
    assert metadata[0]["image_url"] == "https://files.toapis.com/generated/test.png"
    assert call_counts["poll"] >= 2
