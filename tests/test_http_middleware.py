from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from image_this_mcp.core.http_middleware import FixedTokenAuthMiddleware


async def protected(_request):
    return JSONResponse({"ok": True})


def test_fixed_token_auth_middleware_allows_health_and_valid_token():
    app = Starlette(
        routes=[
            Route("/health", protected),
            Route("/mcp", protected),
        ],
        middleware=[
            Middleware(
                FixedTokenAuthMiddleware,
                token="secret-token",
                header_name="Authorization",
            )
        ],
    )

    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/mcp").status_code == 401
    assert client.get("/mcp", headers={"Authorization": "Bearer secret-token"}).status_code == 200
