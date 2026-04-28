"""HTTP middleware for remote MCP deployments."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class FixedTokenAuthMiddleware(BaseHTTPMiddleware):
    """Simple shared-secret header auth for HTTP MCP deployments."""

    def __init__(self, app, token: str, header_name: str = "Authorization", exempt_paths: set[str] | None = None):
        super().__init__(app)
        self.token = token
        self.header_name = header_name
        self.exempt_paths = exempt_paths or {"/health", "/healthz"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        value = request.headers.get(self.header_name)
        if not self._is_authorized(value):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    def _is_authorized(self, header_value: str | None) -> bool:
        if not header_value:
            return False
        if self.header_name.lower() == "authorization":
            return header_value == f"Bearer {self.token}" or header_value == self.token
        return header_value == self.token
