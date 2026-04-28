from fastmcp import FastMCP
import logging
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import uvicorn
from ..config.settings import ServerConfig
from .http_middleware import FixedTokenAuthMiddleware


class NanoBananaMCP:
    """Main FastMCP server class."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize FastMCP server
        self.server = FastMCP(
            name=config.server_name,
            instructions=self._get_server_instructions(),
            mask_error_details=config.mask_error_details,
        )

        # Register components
        self._register_tools()
        self._register_resources()
        self._register_prompts()
        self._register_http_routes()

    def _get_server_instructions(self) -> str:
        """Get server description and instructions."""
        return (
            "This server exposes image generation & editing powered by "
            "Gemini 3.1 Flash Image Preview (aka 'nano banana'). It returns images "
            "as real MCP image content blocks, and also provides structured "
            "JSON with metadata and reproducibility hints."
        )

    def _register_tools(self):
        """Register all tools with the server."""
        from ..tools.generate_image import register_generate_image_tool
        from ..tools.upload_file import register_upload_file_tool
        from ..tools.output_stats import register_output_stats_tool
        from ..tools.maintenance import register_maintenance_tool
        from ..tools.list_models import register_list_models_tool

        register_generate_image_tool(self.server)
        register_upload_file_tool(self.server)
        register_output_stats_tool(self.server)
        register_maintenance_tool(self.server)
        register_list_models_tool(self.server)

    def _register_resources(self):
        """Register all resources with the server."""
        from ..resources.file_metadata import register_file_metadata_resource
        from ..resources.template_catalog import register_template_catalog_resource
        from ..resources.operation_status import register_operation_status_resources

        register_file_metadata_resource(self.server)
        register_template_catalog_resource(self.server)
        register_operation_status_resources(self.server)

    def _register_prompts(self):
        """Register all prompts with the server."""
        from ..prompts.photography import register_photography_prompts
        from ..prompts.design import register_design_prompts
        from ..prompts.editing import register_editing_prompts

        register_photography_prompts(self.server)
        register_design_prompts(self.server)
        register_editing_prompts(self.server)

    def _register_http_routes(self):
        """Register custom HTTP routes for remote deployments."""

        @self.server.custom_route("/health", methods=["GET"], include_in_schema=False)
        async def health(_request: Request):
            return JSONResponse({"ok": True, "server": self.config.server_name})

    def run(self):
        """Start the server."""
        if self.config.transport == "http":
            middleware = []
            if self.config.mcp_auth_token:
                middleware.append(
                    Middleware(
                        FixedTokenAuthMiddleware,
                        token=self.config.mcp_auth_token,
                        header_name=self.config.mcp_auth_header,
                    )
                )
            app = self.server.http_app(middleware=middleware, transport="http")
            uvicorn.run(app, host=self.config.host, port=self.config.port)
        else:
            self.server.run()
