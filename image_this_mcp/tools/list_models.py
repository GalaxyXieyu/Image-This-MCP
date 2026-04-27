"""List available image generation models and their capabilities."""

import logging
from typing import Annotated, Literal, Optional

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from pydantic import Field

from ..models import ModelRegistry

logger = logging.getLogger(__name__)


def register_list_models_tool(server: FastMCP):
    """Register the list_models tool with the FastMCP server."""

    @server.tool(
        annotations={
            "title": "List available image generation models",
            "readOnlyHint": True,
        }
    )
    def list_models(
        provider: Annotated[
            Optional[str],
            Field(
                description="Filter by provider (e.g., 'gemini', 'jimeng', 'jimeng45'). "
                "If omitted, returns all registered models."
            ),
        ] = None,
        tier: Annotated[
            Optional[Literal["flash", "pro", "standard", "custom"]],
            Field(description="Filter by tier. If omitted, returns all tiers."),
        ] = None,
        capability: Annotated[
            Optional[str],
            Field(
                description="Filter by capability flag (e.g., 'thinking', 'grounding', "
                "'high_resolution', 'text_rendering'). If omitted, no capability filter applied."
            ),
        ] = None,
        include_defaults: Annotated[
            bool,
            Field(
                description="Include provider default model assignments in the response."
            ),
        ] = True,
    ) -> ToolResult:
        """
        List all registered image generation models with their metadata and capabilities.

        Models are registered dynamically at runtime from the ModelRegistry.
        You can filter by provider, tier, or required capability.

        Returns a structured list of models including:
        - id: unique model identifier
        - name: human-readable name
        - provider: provider key
        - tier: performance tier (flash/pro/standard/custom)
        - max_resolution: maximum output resolution in pixels
        - capabilities: supported features (thinking, grounding, etc.)
        - description: model description
        - best_for: recommended use cases
        """
        logger.info(
            f"list_models request: provider={provider}, tier={tier}, capability={capability}"
        )

        # Query the registry
        models = ModelRegistry.filter(
            provider=provider,
            tier=tier,
            capability=capability,
        )

        if not models:
            summary = "No models match the requested filters."
            return ToolResult(
                content=[TextContent(type="text", text=summary)],
                structured_content={
                    "models": [],
                    "count": 0,
                    "filters": {
                        "provider": provider,
                        "tier": tier,
                        "capability": capability,
                    },
                },
            )

        model_list = [m.to_dict() for m in models]

        # Build text summary
        lines = [f"📋 Registered Models ({len(model_list)} total)\n"]

        for m in model_list:
            emoji = m.get("emoji") or "🤖"
            lines.append(
                f"{emoji} **{m['name']}** (`{m['id']}`)"
            )
            lines.append(f"   Provider: {m['provider']} | Tier: {m['tier']} | Max: {m['max_resolution']}px")
            if m.get("description"):
                lines.append(f"   {m['description']}")
            if m.get("best_for"):
                lines.append(f"   Best for: {m['best_for']}")

            caps = m.get("capabilities", {})
            enabled = [k for k, v in caps.items() if v]
            if enabled:
                lines.append(f"   Capabilities: {', '.join(enabled)}")
            lines.append("")

        if include_defaults:
            lines.append("🔧 Provider Defaults:")
            for prov, model_id in ModelRegistry._provider_defaults.items():
                lines.append(f"   {prov}: {model_id}")
            lines.append("")

        full_summary = "\n".join(lines)

        structured_content = {
            "models": model_list,
            "count": len(model_list),
            "filters": {
                "provider": provider,
                "tier": tier,
                "capability": capability,
            },
            "provider_defaults": {
                prov: model_id
                for prov, model_id in ModelRegistry._provider_defaults.items()
            }
            if include_defaults
            else {},
        }

        return ToolResult(
            content=[TextContent(type="text", text=full_summary)],
            structured_content=structured_content,
        )
