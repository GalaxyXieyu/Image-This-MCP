import asyncio
import inspect

from fastmcp import FastMCP

from image_this_mcp.models import ModelRegistry, register_default_models
from image_this_mcp.tools.generate_image import register_generate_image_tool
from image_this_mcp.tools.list_models import register_list_models_tool


def test_generate_image_tool_exposes_model_parameter():
    server = FastMCP("test")
    register_generate_image_tool(server)
    tool = asyncio.run(server.get_tool("generate_image"))
    assert "model" in inspect.signature(tool.fn).parameters


def test_list_models_merges_jimeng_family():
    ModelRegistry.reset()
    register_default_models()

    server = FastMCP("test")
    register_list_models_tool(server)
    tool = asyncio.run(server.get_tool("list_models"))
    result = tool.fn(provider="jimeng", include_defaults=True)
    data = result.structured_content

    ids = {m["id"] for m in data["models"]}
    assert "jimeng" in ids
    assert "doubao-seedream-4-5-251128" in ids
    assert "jimeng45" not in data["provider_defaults"]
