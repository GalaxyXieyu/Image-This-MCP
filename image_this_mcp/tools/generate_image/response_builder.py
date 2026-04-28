"""Response formatting for the generate_image tool."""

from typing import List, Optional

from fastmcp.utilities.types import Image as MCPImage
from mcp.types import TextContent
from fastmcp.tools.tool import ToolResult


def build_gemini_response(
    thumbnail_images: List[MCPImage],
    metadata: List[dict],
    detected_mode: str,
    selected_tier: str,
    model_info: dict,
    thinking_level: str,
    resolution: str,
    enable_grounding: bool,
    input_image_paths: Optional[List[str]],
    file_id: Optional[str],
    aspect_ratio: Optional[str],
    prompt: str,
    negative_prompt: Optional[str],
    n: int,
) -> ToolResult:
    """Build ToolResult for a Gemini provider response."""
    action_verb = "Edited" if detected_mode == "edit" else "Generated"
    model_name = model_info["name"]
    model_emoji = model_info["emoji"]
    summary_lines = [
        f"✅ {action_verb} {len(metadata)} image(s) with {model_emoji} {model_name}.",
        f"📊 **Model**: {selected_tier.upper()} tier",
    ]

    if selected_tier.upper() == "PRO":
        summary_lines.append(f"🧠 **Thinking Level**: {thinking_level}")
        summary_lines.append(f"📏 **Resolution**: {resolution}")
        if enable_grounding:
            summary_lines.append("🔍 **Grounding**: Enabled (Google Search)")
    summary_lines.append("")

    if detected_mode == "edit":
        if file_id:
            summary_lines.append(f"📎 **Edit Source**: Files API {file_id}")
        elif input_image_paths and len(input_image_paths) == 1:
            summary_lines.append(f"📁 **Edit Source**: {input_image_paths[0]}")
    elif input_image_paths:
        summary_lines.append(
            f"🖼️ Conditioned on {len(input_image_paths)} input image(s): {', '.join(input_image_paths)}"
        )
    if aspect_ratio and detected_mode == "generate":
        summary_lines.append(f"📐 Aspect ratio: {aspect_ratio}")

    result_label = "Edited Images" if detected_mode == "edit" else "Generated Images"
    summary_lines.append(f"\n📁 **{result_label}:**")
    for i, meta in enumerate(metadata, 1):
        if not meta or not isinstance(meta, dict):
            summary_lines.append(f"  {i}. ❌ Invalid metadata entry")
            continue

        size_bytes = meta.get("size_bytes", 0)
        size_mb = round(size_bytes / (1024 * 1024), 1) if size_bytes else 0
        full_path = meta.get("full_path", "Unknown path")
        width = meta.get("width", "?")
        height = meta.get("height", "?")

        extra_info = ""
        if detected_mode == "edit":
            files_api_info = meta.get("files_api") or {}
            if files_api_info.get("name"):
                extra_info += f" • 🌐 Files API: {files_api_info['name']}"
            if meta.get("parent_file_id"):
                extra_info += f" • 👨‍👩‍👧 Parent: {meta.get('parent_file_id')}"

        summary_lines.append(
            f"  {i}. `{full_path}`\n"
            f"     📏 {width}x{height} • 💾 {size_mb}MB{extra_info}"
        )

    summary_lines.append(
        "\n🖼️ **Thumbnail previews shown below** (actual images saved to disk)"
    )

    full_summary = "\n".join(summary_lines)
    content = [TextContent(type="text", text=full_summary), *thumbnail_images]

    structured_content = {
        "mode": detected_mode,
        "model_tier": selected_tier,
        "model_name": model_info["name"],
        "model_id": model_info["model_id"],
        "requested_tier": selected_tier,
        "auto_selected": selected_tier == "auto",
        "thinking_level": thinking_level if selected_tier.upper() == "PRO" else None,
        "resolution": resolution,
        "grounding_enabled": enable_grounding if selected_tier.upper() == "PRO" else False,
        "requested": n,
        "returned": len(thumbnail_images),
        "negative_prompt_applied": bool(negative_prompt),
        "used_input_images": bool(input_image_paths) or bool(file_id),
        "input_image_paths": input_image_paths or [],
        "input_image_count": len(input_image_paths) if input_image_paths else (1 if file_id else 0),
        "aspect_ratio": aspect_ratio,
        "source_file_id": file_id,
        "edit_instruction": prompt if detected_mode == "edit" else None,
        "generation_prompt": prompt if detected_mode == "generate" else None,
        "output_method": "file_system_with_files_api",
        "workflow": f"workflows.md_{detected_mode}_sequence",
        "images": metadata,
        "file_paths": [
            m.get("full_path")
            for m in metadata
            if m and isinstance(m, dict) and m.get("full_path")
        ],
        "files_api_ids": [
            m.get("files_api", {}).get("name")
            for m in metadata
            if m and isinstance(m, dict) and m.get("files_api", {}) and m.get("files_api", {}).get("name")
        ],
        "parent_relationships": [
            (m.get("parent_file_id"), m.get("files_api", {}).get("name"))
            for m in metadata
            if m and isinstance(m, dict)
        ]
        if detected_mode == "edit"
        else [],
        "total_size_mb": round(
            sum(m.get("size_bytes", 0) for m in metadata if m and isinstance(m, dict))
            / (1024 * 1024),
            2,
        ),
    }

    return ToolResult(content=content, structured_content=structured_content)


def build_jimeng_response(
    thumbnail_images: List[MCPImage],
    metadata: List[dict],
    detected_mode: str,
    input_image_paths: Optional[List[str]],
    aspect_ratio: Optional[str],
    prompt: str,
    n: int,
) -> ToolResult:
    """Build ToolResult for a Jimeng provider response."""
    action_verb = "Edited" if detected_mode == "edit" else "Generated"
    model_id = metadata[0].get("model") if metadata else "jimeng_t2i_v40"
    summary_lines = [
        f"✅ {action_verb} {len(metadata)} image(s) with Jimeng AI ({model_id})."
    ]

    if detected_mode == "edit" and input_image_paths:
        summary_lines.append(f"📁 **Edit Source**: {input_image_paths[0]}")
    if aspect_ratio and detected_mode == "generate":
        summary_lines.append(f"📐 Aspect ratio: {aspect_ratio}")

    result_label = "Edited Images" if detected_mode == "edit" else "Generated Images"
    summary_lines.append(f"\n📁 **{result_label}:**")
    for i, meta in enumerate(metadata, 1):
        if not meta or not isinstance(meta, dict):
            summary_lines.append(f"  {i}. ❌ Invalid metadata entry")
            continue

        size_bytes = meta.get("size_bytes", 0)
        size_mb = round(size_bytes / (1024 * 1024), 1) if size_bytes else 0
        full_path = meta.get("full_path", "Unknown path")
        width = meta.get("width", "?")
        height = meta.get("height", "?")

        summary_lines.append(
            f"  {i}. `{full_path}`\n"
            f"     📏 {width}x{height} • 💾 {size_mb}MB"
        )

    summary_lines.append(
        "\n🖼️ **Thumbnail previews shown below** (actual images saved to disk)"
    )

    full_summary = "\n".join(summary_lines)
    content = [TextContent(type="text", text=full_summary), *thumbnail_images]

    structured_content = {
        "mode": detected_mode,
        "model_tier": "jimeng",
        "model_name": "Jimeng AI",
        "model_id": model_id,
        "requested_tier": None,
        "auto_selected": False,
        "thinking_level": None,
        "resolution": None,
        "grounding_enabled": False,
        "requested": n,
        "returned": len(thumbnail_images),
        "negative_prompt_applied": False,
        "used_input_images": bool(input_image_paths),
        "input_image_paths": input_image_paths or [],
        "input_image_count": len(input_image_paths) if input_image_paths else 0,
        "aspect_ratio": aspect_ratio,
        "source_file_id": None,
        "edit_instruction": prompt if detected_mode == "edit" else None,
        "generation_prompt": prompt if detected_mode == "generate" else None,
        "output_method": "file_system",
        "workflow": f"jimeng_{detected_mode}",
        "images": metadata,
        "file_paths": [
            m.get("full_path")
            for m in metadata
            if m and isinstance(m, dict) and m.get("full_path")
        ],
        "files_api_ids": [],
        "parent_relationships": [],
        "total_size_mb": round(
            sum(m.get("size_bytes", 0) for m in metadata if m and isinstance(m, dict))
            / (1024 * 1024),
            2,
        ),
    }

    return ToolResult(content=content, structured_content=structured_content)


def build_openai_response(
    thumbnail_images: List[MCPImage],
    metadata: List[dict],
    detected_mode: str,
    input_image_paths: Optional[List[str]],
    aspect_ratio: Optional[str],
    prompt: str,
    n: int,
) -> ToolResult:
    """Build ToolResult for an OpenAI provider response."""
    action_verb = "Edited" if detected_mode == "edit" else "Generated"
    model_id = metadata[0].get("model") if metadata else "gpt-image-2"
    summary_lines = [
        f"✅ {action_verb} {len(metadata)} image(s) with OpenAI ({model_id})."
    ]

    if detected_mode == "edit" and input_image_paths:
        summary_lines.append(f"📁 **Edit Source**: {input_image_paths[0]}")
    if aspect_ratio and detected_mode == "generate":
        summary_lines.append(f"📐 Aspect ratio: {aspect_ratio}")

    result_label = "Edited Images" if detected_mode == "edit" else "Generated Images"
    summary_lines.append(f"\n📁 **{result_label}:**")
    for i, meta in enumerate(metadata, 1):
        if not meta or not isinstance(meta, dict):
            summary_lines.append(f"  {i}. ❌ Invalid metadata entry")
            continue

        size_bytes = meta.get("size_bytes", 0)
        size_mb = round(size_bytes / (1024 * 1024), 1) if size_bytes else 0
        full_path = meta.get("full_path", "Unknown path")
        width = meta.get("width", "?")
        height = meta.get("height", "?")
        quality = meta.get("quality", "")
        style = meta.get("style", "")

        extra = ""
        if quality:
            extra += f" • Quality: {quality}"
        if style:
            extra += f" • Style: {style}"

        summary_lines.append(
            f"  {i}. `{full_path}`\n"
            f"     📏 {width}x{height} • 💾 {size_mb}MB{extra}"
        )

    summary_lines.append(
        "\n🖼️ **Thumbnail previews shown below** (actual images saved to disk)"
    )

    full_summary = "\n".join(summary_lines)
    content = [TextContent(type="text", text=full_summary), *thumbnail_images]

    structured_content = {
        "mode": detected_mode,
        "model_tier": "openai",
        "model_name": "OpenAI",
        "model_id": model_id,
        "requested_tier": None,
        "auto_selected": False,
        "thinking_level": None,
        "resolution": None,
        "grounding_enabled": False,
        "requested": n,
        "returned": len(thumbnail_images),
        "negative_prompt_applied": False,
        "used_input_images": bool(input_image_paths),
        "input_image_paths": input_image_paths or [],
        "input_image_count": len(input_image_paths) if input_image_paths else 0,
        "aspect_ratio": aspect_ratio,
        "source_file_id": None,
        "edit_instruction": prompt if detected_mode == "edit" else None,
        "generation_prompt": prompt if detected_mode == "generate" else None,
        "output_method": "file_system",
        "workflow": f"openai_{detected_mode}",
        "images": metadata,
        "file_paths": [
            m.get("full_path")
            for m in metadata
            if m and isinstance(m, dict) and m.get("full_path")
        ],
        "files_api_ids": [],
        "parent_relationships": [],
        "total_size_mb": round(
            sum(m.get("size_bytes", 0) for m in metadata if m and isinstance(m, dict))
            / (1024 * 1024),
            2,
        ),
    }

    return ToolResult(content=content, structured_content=structured_content)


def build_empty_response(detected_mode: str, prompt: str) -> ToolResult:
    """Build a fallback response when no images were generated."""
    summary = "❌ No images were generated. Please check the logs for details."
    content = [TextContent(type="text", text=summary)]
    return ToolResult(content=content)
