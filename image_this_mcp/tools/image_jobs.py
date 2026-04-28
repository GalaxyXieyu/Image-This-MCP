"""Async image job tools."""

import logging
from typing import Annotated, Literal, Optional

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from pydantic import Field

from .. import services
from ..core.exceptions import ValidationError


def register_image_job_tools(server: FastMCP):
    """Register async image job submission and polling tools."""

    @server.tool(
        annotations={
            "title": "Submit an async image generation job",
            "readOnlyHint": True,
            "openWorldHint": True,
        }
    )
    def submit_image_job(
        prompt: Annotated[str, Field(description="Prompt for image generation", min_length=1, max_length=8192)],
        provider: Annotated[Literal["gemini", "jimeng", "openai", "auto"] | None, Field(description="Provider family")] = "auto",
        model: Annotated[Optional[str], Field(description="Explicit model id from list_models")] = None,
        n: Annotated[int, Field(description="Requested image count", ge=1, le=4)] = 1,
        negative_prompt: Annotated[Optional[str], Field(description="Things to avoid", max_length=1024)] = None,
        system_instruction: Annotated[Optional[str], Field(description="Gemini-specific system guidance", max_length=512)] = None,
        model_tier: Annotated[Optional[str], Field(description="Gemini tier: flash/pro/auto")] = "flash",
        resolution: Annotated[Optional[str], Field(description="Gemini resolution hint")] = "high",
        thinking_level: Annotated[Optional[str], Field(description="Gemini Pro thinking level")] = "high",
        enable_grounding: Annotated[bool, Field(description="Enable Gemini grounding")] = True,
        aspect_ratio: Annotated[Optional[Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]], Field(description="Output aspect ratio")] = None,
        output_dir: Annotated[Optional[str], Field(description="Optional custom output directory on the server")] = None,
    ) -> ToolResult:
        job_service = services.get_image_job_service()
        payload = {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "n": n,
            "negative_prompt": negative_prompt,
            "system_instruction": system_instruction,
            "model_tier": model_tier,
            "resolution": resolution,
            "thinking_level": thinking_level,
            "enable_grounding": enable_grounding,
            "aspect_ratio": aspect_ratio,
            "output_dir": output_dir,
            "mode": "generate",
        }
        job_id = job_service.submit_job(payload)
        summary = f"Queued image job `{job_id}`. Use `get_image_job_status` or `get_image_job_result`."
        return ToolResult(
            content=[TextContent(type="text", text=summary)],
            structured_content={"job_id": job_id, "status": "queued", "payload": payload},
        )

    @server.tool(
        annotations={"title": "Get async image job status", "readOnlyHint": True}
    )
    def get_image_job_status(
        job_id: Annotated[str, Field(description="Async image job id")],
    ) -> ToolResult:
        job_service = services.get_image_job_service()
        job = job_service.get_job(job_id)
        if not job:
            raise ValidationError(f"Image job '{job_id}' not found")
        summary = f"Job `{job_id}` status: {job['status']}"
        if job.get("error"):
            summary += f"\nError: {job['error']}"
        return ToolResult(content=[TextContent(type="text", text=summary)], structured_content=job)

    @server.tool(
        annotations={"title": "Get completed async image job result", "readOnlyHint": True}
    )
    def get_image_job_result(
        job_id: Annotated[str, Field(description="Async image job id")],
    ) -> ToolResult:
        job_service = services.get_image_job_service()
        job = job_service.get_job(job_id)
        if not job:
            raise ValidationError(f"Image job '{job_id}' not found")
        if job["status"] != "completed":
            summary = f"Job `{job_id}` is not completed yet. Current status: {job['status']}"
            if job.get("error"):
                summary += f"\nError: {job['error']}"
            return ToolResult(content=[TextContent(type="text", text=summary)], structured_content=job)

        result = job.get("result") or {}
        images = result.get("images") or []
        artifact_urls = [item.get("artifact_url") for item in images if item.get("artifact_url")]
        summary_lines = [
            f"Completed image job `{job_id}`.",
            f"Returned: {result.get('returned', len(images))}",
        ]
        for idx, url in enumerate(artifact_urls, start=1):
            summary_lines.append(f"{idx}. {url}")
        return ToolResult(
            content=[TextContent(type="text", text="\n".join(summary_lines))],
            structured_content=job,
        )

    @server.tool(
        annotations={"title": "List recent async image jobs", "readOnlyHint": True}
    )
    def list_image_jobs(
        status: Annotated[Optional[Literal["queued", "running", "completed", "failed"]], Field(description="Optional status filter")] = None,
        limit: Annotated[int, Field(description="Maximum jobs to return", ge=1, le=100)] = 20,
    ) -> ToolResult:
        job_service = services.get_image_job_service()
        jobs = job_service.list_jobs(status=status, limit=limit)
        summary = f"Recent image jobs: {len(jobs)}"
        return ToolResult(
            content=[TextContent(type="text", text=summary)],
            structured_content={"jobs": jobs, "count": len(jobs), "status": status, "limit": limit},
        )
