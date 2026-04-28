# Remote Deployment on Taurus

This server can run in remote HTTP mode so multiple computers share one MCP endpoint.

## Phase 1 design

- HTTP transport on a dedicated port
- Shared token authentication via request header
- Synchronous `generate_image` responses
- Provider concurrency limits in-process
- Final images uploaded to MinIO and returned as `artifact_url`

## Taurus quick start

1. Copy deployment files to the server:

```bash
scp -r deploy/taurus Taurus:/root/image-this-mcp-deploy
scp Dockerfile Taurus:/root/image-this-mcp-deploy/
```

2. Clone the repo on Taurus or copy the working tree:

```bash
ssh Taurus
git clone https://github.com/GalaxyXieyu/Image-This-MCP.git /root/Image-This-MCP
cd /root/Image-This-MCP/deploy/taurus
cp .env.example .env
```

3. Fill `.env`:

- `MCP_AUTH_TOKEN`: shared secret for all clients
- `OPENAI_*`: your OpenAI-compatible provider config
- `MINIO_*`: Taurus MinIO connection and output bucket

4. Start the service:

```bash
cd /root/Image-This-MCP/deploy/taurus
docker compose up -d --build
```

5. Health check:

```bash
curl http://38.76.197.25:34128/health
```

6. MCP client example:

```json
{
  "mcpServers": {
    "image-this-remote": {
      "url": "http://38.76.197.25:34128/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_AUTH_TOKEN>"
      }
    }
  }
}
```

## Notes

- `Authorization: Bearer <token>` is the default expected header format.
- If `MINIO_PUBLIC_BASE_URL` is omitted, the server falls back to a presigned URL.
- Gemini and OpenAI can run together; only the configured provider credentials are used per request.
