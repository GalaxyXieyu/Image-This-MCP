# Image This MCP 🎨

A production-ready **Model Context Protocol (MCP)** server that provides AI-powered image generation capabilities through **multiple providers** including Google's **Gemini** models and Volcengine's **Jimeng AI** with intelligent provider selection.

## ⭐ NEW: Multi-Provider Support! 🚀

Now supporting multiple image generation providers:

### 🏆 **Gemini (Nano Banana)**
- **Flash Model**: Gemini 3.1 Flash Image Preview by default for fast generation (1024px)
- **Pro Model**: 4K quality up to 3840px with Google Search grounding
- **Smart Selection**: Automatically chooses optimal model based on prompt
- **Advanced Features**: Text rendering, reference images, aspect ratio control

### 🎨 **Jimeng AI (Volcengine)**
- **Chinese-Optimized**: Tailored for Chinese language and cultural contexts
- **High Quality**: Default 3:4 portrait ratio (1536x2048)
- **Reference Images**: Support for image-based generation
- **Serial Queue**: Automatic rate limiting protection

## ✨ Features

- 🎨 **Multi-Provider Support**: Choose between Gemini and Jimeng AI, or auto-select
- ⚡ **Gemini 3.1 Flash Image Preview**: Default fast model (1024px) for rapid prototyping
- 🏆 **Gemini 3 Pro Image**: High-quality up to 4K with Google Search grounding
- 🤖 **Smart Model Selection**: Automatically chooses optimal model based on your prompt
- 🌏 **Jimeng AI Integration**: Chinese-optimized image generation with Volcengine
- 📐 **Aspect Ratio Control**: Specify output dimensions (1:1, 16:9, 9:16, 21:9, and more)
- 📋 **Smart Templates**: Pre-built prompt templates for photography, design, and editing
- 📁 **File Management**: Upload and manage files via Gemini Files API
- 🔍 **Resource Discovery**: Browse templates and file metadata through MCP resources
- 🛡️ **Production Ready**: Comprehensive error handling, logging, and validation
- ⚡ **High Performance**: Optimized architecture with intelligent caching

## 🚀 Quick Start

### Prerequisites

1. **Google Gemini API Key** - [Get one free here](https://makersuite.google.com/app/apikey)
2. **Python 3.11+** (for development only)

### Installation

**Option 1: From GitHub (Recommended)**

Install directly from GitHub using `uv` (recommended):

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install MCP server from GitHub
uv tool install git+https://github.com/GalaxyXieyu/Image-This-MCP.git

# Verify installation
command -v image-this-mcp

# Manage tools
uv tool list
uv tool uninstall image-this-mcp
```

Run without installing (uvx):

```bash
uvx --from git+https://github.com/GalaxyXieyu/Image-This-MCP.git image-this-mcp
```

**Option 2: Local Editable Install (Development)**

```bash
git clone https://github.com/GalaxyXieyu/Image-This-MCP.git
cd Image-This-MCP
uv pip install -e .
```

**Option 3: Using pip**

```bash
pip install git+https://github.com/GalaxyXieyu/Image-This-MCP.git
```

## 🔧 Configuration

### Authentication Methods

Nano Banana supports two authentication methods via `NANOBANANA_AUTH_METHOD`:

1. **API Key** (`api_key`): Uses `GEMINI_API_KEY`. Best for local development and simple deployments.
2. **Vertex AI ADC** (`vertex_ai`): Uses Google Cloud Application Default Credentials. Best for production on Google Cloud (Cloud Run, GKE, GCE).
3. **Automatic** (`auto`): Defaults to API Key if present, otherwise tries Vertex AI.

Note: `NANOBANANA_*` environment variables are historical compatibility names. The current package and CLI entrypoint are `image-this-mcp`.

#### 1. API Key Authentication (Default)
Set `GEMINI_API_KEY` environment variable.

### OpenClaw Plugin (Jimeng 4.5, no MCP server)

If you want to use OpenClaw directly (bypassing the MCP server), install the plugin in this repo and configure OpenClaw:

```bash
openclaw plugins install -l ./openclaw-plugin
openclaw gateway restart
```

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "enabled": true,
    "entries": {
      "img-generator": {
        "enabled": true,
        "config": {
          "apiKey": "<YOUR_ARK_API_KEY>",
          "baseUrl": "https://ark.cn-beijing.volces.com/api/v3/images/generations",
          "model": "doubao-seedream-4.5",
          "size": "1728x2304",
          "watermark": false,
          "timeoutMs": 120000,
          "superbedToken": "<YOUR_SUPERBED_TOKEN>"
        }
      }
    }
  },
  "tools": {
    "allow": ["img-generator"]
  }
}
```

Notes:
- If `superbedToken` is set, the tool uploads the image to Superbed and returns a `MEDIA: <url>` line plus a Markdown image link. This makes the image show up in OpenClaw channels that support media.
- If `superbedToken` is not set, the tool only returns base64 image data in tool output, which may not render as an image in chat.
- Reference images (`referenceImages`) accept:
  - HTTP/HTTPS URLs
  - `data:image/*;base64,...` data URLs (will be sanitized)
  - Raw base64 strings (will be wrapped as data URLs)
  - Local file paths (e.g. `~/Pictures/ref.jpg` or `file:///...`) which are read and encoded
- Size: Jimeng 4.5 rejects small sizes (e.g. `1024x1024`). The plugin auto-falls back to `1728x2304` if total pixels are below 3,686,400, and adds `sizeRequested/sizeAdjusted/sizeNote` to metadata.

#### 2. Third-Party Banana API Support
You can use third-party Banana API services that are compatible with Gemini API by setting a custom API base URL:

```bash
# Set your third-party API key
export GEMINI_API_KEY="your-third-party-api-key"

# Set the custom API base URL
export GEMINI_API_BASE_URL="https://your-banana-api-endpoint.com/v1"
# or
export BANANA_API_BASE_URL="https://your-banana-api-endpoint.com/v1"
```

**Example Configuration** (Claude Desktop):
```json
{
  "mcpServers": {
    "image-this": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-third-party-api-key",
        "GEMINI_API_BASE_URL": "https://your-banana-api-endpoint.com/v1"
      }
    }
  }
}
```

#### 3. Vertex AI Authentication (Google Cloud)
Required environment variables:
- `NANOBANANA_AUTH_METHOD=vertex_ai` (or `auto`)
- `GCP_PROJECT_ID=your-project-id`
- `GCP_REGION=us-central1` (default)

**Prerequisites**:
- Enable Vertex AI API: `gcloud services enable aiplatform.googleapis.com`
- Grant IAM Role: `roles/aiplatform.user` to the service account.

### Provider Selection

Choose your default image generation provider via `IMAGE_PROVIDER` environment variable:

```bash
# Use Gemini (default)
export IMAGE_PROVIDER=gemini

# Use Jimeng AI
export IMAGE_PROVIDER=jimeng

# Use OpenAI-compatible images API
export IMAGE_PROVIDER=openai
```

You can also specify the provider per-request using the `provider` parameter in the `generate_image` tool:
- `"gemini"` - Use Gemini (Nano Banana)
- `"jimeng"` - Use the Jimeng model family (legacy Jimeng + Seedream/Jimeng 4.5 style models)
- `"auto"` - Use default provider from environment

To choose a specific model inside a provider family, use the optional `model` parameter with a model id returned by `list_models`.
For example, `provider="jimeng"` with `model="doubao-seedream-4.5"` will route to the correct Jimeng-family backend automatically.

### Jimeng AI Configuration

To use Jimeng AI provider, you need Volcengine credentials:

1. Get your credentials at [Volcengine Console](https://console.volcengine.com/)
2. Set the following environment variables:

```bash
export JIMENG_ACCESS_KEY=your_access_key_here
export JIMENG_SECRET_KEY=your_secret_key_here
```

**Example Configuration** (Claude Desktop with Jimeng):
```json
{
  "mcpServers": {
    "image-this": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "IMAGE_PROVIDER": "jimeng",
        "JIMENG_ACCESS_KEY": "your-access-key",
        "JIMENG_SECRET_KEY": "your-secret-key"
      }
    }
  }
}
```

**Jimeng AI Features**:
- Default resolution: 1536x2048 (3:4 portrait)
- Supports reference images for image-to-image generation
- Serial request queue to avoid rate limiting
- Automatic retry with exponential backoff

### OpenAI-Compatible Image Configuration

To use OpenAI-compatible image providers such as OpenAI official API or ToAPIs:

```bash
export IMAGE_PROVIDER=openai
export OPENAI_API_KEY="your-openai-compatible-key"
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_MODEL="gpt-image-2"
```

Example:

```json
{
  "mcpServers": {
    "image-this": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "IMAGE_PROVIDER": "openai",
        "OPENAI_API_KEY": "your-key",
        "OPENAI_BASE_URL": "https://your-endpoint.example.com/v1",
        "OPENAI_MODEL": "gpt-image-2"
      }
    }
  }
}
```

### Remote HTTP Deployment

If you want many computers to share one MCP server, you can deploy this project once on a remote machine and connect clients to that HTTP MCP endpoint.

Recommended phase-1 shape:
- HTTP transport
- Shared Bearer token
- Synchronous image generation
- MinIO/S3-compatible artifact publishing for final images

Server environment example:

```bash
export FASTMCP_TRANSPORT=http
export FASTMCP_HOST=0.0.0.0
export FASTMCP_PORT=34128

export MCP_AUTH_TOKEN="replace-with-a-random-token"
export MCP_AUTH_HEADER=Authorization

export IMAGE_PROVIDER=openai
export OPENAI_API_KEY="your-openai-compatible-key"
export OPENAI_BASE_URL="https://your-endpoint.example.com/v1"
export OPENAI_MODEL="gpt-image-2"

# Optional: Gemini provider
export GEMINI_API_KEY="your-gemini-key"
export GEMINI_API_BASE_URL="https://your-gemini-compatible-endpoint/v1"

# Optional: Jimeng legacy provider
export JIMENG_ACCESS_KEY="your-volcengine-access-key"
export JIMENG_SECRET_KEY="your-volcengine-secret-key"

# Optional: Jimeng Seedream / Ark provider
export ARK_API_KEY="your-ark-api-key"
export JIMENG45_API_KEY="your-ark-api-key"

export MINIO_ENDPOINT="127.0.0.1:9000"
export MINIO_ACCESS_KEY="your-minio-access-key"
export MINIO_SECRET_KEY="your-minio-secret-key"
export MINIO_BUCKET="image-this"
export MINIO_SECURE=false
export MINIO_PUBLIC_BASE_URL="http://your-server:9000"
```

Start the server:

```bash
uvx --from git+https://github.com/GalaxyXieyu/Image-This-MCP.git image-this-mcp
```

Remote MCP client example:

```json
{
  "mcpServers": {
    "image-this-remote": {
      "url": "http://your-server:34128/mcp",
      "headers": {
        "Authorization": "Bearer replace-with-the-same-token"
      }
    }
  }
}
```

For a concrete Docker-based deployment example, see [docs/REMOTE_DEPLOYMENT.md](docs/REMOTE_DEPLOYMENT.md).

### Async Remote Jobs

For remote deployments with multiple clients, you can use the async job tools instead of waiting on a single long request:

- `submit_image_job`
- `get_image_job_status`
- `get_image_job_result`
- `list_image_jobs`

Recommended flow:

1. Submit a job with `submit_image_job`
2. Poll with `get_image_job_status`
3. Fetch final URLs and metadata with `get_image_job_result`

This is especially useful when several machines share one remote MCP server.

### Current Provider Scope

Image generation providers currently supported by this repo:
- Gemini
- Jimeng model family
- OpenAI-compatible image APIs

`Moonshot` and `DeepLX` are not image generation providers in this server today, so they are not configurable here yet.

### Claude Desktop

#### Option 1: Using GitHub Directly (Recommended)

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "image-this": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

#### Option 2: Using GitHub Installation

If you installed from GitHub using `uv tool install`, use the installed command directly:

```json
{
  "mcpServers": {
    "image-this": {
      "command": "image-this-mcp",
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

#### Option 3: Using Local Source (Development)

If you are running from source code, point to your local installation:

```json
{
  "mcpServers": {
    "image-this-local": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "image_this_mcp.server"
      ],
      "cwd": "/absolute/path/to/Image-This-MCP",
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```


#### Option 4: Using Vertex AI (ADC)

To authenticate with Google Cloud Application Default Credentials (instead of an API Key):

```json
{
  "mcpServers": {
    "image-this-adc": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "NANOBANANA_AUTH_METHOD": "vertex_ai",
        "GCP_PROJECT_ID": "your-project-id",
        "GCP_REGION": "us-central1"
      }
    }
  }
}
```

**Configuration file locations:**

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Claude Code (VS Code Extension)

Install and configure in VS Code:

1. Install the Claude Code extension
2. Open Command Palette (`Cmd/Ctrl + Shift + P`)
3. Run "Claude Code: Add MCP Server"
4. Configure:
   ```json
   {
     "name": "image-this",
     "command": "uvx",
     "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
     "env": {
       "GEMINI_API_KEY": "your-gemini-api-key-here"
     }
   }
   ```

### Cursor

Add to Cursor's MCP configuration:

```json
{
  "mcpServers": {
    "image-this": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

### Continue.dev (VS Code/JetBrains)

Add to your `config.json`:

```json
{
  "mcpServers": [
    {
      "name": "image-this",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  ]
}
```

### Open WebUI

Configure in Open WebUI settings:

```json
{
  "mcp_servers": {
    "image-this": {
      "command": ["uvx", "--from", "git+https://github.com/GalaxyXieyu/Image-This-MCP.git", "image-this-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

### Gemini CLI / Generic MCP Client

```bash
# Set environment variable
export GEMINI_API_KEY="your-gemini-api-key-here"

# Run server in stdio mode
uvx --from git+https://github.com/GalaxyXieyu/Image-This-MCP.git image-this-mcp

# Or with pip installation
python -m image_this_mcp.server
```

## 🤖 Model Selection

Nano Banana supports two Gemini models with intelligent automatic selection:

### 🏆 Pro Model - Nano Banana Pro (Gemini 3 Pro Image) ⭐ NEW!
**Google's latest and most advanced image generation model**

- **Quality**: Professional-grade, production-ready
- **Resolution**: Up to 4K (3840px) - highest available
- **Speed**: ~5-8 seconds per image
- **Special Features**:
  - 🌐 **Google Search Grounding**: Leverages real-world knowledge for accurate, contextual images
  - 🧠 **Advanced Reasoning**: Configurable thinking levels (LOW/HIGH) for complex compositions
  - 📐 **Media Resolution Control**: Fine-tune vision processing detail (LOW/MEDIUM/HIGH/AUTO)
  - 📝 **Superior Text Rendering**: Exceptional clarity for text-in-image generation
  - 🎨 **Enhanced Context Understanding**: Better interpretation of complex, narrative prompts
- **Best for**: Production assets, marketing materials, professional photography, high-fidelity outputs, images requiring text, factual accuracy
- **Cost**: Higher per image (premium quality)

### ⚡ Flash Model (Gemini 3.1 Flash Image Preview)
**Fast, reliable model for rapid iteration**

- **Speed**: Very fast (2-3 seconds)
- **Resolution**: Up to 1024px
- **Quality**: High quality for everyday use
- **Best for**: Rapid prototyping, iterations, high-volume generation, drafts, sketches
- **Cost**: Lower per image

### 🤖 Automatic Selection (Recommended)

By default, direct calls use the **Flash** tier. You can still choose `auto` to let the server analyze your prompt and requirements:

**Pro Model Selected When**:
- Quality keywords detected: "4K", "professional", "production", "high-res", "HD"
- High resolution requested: `resolution="4k"` or `resolution="high"`
- Google Search grounding enabled: `enable_grounding=True`
- High thinking level requested: `thinking_level="HIGH"`
- Multi-image conditioning with multiple input images

**Flash Model Selected When**:
- Speed keywords detected: "quick", "draft", "sketch", "rapid"
- High-volume batch generation: `n > 2`
- Standard or lower resolution requested
- No special Pro features required

### Usage Examples

```python
# Automatic selection (recommended)
"Generate a professional 4K product photo"  # → Pro model (quality keywords + 4K)
"Quick sketch of a cat"                     # → Flash model (speed keyword)
"Create a diagram with clear text labels"   # → Pro model (text rendering)
"Draft mockup for website hero section"     # → Flash model (draft keyword)

# Explicit model selection
generate_image(
    prompt="A scenic landscape",
    model_tier="flash"  # Force Flash model for speed
)

# Leverage Nano Banana Pro features
generate_image(
    prompt="Professional product photo of vintage camera on wooden desk",
    model_tier="pro",              # Use Pro model
    resolution="4k",               # 4K resolution (Pro-only)
    thinking_level="HIGH",         # Enhanced reasoning
    enable_grounding=True,         # Use Google Search for accuracy
    media_resolution="HIGH"        # High-detail vision processing
)

# Pro model for high-quality text rendering
generate_image(
    prompt="Infographic showing 2024 market statistics with clear labels",
    model_tier="pro",              # Pro excels at text rendering
    resolution="4k"                # Maximum clarity for text
)

# Control aspect ratio for different formats ⭐ NEW!
generate_image(
    prompt="Cinematic landscape at sunset",
    aspect_ratio="21:9"            # Ultra-wide cinematic format
)

generate_image(
    prompt="Instagram post about coffee",
    aspect_ratio="1:1"             # Square format for social media
)

generate_image(
    prompt="YouTube thumbnail design",
    aspect_ratio="16:9"            # Standard video format
)

generate_image(
    prompt="Mobile wallpaper of mountain vista",
    aspect_ratio="9:16"            # Portrait format for phones
)
```

### 📐 Aspect Ratio Control ⭐ NEW!

Control the output image dimensions with the `aspect_ratio` parameter:

**Supported Aspect Ratios**:
- `1:1` - Square (Instagram, profile pictures)
- `4:3` - Classic photo format
- `3:4` - Portrait orientation
- `16:9` - Widescreen (YouTube thumbnails, presentations)
- `9:16` - Mobile portrait (phone wallpapers, stories)
- `21:9` - Ultra-wide cinematic
- `2:3`, `3:2`, `4:5`, `5:4` - Various photo formats

```python
# Examples for different use cases
generate_image(
    prompt="Product showcase for e-commerce",
    aspect_ratio="3:4",    # Portrait format, good for product pages
    model_tier="pro"
)

generate_image(
    prompt="Social media banner for Facebook",
    aspect_ratio="16:9"    # Landscape banner format
)
```

**Note**: Aspect ratio works with both Flash and Pro models. For best results with specific aspect ratios at high resolution, use the Pro model with `resolution="4k"`.

## ⚙️ Environment Variables

Configuration options:

```bash
# Authentication (Required)
# Method 1: API Key (Google Gemini API or Third-party Banana API)
GEMINI_API_KEY=your-gemini-api-key-here

# Third-party Banana API Configuration (Optional)
# If using a third-party Banana API service, set the custom base URL:
GEMINI_API_BASE_URL=https://your-banana-api-endpoint.com/v1
# or
BANANA_API_BASE_URL=https://your-banana-api-endpoint.com/v1

# Method 2: Vertex AI (Google Cloud)
NANOBANANA_AUTH_METHOD=vertex_ai
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# Model Selection (optional)
NANOBANANA_MODEL=pro  # Options: flash, pro, auto (default: pro)

# Optional
IMAGE_OUTPUT_DIR=/path/to/image/directory  # Default: ~/image-this
LOG_LEVEL=INFO                             # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=standard                        # standard, json, detailed
```

## 🐛 Troubleshooting

### Common Issues

**"GEMINI_API_KEY not set"**

- Add your API key to the MCP server configuration in your client
- Get a free API key at [Google AI Studio](https://makersuite.google.com/app/apikey)

**"Server failed to start"**

- Ensure you're using the latest GitHub version: `uvx --from git+https://github.com/GalaxyXieyu/Image-This-MCP.git image-this-mcp`
- Check that your client supports MCP (Claude Desktop 0.10.0+)

**"Permission denied" errors**

- The server creates images in `~/image-this` by default
- Ensure write permissions to your home directory

### Development Setup

For local development:

```bash
# Clone repository
git clone https://github.com/GalaxyXieyu/Image-This-MCP.git
cd Image-This-MCP

# Install with uv
uv sync

# Set environment
export GEMINI_API_KEY=your-api-key-here

# Run locally
uv run python -m image_this_mcp.server
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/GalaxyXieyu/Image-This-MCP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GalaxyXieyu/Image-This-MCP/discussions)
