# Image-This-MCP 安装指南

## 本地安装（开发模式）

如果你已经克隆了仓库，可以使用以下方式安装：

```bash
# 使用 uv（推荐）
uv pip install -e .

# 或使用 pip
pip install -e .
```

## 远程安装（其他电脑）

### 方法 1：从 GitHub 直接安装（推荐）

```bash
# 使用 uv
uv pip install git+https://github.com/GalaxyXieyu/Image-This-MCP.git

# 或使用 pip
pip install git+https://github.com/GalaxyXieyu/Image-This-MCP.git
```

### 方法 2：克隆后安装

```bash
# 1. 克隆仓库
git clone https://github.com/GalaxyXieyu/Image-This-MCP.git
cd Image-This-MCP

# 2. 安装依赖和包
uv pip install -e .
# 或
pip install -e .
```

### 方法 3：使用 uvx（无需安装，直接运行）

```bash
# 直接运行，无需安装
uvx git+https://github.com/GalaxyXieyu/Image-This-MCP.git
```

## 配置第三方 Banana API

安装完成后，需要配置 API 密钥和端点：

### 环境变量配置

```bash
# 设置 API Key
export GEMINI_API_KEY="your-api-key"

# 设置第三方 API Base URL（如果使用第三方服务）
export GEMINI_API_BASE_URL="https://yunwu.ai"
```

### Claude Desktop 配置示例

在 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "image-this": {
      "command": "uvx",
      "args": ["git+https://github.com/GalaxyXieyu/Image-This-MCP.git"],
      "env": {
        "GEMINI_API_KEY": "your-api-key",
        "GEMINI_API_BASE_URL": "https://yunwu.ai"
      }
    }
  }
}
```

### 使用本地安装的版本

如果你已经在其他电脑上安装了，可以这样配置：

```json
{
  "mcpServers": {
    "image-this": {
      "command": "nanobanana-mcp-server",
      "env": {
        "GEMINI_API_KEY": "your-api-key",
        "GEMINI_API_BASE_URL": "https://yunwu.ai"
      }
    }
  }
}
```

## 验证安装

安装完成后，可以运行以下命令验证：

```bash
# 检查命令是否可用
nanobanana-mcp-server --help

# 或
nanobanana-mcp --help
```

## 快速开始

1. **安装**：
   ```bash
   uv pip install git+https://github.com/GalaxyXieyu/Image-This-MCP.git
   ```

2. **配置环境变量**：
   ```bash
   export GEMINI_API_KEY="your-api-key"
   export GEMINI_API_BASE_URL="https://yunwu.ai"  # 可选，第三方 API
   ```

3. **运行**：
   ```bash
   nanobanana-mcp-server
   ```

## 故障排除

### 问题：找不到命令

**解决方案**：
- 确保 Python 环境已激活
- 检查 PATH 环境变量是否包含 Python 的 bin 目录
- 尝试使用完整路径：`python -m nanobanana_mcp_server.server`

### 问题：模块未找到

**解决方案**：
```bash
# 重新安装
uv pip install --force-reinstall git+https://github.com/GalaxyXieyu/Image-This-MCP.git
```

### 问题：API 连接失败

**解决方案**：
- 检查 API Key 是否正确
- 检查 Base URL 格式（只需要域名，不要包含路径）
- 确认网络连接正常

## 更新到最新版本

```bash
# 如果使用 git+ 安装
uv pip install --upgrade git+https://github.com/GalaxyXieyu/Image-This-MCP.git

# 如果使用克隆方式
cd Image-This-MCP
git pull
uv pip install -e .
```

