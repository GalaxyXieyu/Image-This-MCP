# 快速开始指南

## 问题：缺少 API Key 错误

如果看到以下错误：
```
No valid authentication configuration found.
API Key: Set GEMINI_API_KEY or GOOGLE_API_KEY
```

这是因为没有设置环境变量。

## 解决方案

### 方法 1：在命令行中设置环境变量（测试用）

```bash
GEMINI_API_KEY="sk-iwoXbxYKCPB0OK6aw2nY8QV1TYOzpj3bqqzAaKbZLCHZjvgR" \
GEMINI_API_BASE_URL="https://yunwu.ai" \
uvx git+https://github.com/GalaxyXieyu/Image-This-MCP.git
```

### 方法 2：在 Claude Desktop 配置文件中设置（推荐）

编辑 Claude Desktop 配置文件：
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

添加以下配置：

```json
{
  "mcpServers": {
    "image-this": {
      "command": "uvx",
      "args": ["git+https://github.com/GalaxyXieyu/Image-This-MCP.git"],
      "env": {
        "GEMINI_API_KEY": "sk-iwoXbxYKCPB0OK6aw2nY8QV1TYOzpj3bqqzAaKbZLCHZjvgR",
        "GEMINI_API_BASE_URL": "https://yunwu.ai"
      }
    }
  }
}
```

### 方法 3：使用 .env 文件（如果支持）

在项目目录或用户目录创建 `.env` 文件：

```bash
GEMINI_API_KEY=sk-iwoXbxYKCPB0OK6aw2nY8QV1TYOzpj3bqqzAaKbZLCHZjvgR
GEMINI_API_BASE_URL=https://yunwu.ai
```

然后运行：
```bash
uvx git+https://github.com/GalaxyXieyu/Image-This-MCP.git
```

## 验证配置

配置完成后，重启 Claude Desktop，MCP 服务器应该能正常启动。

如果还有问题，检查：
1. API Key 是否正确
2. Base URL 格式是否正确（只需要域名，不要包含路径）
3. Claude Desktop 配置文件格式是否正确（JSON 格式）

