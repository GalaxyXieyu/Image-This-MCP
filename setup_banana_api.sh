#!/bin/bash
# 快速配置第三方 Banana API

echo "=== 配置第三方 Banana API ==="
echo ""

# 设置环境变量
export GEMINI_API_KEY="sk-iwoXbxYKCPB0OK6aw2nY8QV1TYOzpj3bqqzAaKbZLCHZjvgR"
export GEMINI_API_BASE_URL="https://yunwu.ai"

echo "✓ 环境变量已设置："
echo "  GEMINI_API_KEY=${GEMINI_API_KEY:0:20}..."
echo "  GEMINI_API_BASE_URL=${GEMINI_API_BASE_URL}"
echo ""

echo "现在可以运行 MCP 服务器了："
echo "  uvx nanobanana-mcp-server@latest"
echo ""
echo "或者在 Claude Desktop 配置文件中添加："
echo '{'
echo '  "mcpServers": {'
echo '    "nanobanana": {'
echo '      "command": "uvx",'
echo '      "args": ["nanobanana-mcp-server@latest"],'
echo '      "env": {'
echo "        \"GEMINI_API_KEY\": \"${GEMINI_API_KEY}\","
echo "        \"GEMINI_API_BASE_URL\": \"${GEMINI_API_BASE_URL}\""
echo '      }'
echo '    }'
echo '  }'
echo '}'

