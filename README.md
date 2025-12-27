# Johnny v2 - Clean MCP Server

Modern MCP server with BAAI/bge-m3 embeddings (1024 dims, 8192 token context).

## Features

- ✅ **Better model**: bge-m3 with 8192 token context (vs 512 in FastEmbed)
- ✅ **Clean code**: Single file, ~350 lines
- ✅ **Railway-ready**: Dockerfile + environment variables
- ✅ **MCP protocol**: Works with Claude Code
- ✅ **Bearer auth**: Secure token authentication

## Environment Variables

```bash
QDRANT_URL=https://your-qdrant.railway.app
QDRANT_API_KEY=your-qdrant-api-key
BEARER_TOKEN=your-secure-token
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export QDRANT_URL=http://localhost:6333
export BEARER_TOKEN=dev-token

# Run server
python3 server.py
```

## Railway Deployment

1. Create new Railway project
2. Add Qdrant plugin
3. Deploy this repo
4. Set environment variables:
   - `QDRANT_URL` (from Qdrant plugin)
   - `QDRANT_API_KEY` (from Qdrant plugin)
   - `BEARER_TOKEN` (generate secure token)
5. Get public URL

## Claude Code Configuration

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "johnny": {
      "type": "http",
      "url": "https://your-johnny.railway.app/mcp",
      "headers": {
        "Authorization": "Bearer your-token-here"
      }
    }
  }
}
```

## MCP Tools

- `johnny_search` - Semantic search across collections
- `johnny_upsert` - Create/update entities
- `johnny_delete` - Delete entities
- `johnny_list_collections` - List all collections with point counts and metadata

## Model Details

- Model: BAAI/bge-m3
- Dimensions: 1024
- Context window: 8192 tokens (16x more than FastEmbed!)
- Size: 2.3GB (cached in Docker image)
# Build trigger 1766869182
