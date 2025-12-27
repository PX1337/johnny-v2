#!/usr/bin/env python3
"""
Johnny v2 - Clean MCP Server with bge-m3 embeddings

Railway-ready MCP server:
- SentenceTransformer BAAI/bge-m3 (1024 dims, 8192 token context)
- Qdrant vector database
- FastAPI with /mcp endpoint
- Bearer token authentication
- Clean, minimal codebase

Environment variables:
    QDRANT_URL: Qdrant connection URL
    QDRANT_API_KEY: Qdrant API key
    BEARER_TOKEN: Authentication token for MCP calls
"""

import os
import json
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "dev-token-change-me")
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024

# Global state
embedding_model: Optional[SentenceTransformer] = None
qdrant_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global embedding_model, qdrant_client

    # Startup
    print(f"🚀 Loading embedding model: {EMBEDDING_MODEL}...")
    embedding_model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL)
    print(f"✅ Model loaded (1024 dims, 8192 token context)")

    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    qdrant_client = httpx.AsyncClient(
        base_url=QDRANT_URL,
        headers=headers,
        timeout=30.0
    )

    response = await qdrant_client.get("/")
    if response.status_code == 200:
        print(f"✅ Qdrant connected: {QDRANT_URL}")
    else:
        raise Exception(f"Qdrant connection failed: {response.status_code}")

    yield

    # Shutdown
    if qdrant_client:
        await qdrant_client.aclose()
    print("👋 Johnny v2 shutdown complete")


app = FastAPI(
    title="Johnny v2 MCP Server",
    version="2.0.0",
    lifespan=lifespan
)


# Auth middleware
def verify_token(authorization: Optional[str] = Header(None)):
    """Verify Bearer token"""
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Authorization format")

    token = authorization[7:]
    if token != BEARER_TOKEN:
        raise HTTPException(401, "Invalid token")


# Helper functions
async def generate_embedding(text: str) -> List[float]:
    """Generate embedding using bge-m3"""
    if not embedding_model:
        raise RuntimeError("Model not loaded")

    embedding = await asyncio.to_thread(embedding_model.encode, text)
    return embedding.tolist()


async def ensure_collection(collection: str) -> bool:
    """Create collection if it doesn't exist"""
    if not qdrant_client:
        return False

    # Check if exists
    response = await qdrant_client.get(f"/collections/{collection}")
    if response.status_code == 200:
        return True

    # Create collection
    response = await qdrant_client.put(
        f"/collections/{collection}",
        json={
            "vectors": {
                "size": VECTOR_SIZE,
                "distance": "Cosine"
            }
        }
    )

    return response.status_code == 200


# MCP Tools
MCP_TOOLS = [
    {
        "name": "johnny_search",
        "description": "Search for entities by semantic similarity across knowledge collections. Returns ranked results with content, metadata, and relevance scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "collections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Collections to search (e.g., ['shared-knowledge', 'language-style'])",
                    "default": ["shared-knowledge"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 50
                },
                "score_threshold": {
                    "type": "number",
                    "description": "Minimum relevance score (0.0-1.0, default: 0.7)",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "johnny_upsert",
        "description": "Create or update an entity in Johnny. If entity exists, it will be updated; otherwise created. Content is automatically embedded for semantic search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name"
                },
                "entity_name": {
                    "type": "string",
                    "description": "Unique entity name/identifier"
                },
                "content": {
                    "type": "string",
                    "description": "Entity content (will be embedded for search)"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata (type, category, date, etc.)",
                    "additionalProperties": True
                }
            },
            "required": ["collection", "entity_name", "content"]
        }
    },
    {
        "name": "johnny_delete",
        "description": "Delete an entity from Johnny by name. This is permanent and cannot be undone.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name"
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity name to delete"
                }
            },
            "required": ["collection", "entity_name"]
        }
    }
]


# Tool handlers
async def handle_search(args: Dict[str, Any]) -> str:
    """Search entities by semantic similarity"""
    query = args.get("query")
    collections = args.get("collections", ["shared-knowledge"])
    limit = args.get("limit", 50)
    score_threshold = args.get("score_threshold", 0.7)

    if not query:
        return "Error: query is required"

    # Generate embedding
    query_vector = await generate_embedding(query)

    # Search across collections
    all_results = []

    for collection in collections:
        await ensure_collection(collection)

        response = await qdrant_client.post(
            f"/collections/{collection}/points/search",
            json={
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "score_threshold": score_threshold
            }
        )

        if response.status_code == 200:
            data = response.json()
            for hit in data.get("result", []):
                all_results.append({
                    "collection": collection,
                    "entity_name": hit["payload"].get("entity_name"),
                    "content": hit["payload"].get("content"),
                    "metadata": hit["payload"].get("metadata", {}),
                    "score": hit["score"]
                })

    # Sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    all_results = all_results[:limit]

    if not all_results:
        return f"No results found for query: '{query}'\nSearched collections: {', '.join(collections)}\nScore threshold: {score_threshold}"

    # Format output
    output = f"Found {len(all_results)} results:\n\n"
    for i, result in enumerate(all_results, 1):
        output += f"{i}. {result['entity_name']} (score: {result['score']:.3f})\n"
        output += f"   Collection: {result['collection']}\n"
        output += f"   {result['content'][:200]}...\n\n"

    return output


async def handle_upsert(args: Dict[str, Any]) -> str:
    """Create or update entity"""
    collection = args.get("collection")
    entity_name = args.get("entity_name")
    content = args.get("content")
    metadata = args.get("metadata", {})

    if not all([collection, entity_name, content]):
        return "Error: collection, entity_name, and content are required"

    # Ensure collection exists
    await ensure_collection(collection)

    # Generate embedding
    vector = await generate_embedding(content)

    # Upsert to Qdrant
    point_id = abs(hash(entity_name)) % (2**63)

    response = await qdrant_client.put(
        f"/collections/{collection}/points",
        json={
            "points": [{
                "id": point_id,
                "vector": vector,
                "payload": {
                    "entity_name": entity_name,
                    "content": content,
                    "metadata": metadata
                }
            }]
        }
    )

    if response.status_code == 200:
        return f"✅ Successfully upserted '{entity_name}' to '{collection}'"
    else:
        return f"Error: Failed to upsert (status: {response.status_code})"


async def handle_delete(args: Dict[str, Any]) -> str:
    """Delete entity"""
    collection = args.get("collection")
    entity_name = args.get("entity_name")

    if not all([collection, entity_name]):
        return "Error: collection and entity_name are required"

    # Find point by entity_name
    point_id = abs(hash(entity_name)) % (2**63)

    response = await qdrant_client.post(
        f"/collections/{collection}/points/delete",
        json={"points": [point_id]}
    )

    if response.status_code == 200:
        return f"✅ Deleted '{entity_name}' from '{collection}'"
    else:
        return f"Error: Failed to delete (status: {response.status_code})"


# MCP Endpoints
@app.post("/mcp")
async def mcp_endpoint(request: Request, authorization: str = Header(None)):
    """MCP protocol endpoint"""
    verify_token(authorization)

    body = await request.json()
    method = body.get("method")

    # tools/list
    if method == "tools/list":
        return {
            "tools": MCP_TOOLS
        }

    # tools/call
    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "johnny_search":
            result = await handle_search(args)
        elif tool_name == "johnny_upsert":
            result = await handle_upsert(args)
        elif tool_name == "johnny_delete":
            result = await handle_delete(args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

        return {
            "content": [{"type": "text", "text": result}]
        }

    return {"error": f"Unknown method: {method}"}


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "model": EMBEDDING_MODEL,
        "vector_size": VECTOR_SIZE,
        "qdrant": QDRANT_URL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
