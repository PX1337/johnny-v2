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
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "dev-token-change-me")  # Fallback for old clients
USER_TOKENS_JSON = os.getenv("USER_TOKENS", "{}")
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024

# Parse user tokens
try:
    USER_TOKENS = json.loads(USER_TOKENS_JSON)
except json.JSONDecodeError:
    print("⚠️ Invalid USER_TOKENS JSON, using fallback auth")
    USER_TOKENS = {}

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
def verify_token(authorization: Optional[str] = Header(None)) -> Dict[str, str]:
    """Verify Bearer token and return user info"""
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Authorization format")

    token = authorization[7:]

    # Check USER_TOKENS first (new multi-user auth)
    if token in USER_TOKENS:
        return USER_TOKENS[token]

    # Fallback to old BEARER_TOKEN (backward compatibility)
    if token == BEARER_TOKEN:
        return {"user": "anonymous", "role": "write"}

    raise HTTPException(401, "Invalid token")


def check_collection_access(user_info: Dict[str, str], collection: str, operation: str) -> bool:
    """Check if user has access to collection for given operation (read/write)"""
    username = user_info.get("user", "")
    role = user_info.get("role", "read")

    # Private collections: only owner has access
    if collection.startswith("private-"):
        owner = collection[8:]  # Remove "private-" prefix
        return username == owner

    # Shared collections: everyone can read, only write role can write
    if operation == "write":
        return role == "write"

    return True  # Everyone can read shared collections


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
        "description": "Semantic search across knowledge collections. Returns ranked results with 600-char preview. Use johnny_get for full content. Params: query (required), collections (array), limit (default 50, no max), score_threshold (default 0.6).",
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
                    "minimum": 1
                },
                "score_threshold": {
                    "type": "number",
                    "description": "Minimum relevance score (0.0-1.0, default: 0.6)",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "johnny_upsert",
        "description": "Create or update entity. Content is embedded with bge-m3 (8192 tokens context). Params: collection, entity_name, content (all required), metadata (optional object).",
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
        "description": "Permanently delete entity by name. Cannot be undone. Params: collection, entity_name (both required).",
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
    },
    {
        "name": "johnny_list_collections",
        "description": "List all collections with point counts and access status. No params required.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "johnny_get",
        "description": "Get FULL content and metadata for entity. Use after search when you need complete text (search shows 600-char preview). Params: collection, entity_name (both required).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name"
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity name to retrieve (from search results)"
                }
            },
            "required": ["collection", "entity_name"]
        }
    }
]


# Tool handlers
async def handle_search(args: Dict[str, Any], user_info: Dict[str, str]) -> str:
    """Search entities by semantic similarity"""
    query = args.get("query")
    collections = args.get("collections", ["shared-knowledge"])
    limit = args.get("limit", 50)
    score_threshold = args.get("score_threshold", 0.6)

    if not query:
        return "Error: query is required"

    # Generate embedding
    query_vector = await generate_embedding(query)

    # Search across collections
    all_results = []
    denied_collections = []

    for collection in collections:
        # Check access
        if not check_collection_access(user_info, collection, "read"):
            denied_collections.append(collection)
            continue

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

    # Build output
    output = ""
    if denied_collections:
        output += f"⚠️ Access denied to: {', '.join(denied_collections)}\n\n"

    if not all_results:
        searched = [c for c in collections if c not in denied_collections]
        return output + f"No results found for query: '{query}'\nSearched collections: {', '.join(searched)}\nScore threshold: {score_threshold}"

    # Format results
    output += f"Found {len(all_results)} results:\n\n"
    for i, result in enumerate(all_results, 1):
        output += f"{i}. {result['entity_name']} (score: {result['score']:.3f})\n"
        output += f"   Collection: {result['collection']}\n"
        output += f"   {result['content'][:600]}...\n\n"

    return output


async def handle_upsert(args: Dict[str, Any], user_info: Dict[str, str]) -> str:
    """Create or update entity"""
    collection = args.get("collection")
    entity_name = args.get("entity_name")
    content = args.get("content")
    metadata = args.get("metadata", {})

    if not all([collection, entity_name, content]):
        return "Error: collection, entity_name, and content are required"

    # Check write access
    if not check_collection_access(user_info, collection, "write"):
        return f"⛔ Access denied: You don't have write access to collection '{collection}'"

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


async def handle_delete(args: Dict[str, Any], user_info: Dict[str, str]) -> str:
    """Delete entity"""
    collection = args.get("collection")
    entity_name = args.get("entity_name")

    if not all([collection, entity_name]):
        return "Error: collection and entity_name are required"

    # Check write access
    if not check_collection_access(user_info, collection, "write"):
        return f"⛔ Access denied: You don't have write access to collection '{collection}'"

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


async def handle_list_collections(args: Dict[str, Any], user_info: Dict[str, str]) -> str:
    """List all collections with their metadata"""
    if not qdrant_client:
        return "Error: Qdrant client not available"

    # Get all collections from Qdrant
    response = await qdrant_client.get("/collections")

    if response.status_code != 200:
        return f"Error: Failed to fetch collections (status: {response.status_code})"

    data = response.json()
    collections = data.get("result", {}).get("collections", [])

    if not collections:
        return "No collections found in Qdrant"

    # Build output
    output = f"Found {len(collections)} collections:\n\n"

    for coll in collections:
        name = coll.get("name", "unknown")

        # Check if user has read access
        has_access = check_collection_access(user_info, name, "read")
        access_marker = "✅" if has_access else "🔒"

        # Get collection info for point count
        coll_response = await qdrant_client.get(f"/collections/{name}")
        if coll_response.status_code == 200:
            coll_data = coll_response.json()
            point_count = coll_data.get("result", {}).get("points_count", 0)
            output += f"{access_marker} {name}\n"
            output += f"   Points: {point_count:,}\n"

            # Add vector config info
            vectors_config = coll_data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
            if "size" in vectors_config:
                output += f"   Vector size: {vectors_config['size']}\n"
            output += "\n"
        else:
            output += f"{access_marker} {name}\n   (Unable to fetch details)\n\n"

    # Add legend
    output += "Legend: ✅ = readable, 🔒 = no access"

    return output


async def handle_get(args: Dict[str, Any], user_info: Dict[str, str]) -> str:
    """Get full entity content by name"""
    collection = args.get("collection")
    entity_name = args.get("entity_name")

    if not all([collection, entity_name]):
        return "Error: collection and entity_name are required"

    # Check read access
    if not check_collection_access(user_info, collection, "read"):
        return f"⛔ Access denied: You don't have read access to collection '{collection}'"

    # Get point by ID (hash of entity_name)
    point_id = abs(hash(entity_name)) % (2**63)

    response = await qdrant_client.post(
        f"/collections/{collection}/points",
        json={"ids": [point_id], "with_payload": True}
    )

    if response.status_code != 200:
        return f"Error: Failed to fetch entity (status: {response.status_code})"

    data = response.json()
    points = data.get("result", [])

    if not points:
        return f"Entity '{entity_name}' not found in collection '{collection}'"

    point = points[0]
    payload = point.get("payload", {})

    # Format full output
    output = f"Entity: {payload.get('entity_name', entity_name)}\n"
    output += f"Collection: {collection}\n\n"
    output += "--- CONTENT ---\n"
    output += payload.get("content", "(no content)") + "\n\n"
    output += "--- METADATA ---\n"

    metadata = payload.get("metadata", {})
    for key, value in metadata.items():
        if isinstance(value, list):
            output += f"{key}: {', '.join(str(v) for v in value)}\n"
        else:
            output += f"{key}: {value}\n"

    return output


# MCP Endpoints
@app.post("/mcp")
async def mcp_endpoint(request: Request, authorization: str = Header(None)):
    """MCP protocol endpoint"""
    user_info = verify_token(authorization)

    body = await request.json()
    method = body.get("method")

    # ping (health check, allowed before initialize)
    if method == "ping":
        return {
            "jsonrpc": "2.0",
            "result": {},
            "id": body.get("id")
        }

    # initialize (MCP handshake)
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "Johnny v2 MCP Server",
                    "version": "2.0.0"
                }
            },
            "id": body.get("id")
        }

    # notifications (client notifications, server can ignore)
    if method.startswith("notifications/"):
        return {
            "jsonrpc": "2.0",
            "result": {},
            "id": body.get("id")
        }

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
            result = await handle_search(args, user_info)
        elif tool_name == "johnny_upsert":
            result = await handle_upsert(args, user_info)
        elif tool_name == "johnny_delete":
            result = await handle_delete(args, user_info)
        elif tool_name == "johnny_list_collections":
            result = await handle_list_collections(args, user_info)
        elif tool_name == "johnny_get":
            result = await handle_get(args, user_info)
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
