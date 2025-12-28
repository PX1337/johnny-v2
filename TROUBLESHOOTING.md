# Johnny MCP - Troubleshooting Claude Code Integration

## Problem: "Auth: ✘ not authenticated" w Claude Code

### Root Cause (NAPRAWIONE 2025-12-28)

Bug w Johnny `server.py` - odpowiedzi MCP nie były opakowane w JSON-RPC format.

**Błędny format (przed fix):**
```json
{"tools": [...]}
```

**Poprawny format (po fix):**
```json
{"jsonrpc": "2.0", "result": {"tools": [...]}, "id": 1}
```

Claude Code HTTP transport oczekuje pełnego JSON-RPC i zrywał połączenie gdy otrzymał niepoprawny format.

---

## Co NIE było problemem

1. **Token/Auth** - curl z Bearer token działał poprawnie
2. **Railway hosting** - API odpowiadało, health check OK
3. **mcp-remote** - działał poprawnie jako STDIO proxy
4. **Konfiguracja .mcp.json** - Claude Code poprawnie parsował config

---

## Diagnostyka krok po kroku

### 1. Test czy API odpowiada
```bash
curl -s https://johnny-v2-production.up.railway.app/health
# Powinno zwrócić: {"status": "ok", ...}
```

### 2. Test autentykacji
```bash
curl -s -X POST https://johnny-v2-production.up.railway.app/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

**Sprawdź format odpowiedzi!** Musi zaczynać się od `{"jsonrpc":"2.0","result":{...`

### 3. Sprawdź logi Claude Code
```bash
grep -i johnny ~/.claude/debug/latest | tail -30
```

Szukaj:
- `HTTP connection dropped after 0s uptime` - problem z formatem odpowiedzi
- `SDK auth error: HTTP 404` - Claude Code próbuje OAuth (fallback gdy HTTP fails)
- `Successfully connected` + natychmiastowy drop - problem JSON-RPC

### 4. Test mcp-remote ręcznie
```bash
npx mcp-remote https://johnny-v2-production.up.railway.app/mcp \
  --header "Authorization:Bearer TOKEN" &
sleep 3
ps aux | grep mcp-remote
kill %1
```

Powinno pokazać:
```
Connected to remote server using StreamableHTTPClientTransport
Local STDIO server running
Proxy established successfully
```

---

## Konfiguracja .mcp.json

### Opcja 1: HTTP transport (zalecana po fix)
```json
{
  "johnny": {
    "type": "http",
    "url": "https://johnny-v2-production.up.railway.app/mcp",
    "headers": {
      "Authorization": "Bearer TOKEN"
    }
  }
}
```

### Opcja 2: mcp-remote (STDIO proxy)
```json
{
  "johnny": {
    "command": "npx",
    "args": [
      "mcp-remote",
      "https://johnny-v2-production.up.railway.app/mcp",
      "--header",
      "Authorization:Bearer TOKEN"
    ]
  }
}
```

**UWAGA:** Brak spacji po dwukropku w `Authorization:Bearer` (Claude Code bug z escape'owaniem spacji).

---

## Fix w server.py (commit 279e9aa)

Zmienione metody:
- `tools/list` - dodany JSON-RPC wrapper
- `tools/call` - dodany JSON-RPC wrapper
- Error responses - dodany JSON-RPC error format

```python
# PRZED (błędnie)
if method == "tools/list":
    return {"tools": MCP_TOOLS}

# PO (poprawnie)
if method == "tools/list":
    return {
        "jsonrpc": "2.0",
        "result": {"tools": MCP_TOOLS},
        "id": body.get("id")
    }
```

---

## Po zmianach w .mcp.json lub server.py

**ZAWSZE restartuj Claude Code** - zamknij terminal i otwórz nowy.

---

## Znane bugi Claude Code (grudzień 2024)

1. **#2831, #7290** - Claude Code ignoruje Bearer token w HTTP transport i próbuje OAuth discovery
2. **Escape spacji** - problem z argumentami zawierającymi spacje w STDIO transport
3. **env block** - Claude Code nie rozpoznaje `"env": {}` w konfiguracji MCP

---

## Przydatne komendy

```bash
# Status deploymentu
cd ~/projects/johnny-v2 && railway deployment list

# Ręczny deploy
cd ~/projects/johnny-v2 && railway up

# Logi Railway (interaktywne)
cd ~/projects/johnny-v2 && railway logs

# Test pełnego MCP handshake
curl -s -X POST https://johnny-v2-production.up.railway.app/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}},"id":1}'
```

---

## Token Johnny
```
2d398e22edda565a40c609a3753a24123bc2be6e6328e701e4777f4b9b3e10b3
```
