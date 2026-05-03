"""Простой healthcheck — пингует /health и /mcp с авторизацией.

Запуск:
    python -m scripts.healthcheck https://your.app.up.railway.app $MCP_SECRET_KEY
"""

from __future__ import annotations

import sys

import httpx


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: healthcheck.py <base_url> <bearer_token>", file=sys.stderr)
        return 2
    base = sys.argv[1].rstrip("/")
    token = sys.argv[2]

    with httpx.Client(timeout=10.0) as client:
        r1 = client.get(f"{base}/health")
        print(f"GET /health -> {r1.status_code} {r1.text}")
        if r1.status_code != 200:
            return 1

        # MCP initialize handshake — минимальный JSON-RPC поверх Streamable HTTP.
        r2 = client.post(
            f"{base}/mcp",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "healthcheck", "version": "0.1"},
                },
            },
        )
        print(f"POST /mcp initialize -> {r2.status_code}")
        body_preview = r2.text[:300].replace("\n", " ")
        print(f"  body: {body_preview}")
        return 0 if r2.status_code in (200, 202) else 1


if __name__ == "__main__":
    raise SystemExit(main())
