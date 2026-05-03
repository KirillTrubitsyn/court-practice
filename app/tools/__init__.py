"""Регистрация tools на FastMCP-инстансе.

Импорты — внутри register(), чтобы избежать циклов с server.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_all(mcp: FastMCP) -> None:
    from app.tools import (
        find_similar,
        get_case_details,
        list_tags,
        search_practice,
        stats,
    )

    search_practice.register(mcp)
    get_case_details.register(mcp)
    find_similar.register(mcp)
    list_tags.register(mcp)
    stats.register(mcp)
