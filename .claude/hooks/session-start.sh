#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# 1) Installs Python dev deps so pytest/ruff/mypy work in the sandbox.
# 2) Materialises a .env (and exports vars into the session shell) from
#    secrets injected by Claude Code into the process environment.
#    The list of recognised variables is derived from .env.example, so
#    adding a new key there is enough — no hook edit required.
#
# Why not pull from Railway: the cloud sandbox blocks egress to
# *.railway.app / railway.com (HTTP 403 from the proxy), so neither the
# Railway CLI nor the GraphQL API are reachable. Secrets must be set in
# Repository Secrets in Claude Code on the web — they then appear as env
# vars in this hook, the same way RAILWAY_TOKEN does.
#
# Idempotent: safe to run repeatedly. No-op outside Claude Code on the web.

set -euo pipefail

log() { printf '[session-start] %s\n' "$*" >&2; }

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    log "not in Claude Code remote env — skipping"
    exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# --- 1. Python dev deps ---------------------------------------------------
# --user + --ignore-installed: в sandbox часть пакетов установлена через apt
# без RECORD-файлов, иначе pip падает «Cannot uninstall <pkg>...installed by debian».
if [ -f requirements-dev.txt ]; then
    log "installing Python dev deps (user-local)"
    pip install --quiet --no-input --disable-pip-version-check \
        --user --ignore-installed -r requirements-dev.txt
    if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
        printf 'export PATH="%s/.local/bin:$PATH"\n' "$HOME" >> "$CLAUDE_ENV_FILE"
    fi
else
    log "requirements-dev.txt not found — skipping pip install"
fi

# --- 2. Materialise .env from injected secrets ----------------------------
if [ ! -f .env.example ]; then
    log ".env.example not found — skipping .env sync"
    exit 0
fi

python3 - "${CLAUDE_ENV_FILE:-/dev/null}" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

shell_env_path = sys.argv[1]

# Имена переменных из .env.example, включая закомментированные опциональные.
name_re = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=")
names = []
seen = set()
for line in Path(".env.example").read_text().splitlines():
    m = name_re.match(line)
    if m and m.group(1) not in seen:
        seen.add(m.group(1))
        names.append(m.group(1))

dotenv_lines = []
shell_lines = []
present = []
for key in names:
    if key not in os.environ:
        continue
    value = os.environ[key]
    present.append(key)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    dotenv_lines.append(f'{key}="{escaped}"')
    shell_lines.append(f"export {key}={shlex.quote(value)}")

if not dotenv_lines:
    print(
        "[session-start] no recognised secrets in env — "
        "add them to Repository Secrets in Claude Code on the web",
        file=sys.stderr,
    )
    sys.exit(0)

Path(".env").write_text("\n".join(dotenv_lines) + "\n")
print(
    f"[session-start] wrote {len(dotenv_lines)} variables to .env: "
    f"{', '.join(present)}",
    file=sys.stderr,
)

required = ("VOYAGE_API_KEY", "MCP_SECRET_KEY")
missing = [k for k in required if k not in os.environ]
if missing:
    print(
        f"[session-start] WARNING: required secrets missing: {', '.join(missing)}",
        file=sys.stderr,
    )

if shell_env_path != "/dev/null":
    with open(shell_env_path, "a") as f:
        f.write("\n".join(shell_lines) + "\n")
    print(
        f"[session-start] exported {len(shell_lines)} variables to CLAUDE_ENV_FILE",
        file=sys.stderr,
    )
PY

log "done"
