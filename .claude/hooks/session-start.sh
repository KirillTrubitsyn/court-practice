#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# 1) Installs Python dev deps so pytest/ruff/mypy work in the sandbox.
# 2) If RAILWAY_TOKEN is set (Service Token, configured via Repository Secrets),
#    installs Railway CLI and syncs that service's variables into:
#      - .env             (read by pydantic-settings via app/config.py)
#      - $CLAUDE_ENV_FILE (exported into the shell for the rest of the session)
#
# Idempotent: safe to run repeatedly. No-op outside Claude Code on the web.

set -euo pipefail

log() { printf '[session-start] %s\n' "$*" >&2; }

# Skip on local desktop sessions — user has their own venv/.env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    log "not in Claude Code remote env — skipping"
    exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# --- 1. Python dev deps ---------------------------------------------------
# --user + --ignore-installed: sandbox имеет часть пакетов через apt без pip
# RECORD-файлов, иначе pip падает «Cannot uninstall <pkg>...installed by debian».
if [ -f requirements-dev.txt ]; then
    log "installing Python dev deps (user-local)"
    pip install --quiet --no-input --disable-pip-version-check \
        --user --ignore-installed -r requirements-dev.txt
    # Добавляем ~/.local/bin в PATH сессии — там оказываются ruff/mypy/pytest.
    if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
        printf 'export PATH="%s/.local/bin:$PATH"\n' "$HOME" >> "$CLAUDE_ENV_FILE"
    fi
else
    log "requirements-dev.txt not found — skipping pip install"
fi

# --- 2. Railway env sync --------------------------------------------------
if [ -z "${RAILWAY_TOKEN:-}" ]; then
    log "RAILWAY_TOKEN not set — skipping Railway sync"
    log "  (set RAILWAY_TOKEN service token in Repository Secrets to enable)"
    exit 0
fi

# Install Railway CLI on first run (cached on subsequent sessions).
if ! command -v railway >/dev/null 2>&1; then
    log "installing Railway CLI"
    curl -fsSL https://railway.com/install.sh | sh >&2
    export PATH="${HOME}/.railway/bin:${PATH}"
fi

# Re-export PATH in case CLI is at $HOME/.railway/bin.
export PATH="${HOME}/.railway/bin:${PATH}"

if ! command -v railway >/dev/null 2>&1; then
    log "ERROR: Railway CLI install failed"
    exit 0  # don't block session start
fi

log "fetching Railway service variables"
TMP_JSON="$(mktemp)"
TMP_ERR="$(mktemp)"
trap 'rm -f "$TMP_JSON" "$TMP_ERR"' EXIT

if ! railway variables --json >"$TMP_JSON" 2>"$TMP_ERR"; then
    log "ERROR: 'railway variables' failed:"
    sed 's/^/  /' "$TMP_ERR" >&2
    exit 0
fi

# Parse JSON and emit two artefacts:
#   .env              -> KEY="value" lines for pydantic-settings
#   $CLAUDE_ENV_FILE  -> export KEY='value' lines for the shell
python3 - "$TMP_JSON" "${CLAUDE_ENV_FILE:-/dev/null}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

vars_path, shell_env_path = sys.argv[1], sys.argv[2]
data = json.loads(Path(vars_path).read_text())
if not isinstance(data, dict):
    print(f"[session-start] unexpected railway output: {type(data).__name__}", file=sys.stderr)
    sys.exit(0)

dotenv_lines = []
shell_lines = []
for key, value in sorted(data.items()):
    s = "" if value is None else str(value)
    # .env: double-quote, escape backslash and quote
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    dotenv_lines.append(f'{key}="{escaped}"')
    # shell: shlex.quote handles all edge cases
    shell_lines.append(f"export {key}={shlex.quote(s)}")

Path(".env").write_text("\n".join(dotenv_lines) + "\n")
print(f"[session-start] wrote {len(dotenv_lines)} variables to .env", file=sys.stderr)

if shell_env_path != "/dev/null":
    with open(shell_env_path, "a") as f:
        f.write("\n".join(shell_lines) + "\n")
    print(f"[session-start] exported {len(shell_lines)} variables to CLAUDE_ENV_FILE", file=sys.stderr)
PY

log "done"
