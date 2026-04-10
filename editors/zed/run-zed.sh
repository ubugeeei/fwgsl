#!/usr/bin/env bash
# Set up and launch Zed with fwgsl language support.
#
# Usage: ./editors/zed/run-zed.sh [file.fwgsl ...]
#
# First run:
#   1. The script builds fwgsl-lsp
#   2. Install the dev extension in Zed:
#      Cmd+Shift+P -> "zed: install dev extension" -> select editors/zed/
#   3. Add fwgsl-lsp to your Zed settings (see output below)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXT_DIR="$REPO_ROOT/editors/zed"

# -- Build LSP server ---------------------------------------------------------

echo "Building fwgsl-lsp..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" -p fwgsl_language_server --release 2>&1 | tail -1

LSP_BIN="$REPO_ROOT/target/release/fwgsl-lsp"
if [ ! -f "$LSP_BIN" ]; then
  echo "Error: fwgsl-lsp binary not found at $LSP_BIN"
  exit 1
fi
echo "LSP: $LSP_BIN"

# -- Check if fwgsl-lsp is configured ----------------------------------------

ZED_SETTINGS="$HOME/.config/zed/settings.json"
if [ -f "$ZED_SETTINGS" ] && grep -q "fwgsl-lsp" "$ZED_SETTINGS"; then
  echo "fwgsl-lsp already configured in Zed settings."
else
  echo ""
  echo "Add the following to your Zed settings \"lsp\" block:"
  echo ""
  echo "  \"fwgsl-lsp\": { \"binary\": { \"path\": \"$LSP_BIN\" } }"
  echo ""
fi

# -- Clean up stale symlink from old script -----------------------------------

STALE_LINK="$HOME/.local/share/zed/extensions/installed/fwgsl"
if [ -L "$STALE_LINK" ]; then
  rm "$STALE_LINK"
  echo "Removed stale symlink: $STALE_LINK"
fi

# -- Launch Zed ---------------------------------------------------------------

export PATH="$REPO_ROOT/target/release:$PATH"

echo ""
echo "Dev extension dir: $EXT_DIR"
echo "Install once via: Cmd+Shift+P -> 'zed: install dev extension'"
echo ""

exec zed "$@"
