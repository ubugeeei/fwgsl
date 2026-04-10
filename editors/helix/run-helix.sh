#!/usr/bin/env bash
# Launch Helix with fwgsl language support using a temporary config directory.
# Usage: ./editors/helix/run-helix.sh [file.fwgsl ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HX_DIR="/tmp/fwgsl-hx"
CONFIG_DIR="$HX_DIR/helix"
RUNTIME_DIR="$HX_DIR/helix/runtime"

# -- Set up config directory --------------------------------------------------

mkdir -p "$CONFIG_DIR"
mkdir -p "$RUNTIME_DIR/queries/fwgsl"
mkdir -p "$RUNTIME_DIR/grammars"

# Generate languages.toml with absolute grammar path
# (Helix resolves relative paths from the config file location, which is in /tmp,
# so we need to rewrite the grammar source path to be absolute.)
sed "s|path = \"../../tree-sitter-fwgsl\"|path = \"$REPO_ROOT/tree-sitter-fwgsl\"|" \
  "$REPO_ROOT/editors/helix/languages.toml" > "$CONFIG_DIR/languages.toml"

# Symlink query files (from tree-sitter-fwgsl if available, else editors/helix)
if [ -d "$REPO_ROOT/tree-sitter-fwgsl/queries" ]; then
  QUERY_SRC="$REPO_ROOT/tree-sitter-fwgsl/queries"
else
  QUERY_SRC="$REPO_ROOT/editors/helix/queries"
fi

for f in "$QUERY_SRC"/*.scm; do
  [ -f "$f" ] && ln -sf "$f" "$RUNTIME_DIR/queries/fwgsl/$(basename "$f")"
done

# -- Build LSP server ---------------------------------------------------------

echo "Building fwgsl-lsp..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" -p fwgsl_language_server --release 2>&1 | tail -1

LSP_BIN="$REPO_ROOT/target/release/fwgsl-lsp"
if [ ! -f "$LSP_BIN" ]; then
  echo "Warning: fwgsl-lsp binary not found at $LSP_BIN"
fi

# -- Build tree-sitter grammar ------------------------------------------------

export XDG_CONFIG_HOME="$HX_DIR"
export PATH="$REPO_ROOT/target/release:$PATH"

echo "Building fwgsl tree-sitter grammar..."
hx --grammar build 2>&1 | grep -E "(fwgsl|error)" || true

# -- Launch Helix -------------------------------------------------------------

echo "Config:  $CONFIG_DIR/languages.toml"
echo "Runtime: $RUNTIME_DIR"
echo "LSP:     $LSP_BIN"
echo ""

exec hx "$@"
