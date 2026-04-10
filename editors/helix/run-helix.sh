#!/usr/bin/env bash
# Launch Helix with shadml language support using a temporary config directory.
# Usage: ./editors/helix/run-helix.sh [file.shadml ...]
#
# Prerequisites (handled by mise):
#   - shadml-lsp binary (built by `mise run release`)
#   - shadml.dylib grammar (built by `mise run grammar:build`)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HX_DIR="/tmp/shadml-hx"
CONFIG_DIR="$HX_DIR/helix"
RUNTIME_DIR="$HX_DIR/helix/runtime"

# -- Set up config directory --------------------------------------------------

mkdir -p "$CONFIG_DIR"
mkdir -p "$RUNTIME_DIR/queries/shadml"
mkdir -p "$RUNTIME_DIR/grammars"

# Copy languages.toml as-is (grammar is pre-built, no source path needed)
cp "$REPO_ROOT/editors/helix/languages.toml" "$CONFIG_DIR/languages.toml"

# Symlink query files (from tree-sitter-shadml if available, else editors/helix)
if [ -d "$REPO_ROOT/tree-sitter-shadml/queries" ]; then
  QUERY_SRC="$REPO_ROOT/tree-sitter-shadml/queries"
else
  QUERY_SRC="$REPO_ROOT/editors/helix/queries"
fi

for f in "$QUERY_SRC"/*.scm; do
  [ -f "$f" ] && ln -sf "$f" "$RUNTIME_DIR/queries/shadml/$(basename "$f")"
done

# -- Copy pre-built grammar ---------------------------------------------------

GRAMMAR_DIR="$RUNTIME_DIR/grammars"
if [ -f "$REPO_ROOT/tree-sitter-shadml/shadml.dylib" ]; then
  cp "$REPO_ROOT/tree-sitter-shadml/shadml.dylib" "$GRAMMAR_DIR/shadml.dylib"
  cp "$REPO_ROOT/tree-sitter-shadml/shadml.dylib" "$GRAMMAR_DIR/shadml.so"
elif [ -f "$REPO_ROOT/tree-sitter-shadml/shadml.so" ]; then
  cp "$REPO_ROOT/tree-sitter-shadml/shadml.so" "$GRAMMAR_DIR/shadml.so"
  cp "$REPO_ROOT/tree-sitter-shadml/shadml.so" "$GRAMMAR_DIR/shadml.dylib"
fi

# -- Launch Helix -------------------------------------------------------------

LSP_BIN="$REPO_ROOT/target/release/shadml-lsp"
export PATH="$REPO_ROOT/target/release:$PATH"
export XDG_CONFIG_HOME="$HX_DIR"

echo "Config:  $CONFIG_DIR/languages.toml"
echo "Runtime: $RUNTIME_DIR"
echo "LSP:     $LSP_BIN"
echo ""

exec hx "$@"
