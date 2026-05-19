# fwgsl task runner.
#
# Run `just --list` to see every recipe. The names match the old
# `mise run` task aliases so muscle memory keeps working.

# Default recipe: print the recipe list.
default:
    @just --list

# Type-check the MoonBit workspace.
check:
    moon check

# Build the MoonBit workspace.
build:
    moon build

# Build the playground wasm (release, with the JS-string-builtins
# bindings) and surface it under `playground/pkg/fwgsl_wasm_bg.wasm`.
build-playground:
    moon build --target wasm-gc --release
    mkdir -p playground/pkg
    cp _build/wasm-gc/release/build/wasm/wasm.wasm \
       playground/pkg/fwgsl_wasm_bg.wasm

# Run the MoonBit test suite.
test:
    moon test

# Format every MoonBit source file in-place.
fmt:
    moon fmt

# Check the MoonBit formatting without rewriting.
fmt-check:
    moon fmt --check

# Serve the playground locally on http://localhost:3000.
# Build the wasm first so the page picks up the latest compiler.
playground: build-playground
    @echo "🚀 playground: http://localhost:3000"
    @cd playground && python3 -m http.server 3000

# Same as `playground` but skips the build (when iterating on HTML/JS).
dev:
    @echo "🚀 playground: http://localhost:3000"
    @cd playground && python3 -m http.server 3000

# Bump the moon.mod.json version to the next alpha.
bump-alpha:
    #!/usr/bin/env bash
    set -euo pipefail
    file="moon.mod.json"
    current=$(grep '"version"' "$file" | head -1 | sed -E 's/.*"([0-9A-Za-z.+-]+)".*/\1/')
    echo "Current version: $current"
    if [[ "$current" == *"-alpha."* ]]; then
      base="${current%-alpha.*}"
      num="${current##*-alpha.}"
      next=$((num + 1))
      new="${base}-alpha.${next}"
    else
      new="${current}-alpha.1"
    fi
    echo "New version: $new"
    sed -i.bak "s/\"version\": \"${current}\"/\"version\": \"${new}\"/" "$file" && rm "${file}.bak"
    echo "Bumped to $new"

# Drop the pre-release suffix in moon.mod.json for a stable release.
bump-release:
    #!/usr/bin/env bash
    set -euo pipefail
    file="moon.mod.json"
    current=$(grep '"version"' "$file" | head -1 | sed -E 's/.*"([0-9A-Za-z.+-]+)".*/\1/')
    base="${current%-alpha.*}"
    if [ "$base" = "$current" ]; then
      echo "Already a stable version: $current"
      exit 0
    fi
    sed -i.bak "s/\"version\": \"${current}\"/\"version\": \"${base}\"/" "$file" && rm "${file}.bak"
    echo "Released $base"
