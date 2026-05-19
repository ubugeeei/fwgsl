# fwgsl

`fwgsl` is a pure functional language for WebGPU that compiles to [WGSL](https://www.w3.org/TR/WGSL/).

It targets the space between ML/Haskell ergonomics and GPU reality: algebraic data types, pattern matching, Hindley-Milner inference, shader entry-point attributes, and a toolchain that aims to feel like a modern programming language rather than a shader macro layer.

Written in [MoonBit](https://www.moonbitlang.com/), the compiler is a multi-package pipeline that targets `wasm-gc` natively so the CLI, LSP, and web playground all share one garbage-collected codebase. Sources live under `moonbit/`; see `moonbit/README.md` for the per-package layout. The migration from the original Rust implementation is tracked by [#4](https://github.com/ubugeeei/fwgsl/issues/4) and has now completed.

## Highlights

- Pure functional surface language that lowers to valid WGSL.
- HM-style inference with explicit type signatures when needed.
- Algebraic data types and pattern matching.
- `where` bindings and expression-oriented `let`, `if`, and `match`.
- Static dimension-carrying tensor types such as `Tensor 16 F32`, `Vector 3 F32`, and `Matrix 4 4 F32`.
- Generic data declarations and polymorphic constructors, including phantom parameters.
- Builtin `Option`, `Result`, `Pair`, and utility functions such as `$map`, `$zip`, and `$unwrapOr`.
- Rich editor support through LSP and the web playground.

## Current Status

The repository contains both implemented features and planned language goals. The table below is the current state of the codebase.

| Area | Status | Notes |
|------|--------|-------|
| Lexer + layout resolver | Implemented | Indentation-sensitive layout with virtual braces/semicolons |
| Parser | Implemented | Recursive descent + Pratt parser |
| HM inference | Implemented | Let-generalization and unification-based inference |
| ADTs | Implemented | Constructors are registered in semantic analysis |
| Pattern matching | Implemented | Match expressions and constructor patterns |
| `let` / `where` | Implemented | `where` currently desugars to sequential local bindings |
| Dependent dimensions | Implemented | `Nat`-backed dimensions for `Tensor`, `Vector`, `Matrix`, and short aliases such as `Ten`, `Vec`, `Mat`, `Sca` |
| Generic data / phantom types | Implemented | Polymorphic constructor schemes are preserved |
| WGSL code generation | Implemented | AST -> HIR -> MIR -> WGSL pipeline works end-to-end |
| LSP | Implemented | Diagnostics, hover, completion, goto-definition, semantic tokens |
| Playground | Implemented | Monaco editor, live diagnostics, hover/completion, WGSL output, WebGPU preview |
| CLI | Implemented | Argv dispatch with `compile` / `check` / `fmt` / `version` / `help`, `--source` and `--stdin` inputs |
| Formatter | Implemented | Token-stream formatter that canonicalises whitespace; CST-based reformatting is the next step |
| Type classes | In progress | Surface direction is decided, full resolution is not merged yet |
| Kinds / HKT | In progress | Next major type-system layer |
| Algebraic effects | Planned | Will require syntax, typing, and handler-aware IR |
| Linter | Planned | AST-walking lint rule framework with first rules (unused bindings, redundant wildcards) |
| LSP code actions / inlay hints | Planned | Quick-fix-style edits and inferred-type hints on the LSP surface |
| Modules / bundler / web playground polish | Planned | Base structure exists, full system is not complete |

## What It Looks Like

### A small functional program

```haskell
add : I32 -> I32 -> I32
add x y = x + y

double : I32 -> I32
double x = x * 2

main : I32 -> I32
main x =
  let y = double x
  in y + 1
```

### Algebraic data types and pattern matching

```haskell
data Color = Red | Green | Blue

toI32 : Color -> I32
toI32 color = match color
  | Red -> 0
  | Green -> 1
  | Blue -> 2
```

### `where` bindings

```haskell
scaleAndBias : I32 -> I32
scaleAndBias x = y * 2
where
  y = x + 1
```

### Dimension-carrying types

```haskell
weights : Tensor 4 F32
weights = [1.0, 2.0, 3.0, 4.0]

sample : Vector 3 F32 -> F32
sample v = $dot v v
```

### Generic and phantom data

```haskell
data Box a = Box a
data Phantom a = Phantom

unbox : Box a -> a
unbox value = match value
  | Box x -> x
```

### Shader entry points

```haskell
@compute @workgroup_size(64, 1, 1)
main idx =
  let doubled = idx * 2
  in doubled
```

For small verified sample programs, see [examples/README.md](examples/README.md).

## Tooling Experience

### Language Server

The `fwgsl-lsp` binary currently provides:

- Parse and semantic diagnostics on open/change.
- Rich completion items with snippets, details, and markdown documentation.
- Context-aware completion for values, types, and WGSL-style attributes such as `@compute` and `@workgroup_size`.
- Hover for keywords, built-ins, attributes, constructors, and document-local typed bindings.
- Go to definition for local declarations.
- Semantic tokens for keywords, operators, decorators, types, literals, and comments.

### Web Playground

The playground uses Monaco and the WASM compiler build.

Current editor feedback includes:

- Live compile on typing with debounce.
- Inline Monaco markers and whole-line decorations for diagnostics.
- Diagnostics panel with severity, note/help text, and error codes.
- Editor completion and hover mirroring the LSP experience.
- Read-only WGSL output panel.
- WebGPU preview for compute shaders.

Run it locally with:

```sh
just playground
```

## Quick Start

Requirements:

- [Nix](https://nixos.org/) with flakes — `nix develop` drops you into a shell with Node 22, `just`, and the MoonBit installer auto-run on first entry.
- (Without Nix) [MoonBit](https://www.moonbitlang.com/) installed manually with `curl -fsSL https://cli.moonbitlang.com/install/unix.sh | bash`, plus `just` from your package manager.

Common commands (every recipe is in `justfile`):

```sh
# Type-check the whole MoonBit workspace
just check            # or: moon check

# Build everything
just build            # or: moon build

# Run the full test suite
just test             # or: moon test

# Format MoonBit sources
just fmt              # or: moon fmt

# Build the playground wasm + serve it on :3000
just playground

# Serve the playground without rebuilding the wasm (HTML/JS iteration)
just dev
```

[direnv](https://direnv.net/) users can `direnv allow` so the dev shell loads automatically.

## Compiler Pipeline

`fwgsl` is split into explicit phases so language work, tooling work, and WGSL lowering can evolve independently.

```text
Source Text
    |
Lexer
    |
Layout Resolver
    |
Parser
    |
AST
    |
Semantic Analysis
  - name resolution
  - constructor registration
  - HM inference
    |
HIR
  - desugared
  - type-annotated
    |
MIR
  - WGSL-shaped
  - first-order
    |
WGSL Codegen
```

## Repository Layout

Every compiler stage lives under `moonbit/`:

| Package | Purpose |
|---------|---------|
| `moonbit/span` | Source spans, atoms, and source metadata |
| `moonbit/diagnostics` | Structured diagnostics with labels and help text |
| `moonbit/syntax` | `SyntaxKind` definitions for tokens and syntax nodes |
| `moonbit/cst` | Concrete syntax tree |
| `moonbit/ast` | AST type definitions |
| `moonbit/parser` | Hand-written lexer, layout resolver, parser |
| `moonbit/typechecker` | Types, schemes, substitutions, unification, inference engine |
| `moonbit/semantic` | Semantic analysis, environment building, constructor / type registration |
| `moonbit/ast_lowering` | AST -> HIR lowering with full Hindley-Milner inference |
| `moonbit/hir` | Desugared typed high-level IR |
| `moonbit/mir` | Lowered WGSL-oriented IR plus the HIR -> MIR pass |
| `moonbit/wgsl_codegen` | MIR to WGSL text emitter |
| `moonbit/ide` | IDE-facing analyses shared with the LSP and playground |
| `moonbit/language_server` | LSP server implementation (publishDiagnostics, hover, completion, semantic tokens, formatting) |
| `moonbit/formatter` | Token-stream formatter shared by the CLI's `fwgsl fmt` and the LSP's `textDocument/formatting` |
| `moonbit/wasm` | wasm-gc bindings for the playground (`wasm_check` / `wasm_compile`) |
| `moonbit/cli` | CLI library + `moonbit/cli/cmd/main` binary entry point |
| `moonbit/integration_tests` | End-to-end pipeline tests asserting WGSL output |

## Language Design Direction

The target language is intentionally ambitious. The medium-term direction is:

- ML-derived, Haskell-influenced pure functional syntax.
- No symbolic dependency on idioms like `<$>` for core ergonomics.
- Everything is an expression.
- Operators are functions and can be sectioned or used infix with backticks.
- Strong static typing with inference first, annotations when needed.
- Rich algebraic data modeling that can still lower to WGSL's restricted runtime model.
- Tooling-first compiler architecture: parser, diagnostics, LSP, formatter, linter, playground.

## Implemented Language Surface

What is already working in the compiler today:

- Function declarations and type signatures.
- Lambda expressions.
- Function application and infix operators.
- `let ... in ...`.
- `where` clauses on function declarations.
- `if ... then ... else ...`.
- ADTs and constructor registration.
- Pattern matching over constructors.
- Entry-point attributes such as `@compute`.
- Type-level naturals in tensor/vector/matrix-like type applications.
- Builtin `Option`, `Result`, `Pair`, and utility prelude entries.
- Generic type variables in signatures and data declarations.

## Roadmap

The next major areas are:

1. Kind checking and higher-kinded type parameters.
2. Type class declaration and instance resolution.
3. Algebraic effects and handler-aware intermediate representations.
4. Module system and bundling.
5. CST-based formatter (the token-stream pass under `moonbit/formatter` is the starter; full structural reformatting is the follow-up).
6. Linter scaffolding plus first lint rules (unused bindings, redundant wildcards, eventually exhaustiveness and shadowing).
7. LSP upgrades such as inlay hints, code actions, and more structural navigation.

## WGSL Constraints

WGSL is intentionally restrictive. `fwgsl` exists to bridge that gap.

| WGSL Constraint | Compiler Direction |
|----------------|--------------------|
| No recursion | Detect and lower acceptable cases; reject unsupported recursion |
| No first-class functions | Lower higher-order structure toward first-order representations |
| No runtime generics | Specialize polymorphism during lowering |
| No native ADTs | Encode constructors as WGSL-friendly layouts |
| GPU-oriented fixed layouts | Preserve as much static information as possible in the source type system |

## Contributing

A good starting point is usually one of:

- Parser and diagnostics improvements in `moonbit/parser` and `moonbit/diagnostics`
- Semantic/type-system work in `moonbit/semantic` and `moonbit/typechecker`
- WGSL lowering/codegen in `moonbit/mir` and `moonbit/wgsl_codegen`
- Editor experience in `moonbit/ide`, `moonbit/language_server`, and `playground/`

This repository uses [Nix](https://nixos.org/) (`flake.nix` + `direnv`) for the dev shell and [just](https://github.com/casey/just) (`justfile`) for task entry points. The workspace stays split into small MoonBit packages under `moonbit/`.

## License

MIT
