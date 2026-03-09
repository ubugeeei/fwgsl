# fwgsl

`fwgsl` is a pure functional language for WebGPU that compiles to [WGSL](https://www.w3.org/TR/WGSL/).

It targets the space between ML/Haskell ergonomics and GPU reality: algebraic data types, pattern matching, Hindley-Milner inference, shader entry-point attributes, and a toolchain that aims to feel like a modern programming language rather than a shader macro layer.

Written in Rust, the compiler is structured as a fast multi-crate pipeline and is heavily inspired by arena-oriented compiler design in projects like [Oxc](https://github.com/oxc-project/oxc).

## Highlights

- Pure functional surface language that lowers to valid WGSL.
- HM-style inference with explicit type signatures when needed.
- Algebraic data types and pattern matching.
- `where` bindings and expression-oriented `let`, `if`, and `match`.
- Static dimension-carrying array and vector types such as `Array 16 F32` and `Vec 3 F32`.
- Generic data declarations and polymorphic constructors, including phantom parameters.
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
| Dependent dimensions | Implemented | `Nat`-backed dimensions for `Array`, `Vec`, `Mat`-style type forms |
| Generic data / phantom types | Implemented | Polymorphic constructor schemes are preserved |
| WGSL code generation | Implemented | AST -> HIR -> MIR -> WGSL pipeline works end-to-end |
| LSP | Implemented | Diagnostics, hover, completion, goto-definition, semantic tokens |
| Playground | Implemented | Monaco editor, live diagnostics, hover/completion, WGSL output, WebGPU preview |
| Type classes | In progress | Surface direction is decided, full resolution is not merged yet |
| Kinds / HKT | In progress | Next major type-system layer |
| Algebraic effects | Planned | Will require syntax, typing, and handler-aware IR |
| Formatter / linter / LSP code actions | Planned | CST-based tooling is part of the roadmap |
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
weights : Array 4 F32
weights = [1.0, 2.0, 3.0, 4.0]

sample : Vec 3 F32 -> F32
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

For small verified sample programs, see [examples/README.md](/Users/nishimura/projects/oss/ubugeeei/fwgsl/examples/README.md).

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
mise run wasm
mise run playground
```

## Quick Start

Requirements:

- [Rust](https://rustup.rs/)
- [mise](https://mise.jdx.dev/)

Common commands:

```sh
# Build the workspace
mise run build

# Run the full test suite
mise run test

# Run formatting and linting tasks
mise run fmt
mise run lint

# Compile a file through the CLI
mise run cli -- compile examples/hello.fwgsl

# Start the playground dev server
mise run dev
```

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

| Crate | Purpose |
|-------|---------|
| `fwgsl_allocator` | Arena allocation helpers |
| `fwgsl_span` | Source spans, atoms, and source metadata |
| `fwgsl_diagnostics` | Structured diagnostics with labels and help text |
| `fwgsl_syntax` | `SyntaxKind` definitions for tokens and syntax nodes |
| `fwgsl_parser` | Hand-written lexer, layout resolver, parser |
| `fwgsl_typechecker` | Types, schemes, substitutions, unification, inference engine |
| `fwgsl_semantic` | Semantic analysis, environment building, constructor/type registration |
| `fwgsl_hir` | Desugared typed high-level IR |
| `fwgsl_mir` | Lowered WGSL-oriented IR |
| `fwgsl_wgsl_codegen` | MIR to WGSL text emitter |
| `fwgsl_language_server` | LSP server implementation |
| `fwgsl_wasm` | WASM bindings for the playground |
| `fwgsl_cli` | CLI entry point |
| `fwgsl_integration_tests` | End-to-end compiler pipeline tests |

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
- Type-level naturals in array/vector-like type applications.
- Generic type variables in signatures and data declarations.

## Roadmap

The next major areas are:

1. Kind checking and higher-kinded type parameters.
2. Type class declaration and instance resolution.
3. Algebraic effects and handler-aware intermediate representations.
4. Module system and bundling.
5. CST-based formatter and lint rule framework.
6. LSP upgrades such as inlay hints, code actions, and more structural navigation.

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

- Parser and diagnostics improvements in `crates/fwgsl_parser`
- Semantic/type-system work in `crates/fwgsl_semantic` and `crates/fwgsl_typechecker`
- WGSL lowering/codegen in `crates/fwgsl_mir` and `crates/fwgsl_wgsl_codegen`
- Editor experience in `crates/fwgsl_language_server` and `playground/`

This repository uses `mise` for task entry points and keeps the workspace split into small crates.

## License

MIT
