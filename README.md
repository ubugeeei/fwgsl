# shadml

`shadml` is a pure functional language for WebGPU that compiles to [WGSL](https://www.w3.org/TR/WGSL/).

It targets the space between ML/Haskell ergonomics and GPU reality: algebraic data types, pattern matching, Hindley-Milner inference, traits with operator overloading, modules, and a toolchain that aims to feel like a modern programming language rather than a shader macro layer.

Written in Rust, the compiler is structured as a fast multi-crate pipeline and is heavily inspired by arena-oriented compiler design in projects like [Oxc](https://github.com/oxc-project/oxc).

## Highlights

- Pure functional surface language that lowers to valid WGSL.
- HM-style inference with explicit type signatures when needed.
- Algebraic data types, pattern matching, and `where` bindings.
- Traits and operator overloading (arithmetic, bitwise, and user-defined).
- Module system with imports, selective exports, and dead-code elimination.
- Bitfields for packed flag words with typed enum fields.
- Named tail-recursive `loop` expressions that emit native WGSL loops.
- `foldRange` combinator for compile-time loop desugaring.
- `const` declarations for module-level constants.
- Static dimension-carrying types: `Vec<3, F32>`, `Mat<4, 4, F32>`, `Array<F32, 64>`.
- Conditional compilation with `when cfg.feature` / `else when` / `else`.
- Rich editor support through LSP and the web playground.

## What It Looks Like

### Functions and `where` bindings

```haskell
scaleAndBias : F32 -> F32 -> F32
scaleAndBias x bias = scaled + bias
  where
    scaled = x * 2.0
```

### Algebraic data types and pattern matching

```haskell
data Shape = Circle F32 | Rect F32 F32

area : Shape -> F32
area shape = match shape
  | Circle r    -> 3.14159 * r * r
  | Rect w h    -> w * h
```

### Traits and operator overloading

```haskell
data Fp64 = Fp64 { high : F32, low : F32 }

impl Add Fp64 where
  (+) a b =
    let s = twoSum a.high b.high
        t = twoSum a.low  b.low
    in quickTwoSum (s.high, s.low + t.high)
```

### Compute shader with struct-based I/O

```haskell
data ComputeInput = ComputeInput {
  @builtin(global_invocation_id) gid : Vec<3, U32>
}

@group(0) @binding(0) storage(read_write) buf : Array<F32, 64>

main : ComputeInput -> ()
@compute @workgroup_size(64, 1, 1)
main input =
  let idx = toI32 input.gid.x
      val = load (buf[toU32 idx])
  in writeAt buf (toU32 idx) (val * 2.0)
```

### Loop expressions and `foldRange`

```haskell
-- Named tail-recursive loop
sumTo : I32 -> I32
sumTo n = loop go (acc = 0) (i = 0) in
  if i >= n then acc else go (acc + i) (i + 1)

-- Fold over a range (desugars to a WGSL for-loop)
sumRange : I32 -> I32 -> I32
sumRange lo hi = foldRange lo hi 0 (\acc i -> acc + i)
```

### Bitfields

```haskell
bitfield CapFlags : U32 = {
  endCap   : 1,
  startCap : 1,
  capRound : 1,
}

setStartCap : CapFlags -> CapFlags
setStartCap flags = flags { startCap = 1 }
```

### Conditional compilation

```haskell
when cfg.debug
  trace : F32 -> F32
  trace x = x
else
  trace : F32 -> F32
  trace x = 0.0
```

For more examples, see [examples/README.md](examples/README.md).

## Current Status

| Area | Status | Notes |
|------|--------|-------|
| Lexer + layout resolver | Implemented | Indentation-sensitive layout with virtual braces/semicolons |
| Parser | Implemented | Recursive descent + Pratt parser |
| HM inference | Implemented | Let-generalization and unification-based inference |
| ADTs | Implemented | Constructors, records, enums with discriminants |
| Pattern matching | Implemented | Constructor, literal, wildcard, multi-value, when-guards |
| `let` / `where` | Implemented | `where` desugars to local bindings |
| Traits | Implemented | Static dispatch, operator overloading, method-call syntax |
| Modules | Implemented | Imports, selective exports, topological ordering, DCE |
| Bitfields | Implemented | Packed flags with typed enum fields, construction, update |
| Loop / foldRange | Implemented | Named tail-recursive loops, fold-over-range combinator |
| Constants | Implemented | `const NAME : TYPE = EXPR` at module level |
| Conditional compilation | Implemented | `when cfg.x` / `else when` / `else` with `--feature` flags |
| Dimension types | Implemented | `Nat`-backed dimensions for `Vec`, `Mat`, `Array` |
| Generic data / phantom types | Implemented | Polymorphic constructor schemes |
| WGSL code generation | Implemented | AST -> HIR -> MIR -> WGSL pipeline end-to-end |
| LSP | Implemented | Diagnostics, hover, completion, goto-definition, references, semantic tokens |
| Playground | Implemented | Monaco editor, live diagnostics, hover/completion, WGSL output, WebGPU preview |
| Kinds / HKT | Planned | Next major type-system layer |
| Algebraic effects | Planned | Will require syntax, typing, and handler-aware IR |
| Formatter / linter / LSP code actions | In progress | CST-based tooling is part of the roadmap |

## Tooling Experience

### Language Server

The `shadml-lsp` binary provides:

- Parse and semantic diagnostics on open/change.
- Rich completion items with snippets, details, and markdown documentation.
- Context-aware completion for values, types, and WGSL-style attributes.
- Hover for keywords, built-ins, attributes, constructors, and typed bindings.
- Go to definition and find references for local declarations.
- Semantic tokens for keywords, operators, decorators, types, literals, and comments.

### Web Playground

The playground uses Monaco and the WASM compiler build.

Current editor feedback includes:

- Live compile on typing with debounce.
- Inline Monaco markers and whole-line decorations for diagnostics.
- Diagnostics panel with severity, note/help text, and error codes.
- Editor completion and hover mirroring the LSP experience.
- Read-only WGSL output panel.
- WebGPU preview for fragment and compute shaders.

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

# Compile a file through the CLI
cargo run -p shadml_cli -- compile examples/hello.shadml

# Type-check a file
cargo run -p shadml_cli -- check examples/lines.shadml

# Start the playground dev server
mise run dev
```

## Compiler Pipeline

`shadml` is split into explicit phases so language work, tooling work, and WGSL lowering can evolve independently.

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
  - trait / impl resolution
  - HM inference
    |
HIR
  - desugared
  - type-annotated
  - trait calls resolved
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
| `shadml_allocator` | Arena allocation helpers |
| `shadml_span` | Source spans, atoms, and source metadata |
| `shadml_diagnostics` | Structured diagnostics with labels and help text |
| `shadml_syntax` | `SyntaxKind` definitions for tokens and syntax nodes |
| `shadml_parser` | Hand-written lexer, layout resolver, parser, module resolver |
| `shadml_typechecker` | Types, schemes, substitutions, unification, inference engine |
| `shadml_semantic` | Semantic analysis, environment building, trait/impl resolution |
| `shadml_ast` | AST node definitions |
| `shadml_ast_lowering` | AST to HIR lowering with trait dispatch and desugaring |
| `shadml_hir` | Desugared typed high-level IR |
| `shadml_mir` | Lowered WGSL-oriented IR |
| `shadml_wgsl_codegen` | MIR to WGSL text emitter |
| `shadml_ide` | Shared IDE logic (completions, hover, goto-def, references) |
| `shadml_language_server` | LSP server implementation |
| `shadml_formatter` | CST-based code formatter |
| `shadml_wasm` | WASM bindings for the playground |
| `shadml_cli` | CLI entry point |
| `shadml_integration_tests` | End-to-end compiler pipeline tests |

## Language Design Direction

- ML-derived, Haskell-influenced pure functional syntax.
- Everything is an expression.
- Operators are functions and can be sectioned or used infix.
- Strong static typing with inference first, annotations when needed.
- Trait-based operator overloading for user types.
- Rich algebraic data modeling that lowers to WGSL's restricted runtime model.
- Tooling-first compiler architecture: parser, diagnostics, LSP, formatter, linter, playground.

## WGSL Constraints

WGSL is intentionally restrictive. `shadml` exists to bridge that gap.

| WGSL Constraint | Compiler Direction |
|----------------|--------------------|
| No recursion | Detect and lower acceptable cases; reject unsupported recursion |
| No first-class functions | Lower higher-order structure toward first-order representations |
| No runtime generics | Specialize polymorphism during lowering |
| No native ADTs | Encode constructors as WGSL-friendly layouts |
| GPU-oriented fixed layouts | Preserve as much static information as possible in the source type system |

## Contributing

A good starting point is usually one of:

- Parser and diagnostics improvements in `crates/shadml_parser`
- Semantic/type-system work in `crates/shadml_semantic` and `crates/shadml_typechecker`
- WGSL lowering/codegen in `crates/shadml_mir` and `crates/shadml_wgsl_codegen`
- Editor experience in `crates/shadml_language_server` and `playground/`

This repository uses `mise` for task entry points and keeps the workspace split into small crates.

## License

MIT
