# fwgsl (MoonBit)

This directory contains the MoonBit rewrite of the `fwgsl` compiler. The
Rust implementation under `crates/` is the current source of truth; the
MoonBit port reaches functional parity once the items in the Status table
below are all marked Done. Tracking issue: [#4](https://github.com/ubugeeei/fwgsl/issues/4).

## Layout

Each Rust crate has a corresponding MoonBit package directly under
`moonbit/`. Packages mirror the Rust dependency graph so we can land them
incrementally.

| Rust crate                | MoonBit package           |
| ------------------------- | ------------------------- |
| `fwgsl_span`              | `moonbit/span`            |
| `fwgsl_diagnostics`       | `moonbit/diagnostics`     |
| `fwgsl_syntax`            | `moonbit/syntax`          |
| `fwgsl_cst`               | `moonbit/cst`             |
| `fwgsl_ast`               | `moonbit/ast`             |
| `fwgsl_parser`            | `moonbit/parser`          |
| `fwgsl_ast_lowering`      | `moonbit/ast_lowering`    |
| `fwgsl_typechecker`       | `moonbit/typechecker`     |
| `fwgsl_semantic`          | `moonbit/semantic`        |
| `fwgsl_hir`               | `moonbit/hir`             |
| `fwgsl_mir`               | `moonbit/mir`             |
| `fwgsl_wgsl_codegen`      | `moonbit/wgsl_codegen`    |
| `fwgsl_ide`               | `moonbit/ide`             |
| `fwgsl_language_server`   | `moonbit/language_server` |
| `fwgsl_cli`               | `moonbit/cli` (+ `cmd/main`) |
| `fwgsl_wasm`              | `moonbit/wasm` (no wasm-bindgen layer) |
| `fwgsl_integration_tests` | `moonbit/integration_tests` |

`fwgsl_allocator` has no MoonBit equivalent: the arena pattern it
provides is unnecessary under MoonBit's garbage collector.

## Status

| Layer            | Status   | Notes |
| ---------------- | -------- | ----- |
| span             | Done     | Span / Atom / Spanned tested |
| diagnostics      | Done     | Builder API + miette-shaped renderer |
| syntax           | Done     | All `SyntaxKind` variants + `keyword_from_str` |
| cst              | Done     | Minimal GreenNode / GreenToken pair |
| ast              | Done     | Lit, Type, Pat, Expr, DoStmt, ConDecl, Decl, Program |
| parser           | Done     | Lexer + layout resolver + recursive-descent / Pratt parser |
| typechecker      | Done     | Ty / Scheme / TypeEnv / Substitution / InferEngine / ty_to_wgsl |
| semantic         | Done     | Full builtin prelude + Algorithm W inference |
| ast_lowering     | Done     | Full AST -> HIR lowering with finalisation |
| hir              | Done     | HirProgram + per-variant `ty()` accessor |
| mir              | Done     | Data model + HIR -> MIR lowering |
| wgsl_codegen     | Done     | Tree-walk emitter; vec / mat constructor inference |
| ide              | Partial  | Tokens off lexer; hover / goto are stubs pending name resolution |
| language_server  | Partial  | Document store + LSP-shaped diagnostics; JSON-RPC transport TBD |
| wasm             | Partial  | Pure functions returning JSON; needs host glue to mount as wasm-bindgen replacement |
| cli              | Partial  | Library drives full pipeline; binary main waits on stable `@x/fs` + argv API |
| integration_tests | Done    | Full pipeline asserts WGSL output for canonical examples |

## Building

Requires the MoonBit toolchain (`moon` and `moonc`). With `mise`:

```sh
mise run mb-check    # type check
mise run mb-build    # build everything
mise run mb-test     # run tests
mise run mb-fmt      # format
```

CI runs the same `moon check` + `moon test` steps as a required job on
every PR (`.github/workflows/ci.yml` -> `moonbit` job).
