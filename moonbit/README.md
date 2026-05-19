# fwgsl (MoonBit)

This directory contains the MoonBit implementation of the `fwgsl`
compiler. It is the source of truth — the original Rust workspace has
been removed (#4). Every compiler stage lives as its own MoonBit
package below.

## Layout

| Package                     | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `moonbit/span`              | Source spans, atoms, `Spanned[T]`                      |
| `moonbit/diagnostics`       | Builder-style structured diagnostics + renderer        |
| `moonbit/syntax`            | `SyntaxKind` enum, keyword resolver, classification    |
| `moonbit/cst`               | Concrete syntax tree nodes (GreenToken / GreenNode)    |
| `moonbit/ast`               | AST types (`Lit`, `Type`, `Pat`, `Expr`, `Decl`, ...)  |
| `moonbit/parser`            | Lexer + layout resolver + recursive descent / Pratt parser |
| `moonbit/typechecker`       | `Ty`, `Scheme`, `Substitution`, `InferEngine`, `WgslType`, `ty_to_wgsl` |
| `moonbit/semantic`          | Full builtin prelude + Algorithm-W inference over the AST |
| `moonbit/ast_lowering`      | AST -> HIR lowering with Hindley-Milner inference      |
| `moonbit/hir`               | Desugared typed high-level IR                          |
| `moonbit/mir`               | WGSL-shaped IR + the HIR -> MIR lowering pass          |
| `moonbit/wgsl_codegen`      | Tree-walk emitter that turns MIR into WGSL text        |
| `moonbit/ide`               | IDE-facing analyses (completions, hover, semantic tokens) shared with the LSP and playground |
| `moonbit/language_server`   | LSP server (document store, publishDiagnostics, hover) |
| `moonbit/wasm`              | wasm-gc entry points (`wasm_check` / `wasm_compile`) for the playground |
| `moonbit/cli`               | CLI library; `moonbit/cli/cmd/main` is the binary      |
| `moonbit/integration_tests` | End-to-end pipeline tests asserting WGSL output        |

## Building

Requires the MoonBit toolchain (`moon` and `moonc`):

```sh
moon check     # type check
moon build     # build everything
moon test      # run tests
moon fmt       # format
```

`mise` wrappers exist for the same commands: `mise run check`,
`mise run build`, `mise run test`, `mise run fmt`.

CI runs `moon check` + `moon test` as a required job on every PR
(`.github/workflows/ci.yml` → `moonbit` job, `continue-on-error` off).
