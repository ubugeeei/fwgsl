# fwgsl (MoonBit)

This directory contains the MoonBit implementation of the `fwgsl`
compiler. It is the source of truth — the original Rust workspace has
been removed (#4). Every compiler stage lives as its own MoonBit
package below.

## Layout

| Package                     | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `moonbit/span`              | Source spans, atoms, `Spanned[T]`                      |
| `moonbit/diagnostics`       | Builder-style structured diagnostics + miette-style source-snippet renderer |
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
| `moonbit/formatter`         | Token-stream formatter shared by `fwgsl fmt` and `textDocument/formatting` |
| `moonbit/linter`            | AST-walking lint rules (`unused-let`, `unused-where`, `redundant-wildcard`, `shadowed-binding`) |
| `moonbit/ide`               | IDE-facing analyses (type-aware hover, env-aware completions, inlay hints, code actions, document symbols, semantic tokens, goto-definition) shared with the LSP and playground |
| `moonbit/language_server`   | LSP server (document store, publishDiagnostics, hover, completions, inlay hints, code actions, document symbols, semantic tokens, formatting) |
| `moonbit/wasm`              | wasm-gc entry points (`wasm_check` / `wasm_compile`) for the playground; emits real WGSL and includes lint warnings |
| `moonbit/cli`               | CLI library + binary at `moonbit/cli/cmd/main` (`compile` / `check` / `fmt` / `version` / `help`, with `--source` / `--stdin` inputs) |
| `moonbit/integration_tests` | End-to-end pipeline tests asserting WGSL output + diagnostic accuracy |

## Building

Requires the MoonBit toolchain (`moon` and `moonc`):

```sh
moon check     # type check
moon build     # build everything
moon test      # run tests
moon fmt       # format
```

`just` wrappers exist for the same commands: `just check`,
`just build`, `just test`, `just fmt`. The dev shell that bundles
MoonBit + Node + `just` is defined in the repo-root `flake.nix`
(`nix develop`).

CI runs `moon check` + `moon test` as a required job on every PR
(`.github/workflows/ci.yml` → `moonbit` job, `continue-on-error` off).
