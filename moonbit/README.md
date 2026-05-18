# fwgsl (MoonBit)

This directory contains the MoonBit rewrite of the `fwgsl` compiler. The Rust
implementation under `crates/` is the source of truth until the MoonBit port
reaches functional parity. Tracked by [issue #4](https://github.com/ubugeeei/fwgsl/issues/4).

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
| `fwgsl_cli`               | `moonbit/cli`             |
| `fwgsl_wasm`              | replaced by direct MoonBit WASM build |
| `fwgsl_integration_tests` | `moonbit/integration_tests` |

`fwgsl_allocator` has no MoonBit equivalent: the arena pattern it provides is
unnecessary under MoonBit's garbage collector.

## Building

Requires the MoonBit toolchain (`moon` and `moonc`). With `mise`:

```sh
mise run mb-check    # type check
mise run mb-build    # build everything
mise run mb-test     # run tests
```
