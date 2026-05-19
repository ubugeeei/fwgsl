# Examples

This directory is split into two kinds of examples:

- Root-level `.fwgsl` files are small canonical examples for the current compiler.
- [`shadorial/`](./shadorial) contains larger shader sketches and tutorial-style samples.

## Quick Use

The library's `@cli.compile_source` / `@cli.check_source` entry points drive every example. From a MoonBit shell, you can run a sample inline:

```sh
moon test ubugeeei/fwgsl/integration_tests   # exercises the canonical programs
```

For an interactive feel, paste a sample into the [playground](../playground/) and watch the WGSL output update live.

## Canonical Examples

| File | Purpose | Suggested command |
|------|---------|-------------------|
| `hello.fwgsl` | Minimal arithmetic and `let` | `compile` |
| `where.fwgsl` | Function-level `where` bindings | `compile` |
| `adt-match.fwgsl` | ADT constructors and pattern matching | `compile` |
| `compute-basic.fwgsl` | Basic `@compute` entry point | `compile` |
| `vec-literals.fwgsl` | Vector literals and swizzles | `compile` |
| `generic-data.fwgsl` | Generic and phantom data declarations | `check` |
| `dependent-types.fwgsl` | Type-level naturals in tensor/vector-shaped signatures | `check` |
| `tensor-aliases.fwgsl` | `Tensor` / `Vector` / `Matrix` / `Scalar` plus short aliases | `check` |
| `option-result.fwgsl` | Builtin `Option` / `Result` constructors and matching | `check` |
| `prelude-utils.fwgsl` | `$map`, `$fold`, `$zip`, `$flatMap`, `$all`, `$any` | `check` |

## Notes

- The current parser still has edge cases around standalone type signatures followed by definitions on the next line. The root examples avoid that shape unless the file is intended for `check`.
- `Option`, `Result`, and the higher-order prelude utilities are currently best explored with `check`; they are registered in the type environment before the WGSL backend grows dedicated lowering support for them.
- `shadorial/` is useful for playground exploration and visual experimentation, but some samples are ahead of the currently implemented backend and are not guaranteed to lower fully through every compiler stage yet.
