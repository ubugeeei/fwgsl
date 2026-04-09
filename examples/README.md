# Examples

This directory is split into two kinds of examples:

- Root-level `.fwgsl` files are small canonical examples for the current compiler.
- [`shadorial/`](/Users/nishimura/projects/oss/ubugeeei/fwgsl/examples/shadorial) contains larger shader sketches and tutorial-style samples.

## Quick Use

Compile a known end-to-end example:

```sh
cargo run -p fwgsl_cli -- compile examples/hello.fwgsl
```

Type-check a richer type-system example:

```sh
cargo run -p fwgsl_cli -- check examples/generic-data.fwgsl
```

Check the newer builtins and tensor-oriented surface:

```sh
cargo run -p fwgsl_cli -- check examples/option-result.fwgsl
cargo run -p fwgsl_cli -- check examples/prelude-utils.fwgsl
```

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
| `record-update.fwgsl` | Record update syntax: `p { x = p.x + dx }` | `compile` |
| `modules/Main.fwgsl` | Multi-file module system (imports `Particle`, `Math.Vec`) | `compile` |
| `enlightenment-stars.fwgsl` | Creative line/star/enlightenment animated compute sketch using new surface syntax | `check` |

## Notes

- The current parser still has edge cases around standalone type signatures followed by definitions on the next line. The root examples avoid that shape unless the file is intended for `check`.
- `Option`, `Result`, and the higher-order prelude utilities are currently best explored with `check`; they are registered in the type environment before the WGSL backend grows dedicated lowering support for them.
- `shadorial/` is useful for playground exploration and visual experimentation, but some samples are ahead of the currently implemented backend and are not guaranteed to lower fully through every compiler stage yet.

- New surface-syntax example: `examples/language-surface-upgrade.fwgsl`
