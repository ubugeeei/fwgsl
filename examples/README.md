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

## Canonical Examples

| File | Purpose | Suggested command |
|------|---------|-------------------|
| `hello.fwgsl` | Minimal arithmetic and `let` | `compile` |
| `where.fwgsl` | Function-level `where` bindings | `compile` |
| `adt-match.fwgsl` | ADT constructors and pattern matching | `compile` |
| `compute-basic.fwgsl` | Basic `@compute` entry point | `compile` |
| `generic-data.fwgsl` | Generic and phantom data declarations | `check` |
| `dependent-types.fwgsl` | Type-level naturals in array/vector-shaped signatures | `check` |

## Notes

- The current parser still has edge cases around standalone type signatures followed by definitions on the next line. The root examples avoid that shape unless the file is intended for `check`.
- `shadorial/` is useful for playground exploration and visual experimentation, but some samples are ahead of the currently implemented backend and are not guaranteed to lower fully through every compiler stage yet.
