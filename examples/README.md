# Examples

This directory contains example fwgsl programs demonstrating language features.

## Quick Use

Compile an example:

```sh
cargo run -p fwgsl_cli -- compile examples/hello.fwgsl
```

Type-check a richer example:

```sh
cargo run -p fwgsl_cli -- check examples/lines.fwgsl
```

Compile with a feature flag:

```sh
cargo run -p fwgsl_cli -- compile examples/conditional-compilation.fwgsl --feature debug
```

## Examples

| File | Purpose | Command |
|------|---------|---------|
| `hello.fwgsl` | Minimal arithmetic and `let` | `compile` |
| `where.fwgsl` | Function-level `where` bindings | `compile` |
| `adt-match.fwgsl` | ADT constructors and pattern matching | `compile` |
| `compute-basic.fwgsl` | Basic `@compute` entry point | `compile` |
| `vec-literals.fwgsl` | Vector literals and swizzles | `compile` |
| `generic-data.fwgsl` | Generic and phantom data declarations | `check` |
| `dependent-types.fwgsl` | Type-level naturals in vector/matrix signatures | `check` |
| `tensor-aliases.fwgsl` | `Tensor` / `Vector` / `Matrix` / `Scalar` aliases | `check` |
| `option-result.fwgsl` | Builtin `Option` / `Result` constructors and matching | `check` |
| `traits.fwgsl` | Trait declarations, impl blocks, operator overloading | `compile` |
| `method-syntax.fwgsl` | Method-call syntax sugar (`x.method y`) | `compile` |
| `bitfield.fwgsl` | Bitfield construction, field access, functional update | `compile` |
| `typed-bitfield.fwgsl` | Bitfields with typed enum fields | `compile` |
| `bitwise-ops.fwgsl` | Bitwise AND, XOR, shift, NOT operators | `compile` |
| `enum-discriminants.fwgsl` | Enums with explicit discriminant values | `compile` |
| `record-update.fwgsl` | Record update syntax: `p { x = p.x + dx }` | `compile` |
| `const-promotion.fwgsl` | Module-level `const` declarations | `compile` |
| `when-guards.fwgsl` | When-guards on match arms | `compile` |
| `doc-comments.fwgsl` | Documentation comment support | `check` |
| `conditional-compilation.fwgsl` | `when cfg.x` / `else when` / `else` | `check --feature debug` |
| `lines.fwgsl` | Comprehensive GPU line renderer (vertex/fragment/compute) | `check` |
| `enlightenment-stars.fwgsl` | Animated compute sketch with new surface syntax | `check` |

## Shadorial

The `shadorial/` subdirectory contains shader tutorial examples (chapters 01-19) that work in the web playground with live WebGPU preview.

## Notes

- Examples using `compile` produce valid WGSL output.
- Examples using `check` type-check successfully but may exercise features (like `Option`/`Result`) that don't yet have full WGSL lowering.
- The `lines.fwgsl` example demonstrates nearly every language feature in a single real-world shader.
