# Examples

This directory contains example shadml programs demonstrating language features.

## Quick Use

Compile an example:

```sh
cargo run -p shadml_cli -- compile examples/hello.shadml
```

Type-check a richer example:

```sh
cargo run -p shadml_cli -- check examples/lines.shadml
```

Compile with a feature flag:

```sh
cargo run -p shadml_cli -- compile examples/conditional-compilation.shadml --feature debug
```

## Examples

| File | Purpose | Command |
|------|---------|---------|
| `hello.shadml` | Minimal arithmetic and `let` | `compile` |
| `where.shadml` | Function-level `where` bindings | `compile` |
| `adt-match.shadml` | ADT constructors and pattern matching | `compile` |
| `compute-basic.shadml` | Basic `@compute` entry point | `compile` |
| `vec-literals.shadml` | Vector literals and swizzles | `compile` |
| `generic-data.shadml` | Generic and phantom data declarations | `check` |
| `dependent-types.shadml` | Type-level naturals in vector/matrix signatures | `check` |
| `tensor-aliases.shadml` | `Tensor` / `Vector` / `Matrix` / `Scalar` aliases | `check` |
| `option-result.shadml` | Builtin `Option` / `Result` constructors and matching | `check` |
| `traits.shadml` | Trait declarations, impl blocks, operator overloading | `compile` |
| `method-syntax.shadml` | Method-call syntax sugar (`x.method y`) | `compile` |
| `bitfield.shadml` | Bitfield construction, field access, functional update | `compile` |
| `typed-bitfield.shadml` | Bitfields with typed enum fields | `compile` |
| `bitwise-ops.shadml` | Bitwise AND, XOR, shift, NOT operators | `compile` |
| `enum-discriminants.shadml` | Enums with explicit discriminant values | `compile` |
| `record-update.shadml` | Record update syntax: `p { x = p.x + dx }` | `compile` |
| `const-promotion.shadml` | Module-level `const` declarations | `compile` |
| `when-guards.shadml` | When-guards on match arms | `compile` |
| `doc-comments.shadml` | Documentation comment support | `check` |
| `conditional-compilation.shadml` | `when cfg.x` / `else when` / `else` | `check --feature debug` |
| `lines.shadml` | Comprehensive GPU line renderer (vertex/fragment/compute) | `check` |
| `enlightenment-stars.shadml` | Animated compute sketch with new surface syntax | `check` |

## Shadorial

The `shadorial/` subdirectory contains shader tutorial examples (chapters 01-19) that work in the web playground with live WebGPU preview.

## Notes

- Examples using `compile` produce valid WGSL output.
- Examples using `check` type-check successfully but may exercise features (like `Option`/`Result`) that don't yet have full WGSL lowering.
- The `lines.shadml` example demonstrates nearly every language feature in a single real-world shader.
