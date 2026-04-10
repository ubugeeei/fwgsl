# Language surface upgrade (incremental)

## Added syntax

- Pipeline operator: `x |> f a b`.
- Type declarations:
  - `alias Name = ...`
  - `newtype Name = ...`
  - `type Name = { field : Ty, ... }` (lowered conservatively to a single-constructor record ADT)
- Angle-bracket type args in type grammar: `Vec<3, F32>`, `Array<Vec<4, F32>, 8>`.
- Binding declarations:
  - `@group(N) @binding(M) uniform name : T`
  - `@group(N) @binding(M) storage name : T`
  - `@group(N) @binding(M) storage(read_write) name : T`
- Index expressions: `m[0]`, `m[0][1]`.

## Desugaring rules

- Pipeline is desugared in the parser:
  - `x |> f` => `f x`
  - `x |> f a b` => `f x a b` (lhs inserted as first argument of rhs application spine)
- `newtype T = U` is lowered to `data T = T U`.
- `type T = { ... }` is lowered to a single-constructor record `data` shape.

## Type rules (current conservative implementation)

- Existing HM inference pipeline is preserved.
- Field access continues to support record-like fields and vector swizzles.
- Indexing currently infers through existing expression inference with conservative fallback typing.

## WGSL lowering

- No semantic lowering model was made more implicit.
- Existing field and index MIR/WGSL lowering paths are used.
- Resource declarations are now parsed and available in the AST for follow-up reflection/bindgen wiring.

## Deferred

- Writable swizzle assignment.
- Advanced resource semantic validation/reflection emission beyond AST-level declaration support.
- Expanded effect/type-class machinery beyond current compiler model.

## Showcase

- See `examples/enlightenment-stars.shadml` for a creative end-to-end sketch combining:
  - `alias`, `newtype`, record `type`, and `data` ADTs
  - binding declarations (`@group/@binding`)
  - pipeline `|>`
  - record field access + vector swizzles
  - matrix indexing (`m[0][1]`)

