# fwgsl

A pure functional language that compiles to [WGSL](https://www.w3.org/TR/WGSL/) (WebGPU Shading Language).

fwgsl brings ML/Haskell-style programming to the GPU — static typing, Hindley-Milner inference, algebraic data types, pattern matching, and function composition — all compiled down to valid WGSL that runs on WebGPU.

## Examples

### Hello World

```haskell
double : I32 -> I32
double x = x + x

main : I32 -> I32
main x =
  let y = double x
  in y + 1
```

### Algebraic Data Types and Pattern Matching

```haskell
data Color = Red | Green | Blue

show : Color -> I32
show c = match c
  | Red   -> 0
  | Green -> 1
  | Blue  -> 2
```

### Lambda Expressions and Function Composition

```haskell
apply : (I32 -> I32) -> I32 -> I32
apply f x = f x

inc : I32 -> I32
inc = \x -> x + 1

doubleInc : I32 -> I32
doubleInc = double . inc
```

### Shader Entry Points

```haskell
@vertex
vs_main : VertexInput -> Vec4F
vs_main input = input.position

@fragment
fs_main : FragmentInput -> Vec4F
fs_main input = vec4 1.0 0.0 0.0 1.0

@compute @workgroup_size(64, 1, 1)
cs_main : ComputeInput -> ()
cs_main input =
  let idx = input.global_invocation_id.x
  in store output idx (load input_buf idx)
```

### GPU Particle Simulation

A more complete example showing ADTs, pattern matching, and compute shaders working together:

```haskell
data ParticleState
  = Active { position : Vec3F, velocity : Vec3F, life : F32 }
  | Dead

stepParticle : F32 -> ParticleState -> ParticleState
stepParticle dt particle =
  match particle
    | Active { position, velocity, life } ->
      if life - dt <= 0.0
        then Dead
        else Active
          { position = position + velocity * dt
          , velocity = velocity + gravity * dt
          , life     = life - dt
          }
    | Dead -> Dead

@compute @workgroup_size(64, 1, 1)
main : ComputeInput -> ()
main input =
  let idx    = input.global_invocation_id.x
      state  = load particles idx
      state' = stepParticle deltaTime state
  in  store particles idx state'
```

Compiles to:

```wgsl
struct ParticleState {
  tag: u32,
  position: vec3<f32>,
  velocity: vec3<f32>,
  life: f32,
}

fn stepParticle(dt: f32, particle: ParticleState) -> ParticleState {
  if (particle.tag == 0u) {
    let position = particle.position;
    let velocity = particle.velocity;
    let life = particle.life;
    if (life - dt <= 0.0) {
      return ParticleState(1u, vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
    } else {
      return ParticleState(0u, position + velocity * dt,
        velocity + vec3<f32>(0.0, -9.8, 0.0) * dt, life - dt);
    }
  } else {
    return particle;
  }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let state = particles[idx];
  let state_prime = stepParticle(deltaTime, state);
  particles[idx] = state_prime;
}
```

## Language Features

### Type System

- **Hindley-Milner type inference** — full type inference with let-generalization; type annotations optional but supported
- **Static typing** — all types resolved at compile time, no runtime type checks
- **Dependent-type dimensions** — `Vec 3 F32` compiles to `vec3<f32>`, `Mat 4 4 F32` to `mat4x4<f32>`, with compile-time dimension validation
- **Type classes** — Functor, Applicative, Monad with instance resolution (no special symbols like `<$>`)
- **Type signatures** — `add : I32 -> I32 -> I32` using `:` (not `::`)

### Functional Programming

- **Algebraic data types** — `data Color = Red | Green | Blue` compiled to tagged WGSL structs
- **Pattern matching** — `match` with pipe-delimited arms, compiled to decision trees via Maranget's algorithm
- **Lambda expressions** — `\x y -> x + y`
- **Let bindings** — `let x = 1 in x + 2` with block-style `let ... in ...`
- **Function composition** — `normalize . getVelocity` with the dot operator
- **Currying and partial application** — all functions are curried
- **Operators as functions** — `(+)`, `(*)` can be passed as first-class values
- **Backtick infix** — `` x `clamp` lo `` turns any function into an infix operator
- **If-then-else** — `if x == 0 then 1 else 2` as expressions

### Syntax

- **Indentation-sensitive layout** — Haskell 2010-style layout rules with virtual braces and semicolons
- **Entry point attributes** — `@vertex`, `@fragment`, `@compute @workgroup_size(64, 1, 1)` map directly to WGSL shader stages
- **Expression-oriented** — everything is an expression, including `if`, `let`, and `match`
- **Pratt parsing** — operator precedence: `.` > application > `*`/`/` > `+`/`-` > comparison > `&&` > `||` > backtick > `$`

### Compiler Transforms

- **Monomorphization** — polymorphic functions specialized at each call site for WGSL compatibility
- **Defunctionalization** — higher-order functions lowered to tagged structs with switch dispatch
- **Tail-call elimination** — tail recursion converted to loops (WGSL forbids recursion)
- **ADT lowering** — algebraic data types compiled to flat structs with tag fields

## Architecture

Written in Rust, inspired by [Oxc](https://github.com/oxc-project/oxc)'s arena-allocated design for fast compilation.

```
Source Text
    |
  Lexer + Layout Resolver
    |
  Parser (recursive descent + Pratt)
    |
  [AST] ── Semantic Analysis (name resolution + HM inference)
    |
  [HIR] ── Desugared, type-annotated
    |
  [MIR] ── Monomorphized, first-order, WGSL-shaped
    |
  WGSL Text
```

### Crates

| Crate | Purpose |
|-------|---------|
| `fwgsl_allocator` | Arena allocation (bumpalo wrapper) |
| `fwgsl_span` | Source spans, atoms, source types |
| `fwgsl_diagnostics` | Error/warning reporting with labels (miette-based) |
| `fwgsl_syntax` | SyntaxKind enum for all tokens and node kinds |
| `fwgsl_parser` | Hand-written lexer, Haskell-style layout resolver, recursive descent + Pratt parser |
| `fwgsl_typechecker` | Type representation, HM inference engine, union-find unification |
| `fwgsl_semantic` | Name resolution, scope analysis, type checking with constructor registration |
| `fwgsl_hir` | High-level IR — desugared, type-annotated, still functional |
| `fwgsl_mir` | Mid-level IR — monomorphized, first-order, imperative, WGSL-shaped |
| `fwgsl_wgsl_codegen` | MIR to WGSL text emission with struct ordering and name mangling |
| `fwgsl_language_server` | LSP server (tower-lsp) — diagnostics, hover, completion, goto-definition, semantic tokens |
| `fwgsl_integration_tests` | End-to-end pipeline tests (256+ tests) |
| `fwgsl_cli` | Command-line interface |
| `fwgsl_wasm` | WASM target for web playground |

## Getting Started

Requires [Rust](https://rustup.rs/) and [mise](https://mise.jdx.dev/).

```sh
# Build
mise run build

# Run tests
mise run test

# Compile a .fwgsl file
mise run cli -- compile examples/hello.fwgsl

# Lint and format
mise run lint
mise run fmt
```

## Tooling

### Language Server (LSP)

The `fwgsl-lsp` binary provides IDE support via the Language Server Protocol:

- **Diagnostics** — parse and type errors reported on open/change
- **Hover** — shows inferred types for identifiers and keyword descriptions
- **Completion** — keywords, built-in types (`I32`, `F32`, `Vec3F`, ...), WGSL builtins, and document identifiers
- **Go to Definition** — jump to function and type definitions
- **Semantic Tokens** — syntax-aware highlighting for keywords, types, operators, strings, numbers, and comments

### Web Playground

A browser-based playground with Monaco editor, WGSL output pane, and WebGPU live preview.

```sh
mise run wasm       # build WASM module
mise run playground # serve playground
mise run dev        # quick dev server (no WASM rebuild)
```

### Linter and Formatter

Planned: CST-based formatter (preserves comments, canonical indentation) and modular lint rules (unused variables, incomplete patterns, WGSL-incompatible recursion).

## WGSL Constraints

WGSL imposes severe restrictions that the compiler must bridge:

| Constraint | How fwgsl handles it |
|-----------|---------------------|
| No recursion | Tail-call elimination to loops; general recursion detected and rejected |
| No dynamic allocation | All data is fixed-size; arena allocation is compile-time only |
| No first-class functions | Defunctionalization: closures become tagged structs + dispatch |
| No generics | Monomorphization: polymorphic code specialized per call site |
| No ADTs | Tagged structs with flat field layout |

## License

MIT
