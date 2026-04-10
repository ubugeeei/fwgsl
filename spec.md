# fwgsl Language Specification

**Version:** 0.1.0
**Status:** Working draft

fwgsl is a purely functional language that compiles to WGSL (WebGPU Shading Language). It provides Haskell-inspired syntax with indentation-sensitive layout, Hindley-Milner type inference, algebraic data types, pattern matching, and traits — all targeting the GPU via WGSL code generation.

---

## Table of Contents

1. [Lexical Structure](#1-lexical-structure)
2. [Layout Rules](#2-layout-rules)
3. [Types](#3-types)
4. [Declarations](#4-declarations)
5. [Expressions](#5-expressions)
6. [Patterns](#6-patterns)
7. [Operators](#7-operators)
8. [Type System](#8-type-system)
9. [Traits and Implementations](#9-traits-and-implementations)
10. [Module System](#10-module-system)
11. [Conditional Compilation](#11-conditional-compilation)
12. [Bitfields](#12-bitfields)
13. [Entry Points and GPU Resources](#13-entry-points-and-gpu-resources)
14. [Prelude and Builtins](#14-prelude-and-builtins)
15. [Compilation Pipeline](#15-compilation-pipeline)
16. [WGSL Code Generation](#16-wgsl-code-generation)
17. [Tooling](#17-tooling)

---

## 1. Lexical Structure

### 1.1 Comments

```
-- Line comment (to end of line)
{- Block comment (may be nested) -}
```

### 1.2 Identifiers

| Form | Description | Examples |
|------|-------------|---------|
| `Ident` | Lower-case or underscore-leading | `foo`, `_tmp`, `myVar` |
| `UpperIdent` | Starts with uppercase letter | `Vec`, `MyStruct`, `Some` |
| `Operator` | Symbolic operator in parentheses | `(+)`, `(==)`, `(%)` |

### 1.3 Keywords

```
module  where   import  data    alias   extern  resource
trait   impl    let     in      case    of      match
if      then    else    do      forall  infixl  infixr
infix   deriving  bitfield  const  loop  as  when  cfg
```

### 1.4 Integer Literals

| Form | Description | Examples |
|------|-------------|---------|
| Decimal | Unsuffixed → I32 | `42`, `0`, `1000` |
| Decimal + `u` | U32 suffix | `42u`, `0u` |
| Decimal + `i` | Explicit I32 suffix | `42i` |
| Hexadecimal | `0x` or `0X` prefix | `0xFF`, `0xFFu` |
| Octal | `0o` or `0O` prefix | `0o77`, `0o10u` |
| Binary | `0b` or `0B` prefix | `0b1010`, `0b1010u` |

The `u`/`i` suffix is only recognized when **not** followed by an identifier-continue character.

### 1.5 Float Literals

```
3.14    0.5    1.0e10    2.5E-3    0.0
```

Floats always contain a decimal point or exponent. There is no `f` suffix.

### 1.6 Negative Literal Atoms

A minus sign (`-`) immediately followed by a digit (no intervening whitespace) is parsed as a **negative literal atom** in function application position. This allows:

```
vec2 -0.35 0.5    -- parsed as vec2(-0.35, 0.5)
```

When there is whitespace before the `-`, it is parsed as the binary subtraction operator:

```
x - 0.5           -- subtraction
```

### 1.7 String and Character Literals

String literals are delimited by `"..."` and character literals by `'...'`. These are parsed by the lexer but have limited use in the current WGSL target (WGSL has no string type).

### 1.8 Punctuation and Operators

| Token | Symbol | Token | Symbol |
|-------|--------|-------|--------|
| `(` `)` | Parentheses | `[` `]` | Brackets |
| `{` `}` | Braces | `,` | Comma |
| `;` | Semicolon | `:` | Colon |
| `::` | Double colon | `.` | Dot |
| `..` | Range | `->` | Arrow |
| `=>` | Fat arrow | `\` | Backslash (lambda) |
| `@` | Attribute prefix | `\|` | Pipe (match arms) |
| `=` | Equals | `_` | Wildcard |
| `\|>` | Pipeline forward | `$` | Low-precedence apply |
| `` ` `` | Backtick (infix) | `!` | Boolean not |
| `<-` | Left arrow | | |

**Arithmetic:** `+`, `-`, `*`, `/`, `%`

**Comparison:** `==`, `/=` (not equal), `<`, `>`, `<=`, `>=`

**Logical:** `&&`, `||`

**Bitwise:** `&` (AND), `^` (XOR), `~` (NOT prefix), `<<` (shift left). Bitwise OR uses the `bor` function; shift right uses `>>` (two adjacent `>`) or the `shr` function.

---

## 2. Layout Rules

fwgsl uses **Haskell 2010-style indentation-sensitive layout**. The layout resolver inserts virtual tokens into the token stream:

- `LayoutBraceOpen` — opens an indentation context
- `LayoutSemicolon` — separates declarations/bindings at the same indentation level
- `LayoutBraceClose` — closes an indentation context

### 2.1 Layout Keywords

After `where`, `let`, `of`, and `do`, if the next non-trivia token is not an explicit `{`, the resolver inserts `LayoutBraceOpen` at that token's column and pushes the column onto the indent stack.

### 2.2 Indentation Rules

On each newline, the resolver compares the next token's column to the top of the indent stack:

- **Equal column** → insert `LayoutSemicolon` (new declaration/binding)
- **Lesser column** → pop the stack and insert `LayoutBraceClose`, repeat
- **Greater column** → continuation of the current declaration

### 2.3 Explicit Braces

Within explicit `{ }` braces (records, bitfield declarations), virtual layout tokens are suppressed. The resolver tracks `explicit_brace_depth` and only generates layout tokens at depth 0.

---

## 3. Types

### 3.1 Primitive Types

| fwgsl Type | WGSL Type | Description |
|------------|-----------|-------------|
| `I32` | `i32` | Signed 32-bit integer |
| `U32` | `u32` | Unsigned 32-bit integer |
| `F32` | `f32` | 32-bit floating point |
| `Bool` | `bool` | Boolean |
| `()` | (void) | Unit type |

### 3.2 Vector Types

```
Vec<N, T>       -- e.g. Vec<2, F32> → vec2<f32>
```

`N` is a natural number literal (2, 3, or 4). `T` is a scalar type.

Aliases: `Vector` is accepted as a synonym for `Vec`.

### 3.3 Matrix Types

```
Mat<R, C, T>    -- e.g. Mat<4, 4, F32> → mat4x4<f32>
```

Alias: `Matrix` is accepted as a synonym for `Mat`.

### 3.4 Array Types

```
Array<T, N>     -- Fixed-size array: array<T, N>
Array<T>        -- Runtime-sized (unsized) array: array<T>
```

Surface `Array<T, N>` is normalized internally to `Tensor T N` (element type first, dimension second).

Aliases: `Tensor` and `Ten` are accepted as synonyms for `Array`.

### 3.5 Function Types

```
A -> B          -- Function from A to B
A -> B -> C     -- Curried: A -> (B -> C)
```

Arrow types are right-associative.

### 3.6 Tuple Types

```
(A, B)          -- Pair
(A, B, C)       -- Triple
```

Tuple types exist in the surface language and type system but are **desugared before codegen**. WGSL has no tuple type.

Tuple function parameters are desugared to curried parameters:

```
f : (A, B) -> R     -- desugared to: f : A -> B -> R
f (a, b) = ...      -- desugared to: f a b = ...
```

Tuple call-site arguments are flattened:

```
f (x, y)            -- desugared to: f x y  (i.e. App(App(f, x), y))
```

### 3.7 Type Application

```
Maybe I32           -- Type constructor applied to argument
Vec 3 F32           -- Multi-argument application (curried)
```

### 3.8 Type Aliases

```
alias Vec4f = Vec<4, F32>
alias MyArray = Array<F32, 10>
```

Type aliases are expanded during semantic analysis. The `alias` keyword is the only way to declare type aliases (`type` and `newtype` are not supported).

### 3.9 Type Variables

Lowercase identifiers in type positions are type variables, used for polymorphism:

```
extern sin : a -> a             -- polymorphic
identity : a -> a
identity x = x
```

### 3.10 Resource Wrapper Types

```
Uniform<T>                      -- Uniform buffer
Storage<ReadWrite, T>           -- Read-write storage buffer
Storage<Read, T>                -- Read-only storage buffer (StorageRead)
```

These are used in `extern resource` declarations. The wrapper is unwrapped to the inner type `T` in the type environment.

### 3.11 Dependent Dimensions (Nat)

Natural number literals in type position (`2`, `3`, `4`) are `Ty::Nat` values used for vector/matrix/array dimensions:

```
Vec<3, F32>     -- Nat(3) applied to Vec
```

---

## 4. Declarations

### 4.1 Type Signatures

```
name : Type
```

Type signatures are optional but recommended. They precede the corresponding function definition:

```
add : I32 -> I32 -> I32
add x y = x + y
```

### 4.2 Function Declarations

```
name params = body
```

Functions are defined with pattern parameters on the left of `=`:

```
length v = sqrt (dot v v)
```

#### 4.2.1 Guard Clauses

Functions can use guard clauses instead of a single body:

```
abs x
  | x < 0    = 0 - x
  | otherwise = x
```

Guards are delimited by `|` and tested top-to-bottom. `otherwise` is a conventional name (it must be a truthy value).

#### 4.2.2 Where Clauses

Local bindings can be introduced with `where`:

```
circleArea r = pi * r * r
  where pi = 3.14159
```

### 4.3 Data Type Declarations

The `data` keyword declares algebraic data types. It supports several forms:

#### 4.3.1 Records (Structs)

```
data Point = Point {
  x : F32,
  y : F32,
}
```

Record fields may carry attributes:

```
data VertexOutput = VertexOutput {
  @builtin(position) clip_position : Vec<4, F32>,
  @location(0)       color         : Vec<4, F32>,
}
```

#### 4.3.2 Pure Enums

```
data Color = Red | Green | Blue
```

Pure enums (constructors with no fields) compile to `u32` values in WGSL.

#### 4.3.3 Explicit Discriminant Values

```
data CapType = NoCap = 0 | Arrow = 1 | Round = 2 | Butt = 3
```

If discriminants are omitted, constructors are assigned sequential indices starting from 0.

#### 4.3.4 Sum Types (ADTs)

```
data Shape = Circle F32 | Rect F32 F32
```

Sum types compile to structs with a `tag: u32` field plus fields for each constructor's payload.

#### 4.3.5 Wrapper Types

```
data Meters = Meters F32
```

#### 4.3.6 Parameterized Types

```
data Option a = Some a | None
data Pair a b = Pair a b
```

### 4.4 Type Alias Declarations

```
alias Name = Type
alias Name params = Type
```

Examples:

```
alias Vec4f = Vec<4, F32>
alias Pos = Vec<3, F32>
```

### 4.5 Constant Declarations

```
const NAME : TYPE = EXPR
```

Constants are module-level immutable bindings that compile to WGSL `const` declarations:

```
const MAX_LIGHTS : I32 = 16
const PI : F32 = 3.14159
```

The name may be in any case (camelCase, SCREAMING_SNAKE_CASE, etc.).

### 4.6 Bitfield Declarations

See [Section 12: Bitfields](#12-bitfields).

### 4.7 Extern Declarations

```
extern name : Type
```

Declares a built-in name with a type signature but no body. Used in the prelude for WGSL builtins:

```
extern sin : a -> a
extern vec2 : a -> a -> Vec<2, a>
```

### 4.8 Resource Declarations

```
extern resource name : WrapperType @group G @binding B
```

Declares a GPU resource binding:

```
extern resource frame : Uniform<FrameData>                    @group 0 @binding 0
extern resource particles : Storage<ReadWrite, Array<Particle>> @group 1 @binding 0
```

### 4.9 Trait Declarations

See [Section 9: Traits](#9-traits-and-implementations).

### 4.10 Impl Declarations

See [Section 9: Traits](#9-traits-and-implementations).

### 4.11 Entry Point Declarations

Entry points are function declarations preceded by stage attributes:

```
main : ComputeInput -> ()
@compute @workgroup_size(64, 1, 1)
main input = ...
```

Supported stages:

| Attribute | WGSL Stage |
|-----------|------------|
| `@compute` | `@compute` |
| `@vertex` | `@vertex` |
| `@fragment` | `@fragment` |

Additional attributes: `@workgroup_size(x, y, z)`

A type signature is required before the entry point definition.

### 4.12 Module Declarations

```
module Name.Path
```

Optional header. If absent, the module name is derived from the file path. See [Section 10: Module System](#10-module-system).

### 4.13 Import Declarations

```
import Foo
import Foo (bar, baz)
import Foo as F
import Foo.*
import Debug when cfg.debug
```

See [Section 10: Module System](#10-module-system).

---

## 5. Expressions

### 5.1 Literals

```
42              -- I32 integer
42u             -- U32 integer
42i             -- I32 (explicit suffix)
0xFF            -- Hex I32
0xFFu           -- Hex U32
3.14            -- F32 float
true / false    -- Bool (constructors, not keywords)
```

### 5.2 Variables and Constructors

```
foo             -- Variable (lowercase)
MyConstructor   -- Constructor (uppercase)
```

### 5.3 Function Application

```
f x             -- Apply f to x
f x y           -- Curried: (f x) y
f (g x)         -- Nested application
```

Application is left-associative and binds tighter than all infix operators (binding power 11).

### 5.4 Binary Operators

```
x + y
a == b
p && q
```

See [Section 7: Operators](#7-operators) for the full precedence table.

### 5.5 Unary Operators

```
-x              -- Arithmetic negation
!b              -- Boolean not
```

Both unary operators have binding power 13 (same as field access and indexing).

### 5.6 Pipeline Operator

```
x |> f          -- Desugars to: f x
x |> f |> g     -- Desugars to: g (f x)
```

The pipeline operator `|>` has the lowest precedence (binding power 1) and is left-associative.

### 5.7 Lambda Expressions

```
\x -> x + 1
\x y -> x + y
```

Lambdas are introduced with `\` (backslash) and use `->` to separate parameters from the body.

When applied immediately, lambdas are **beta-reduced** at compile time:

```
(\x -> x + 1) 5    -- becomes: let x = 5 in x + 1
```

There is no lambda variant in HIR — all lambdas are resolved at application sites.

### 5.8 Let Expressions

```
let x = 1
    y = 2
in x + y
```

Multiple bindings are separated by layout (same indentation level). The scope of each binding extends to all subsequent bindings and the body.

### 5.9 If-Then-Else

```
if cond then trueBranch else falseBranch
```

Both branches are required (expressions, not statements). Multi-line:

```
if x > 0
then x
else -x
```

### 5.10 Match Expressions

```
match expr
  | pattern1 -> body1
  | pattern2 -> body2
  | _        -> default
```

#### 5.10.1 When-Guards

Match arms support optional boolean guards:

```
match expr
  | pattern when condition -> body
  | _                      -> default
```

The guard is evaluated after the pattern matches. If the guard fails, execution falls through to the next arm.

```
classify x = match x
  | _ when x < 0.0  -> 0
  | _ when x < 1.0  -> 1
  | _                -> 2
```

Guards can be combined with any pattern kind (wildcard, variable, constructor, literal):

```
match shape
  | Circle r when r > 10.0 -> "big circle"
  | Circle r               -> "small circle"
  | _                      -> "not a circle"
```

### 5.11 Loop Expressions

Named tail-recursive loops (Scheme-style named `let`):

```
loop name (var1 = init1) (var2 = init2) in body
```

Inside the body, calling `name` with new values performs a tail-recursive jump:

```
sumTo n = loop go (acc = 0) (i = 0) in
  if i < n then go (acc + i) (i + 1) else acc
```

Compiles to WGSL `loop { ... break; }` with `var` bindings and `continue`.

### 5.12 Fold Over Range

```
foldRange start end init (\acc i -> body)
```

Folds a function over an integer range `[start, end)`, threading an accumulator. This is a compiler-recognized function (declared in the prelude) that desugars to an efficient WGSL `for`-style loop:

```
-- Sum integers 0..n
sumTo : I32 -> I32
sumTo n = foldRange 0 n 0 (\acc i -> acc + i)

-- Accumulate 20 particles
color = foldRange 0 20 (vec3 0.0 0.0 0.0) (\acc i ->
  acc + particle uv (toF32 i) time aspect)
```

The function argument can also be a named top-level function:

```
addToAcc : I32 -> I32 -> I32
addToAcc acc i = acc + i

sumNamed : I32 -> I32
sumNamed n = foldRange 0 n 0 addToAcc
```

Prelude signature: `foldRange : I32 -> I32 -> a -> (a -> I32 -> a) -> a`

### 5.13 Record Construction

```
Point { x = 1.0, y = 2.0 }
```

Named record construction. The type is determined by the constructor name (uppercase identifier before `{`).

### 5.14 Record / Bitfield Functional Update

```
point { x = 3.0 }          -- copies point, overriding x
flags { capArrow = 1 }     -- bitfield update
```

Postfix `{ field = expr, ... }` after an expression creates a copy with the specified fields overridden.

### 5.15 Field Access

```
point.x
info.thickness
```

Dot syntax for record field access. Also used for:
- **Vec swizzle**: `v.xy`, `v.xyz`, `v.rgba`, `v.x`
- **Method-call sugar**: `v.normalize` → `normalize v`

### 5.16 Vec Swizzle

Swizzle patterns use `xyzw` or `rgba` component names (1–4 characters):

```
v.x             -- scalar component
v.xy            -- vec2 from vec3/vec4
v.xyz           -- vec3 from vec4
v.rgba          -- same as v.xyzw
```

The swizzle length determines the result type:
- 1 component → scalar
- 2+ components → vector of that length

### 5.17 Method-Call Syntax Sugar

```
x.method y      -- desugars to: method x y
a.collapse      -- desugars to: collapse a
```

Priority: **swizzle** > **method call** (if name is in scope) > **struct field access**.

### 5.18 Index Access

```
arr[i]
buffer[toU32 idx]
```

### 5.19 Vec Literals

```
[1.0, 2.0, 3.0]    -- desugars to: vec3 1.0 2.0 3.0
[x, y]              -- desugars to: vec2 x y
```

### 5.20 Parenthesized Expressions

```
(x + y)
(f x)
(-expr)             -- negation (not operator section)
```

### 5.20 Backtick Infix

```
a `max` b           -- desugars to: max a b
```

Any function can be used as an infix operator by enclosing it in backticks.

### 5.21 Dollar Application

```
f $ g x             -- desugars to: f (g x)
```

The `$` operator has the lowest binding power (right-associative), enabling parenthesis-free chains.

---

## 6. Patterns

### 6.1 Wildcard Pattern

```
_               -- Matches anything, binds nothing
```

### 6.2 Variable Pattern

```
x               -- Matches anything, binds value to x
```

### 6.3 Constructor Pattern

```
Circle r        -- Matches Circle constructor, binds radius to r
Rect w h        -- Matches Rect constructor
Some x          -- Matches Some constructor
None            -- Matches None (no payload)
```

### 6.4 Literal Pattern

```
0               -- Matches integer 0 (polymorphic: works with I32 or U32)
42u             -- Matches U32 value 42
1               -- Matches integer 1
```

Integer literal patterns on I32/U32 scrutinees emit **native WGSL `switch/case`** when all arms are literal patterns without guards.

### 6.5 Or-Pattern (Multi-Value)

```
| 4 | 8  -> ...     -- Matches 4 or 8
| Red | Blue -> ...  -- Matches Red or Blue
```

Or-patterns generate multi-value `case` arms in WGSL switch statements:

```wgsl
case 4i, 8i: { ... }
```

### 6.6 Tuple Pattern

```
(a, b)          -- Destructure a pair
```

Tuple patterns in function parameters are desugared to curried parameters.

### 6.7 As-Pattern

```
x@(Circle r)    -- Bind the whole value to x AND destructure
```

### 6.8 Parenthesized Pattern

```
(Circle r)      -- Grouping
```

---

## 7. Operators

### 7.1 Precedence Table

From **lowest** to **highest** binding power:

| Precedence | Operators | Associativity | Description |
|-----------|-----------|---------------|-------------|
| 1 | `\|>` | Left | Pipeline |
| 1 | `$` | Right | Low-precedence application |
| 1–2 | `\|\|` | Left | Logical OR |
| 3–4 | `&&` | Left | Logical AND |
| 5–6 | `^` | Left | Bitwise XOR |
| 7–8 | `&` | Left | Bitwise AND |
| 9–10 | `==` `/=` `<` `>` `<=` `>=` | Left | Comparison |
| 11–12 | `<<` `>>` | Left | Bit shift |
| 13–14 | `+` `-` | Left | Additive |
| 15–16 | `*` `/` `%` | Left | Multiplicative |
| 19 | (application) | Left | Function application |
| 21 | `-` `!` `~` (prefix), `.field`, `[index]` | — | Unary, access |

### 7.2 Operator Syntax

- Infix operators: `a + b`, `x == y`, `x & y`, `x ^ y`, `x << y`
- Prefix operators: `-x` (negation), `!b` (boolean not), `~x` (bitwise not)
- Operator sections: `(+)` wraps an operator as a function value
- Backtick infix: `` a `f` b `` → `f a b`

### 7.3 Not-Equal Operator

fwgsl uses `/=` (Haskell-style) for not-equal in source code. It compiles to `!=` in WGSL.

### 7.4 Bitwise Operators

| fwgsl | WGSL | Description |
|-------|------|-------------|
| `x & y` | `x & y` | Bitwise AND |
| `x ^ y` | `x ^ y` | Bitwise XOR |
| `bor x y` | `x \| y` | Bitwise OR (function; `\|` is reserved for pattern syntax) |
| `x << y` | `x << y` | Shift left |
| `x >> y` | `x >> y` | Shift right (two adjacent `>` with no space between them) |
| `shr x y` | `x >> y` | Shift right (function alternative) |
| `~x` | `~x` | Bitwise NOT |

**Note:** `|` cannot be used as an infix bitwise OR operator because it is reserved for pattern matching guards, match arms, data constructor separators, and or-patterns. Use the `bor` function instead. Similarly, `>>` must be written with no space between the two `>` characters to distinguish it from nested generic type closers (e.g., `Vec<4, Vec<3, F32>>`). The `shr` function is available as an alternative.

### 7.5 Vector Literals

Vector literals use bracket syntax and desugar to WGSL vector constructors:

```
[1.0, 2.0, 3.0]       -- desugars to vec3<f32>(1.0, 2.0, 3.0)
[x, y, z, 1.0]         -- desugars to vec4<f32>(x, y, z, 1.0)
[base.xy, 0.0, 1.0]   -- components can be swizzled vectors
```

The number of components (2–4) is inferred from the elements. Scalar and vector elements can be mixed (e.g., a `vec2` element contributes 2 components).

---

## 8. Type System

### 8.1 Hindley-Milner Type Inference

fwgsl uses a constraint-based Hindley-Milner type inference engine:

- **Fresh type variables** are generated for unknown types
- **Unification** resolves constraints between types
- **Substitution** maps type variables to concrete types
- **Generalization** creates polymorphic type schemes over variables not free in the environment
- **Instantiation** replaces scheme variables with fresh variables at each use site

### 8.2 Type Unification Rules

- `Var(v)` unifies with any type (occurs check prevents infinite types)
- `Con(a)` unifies with `Con(b)` only if `a == b`
- `Arrow(a1, a2)` unifies with `Arrow(b1, b2)` by unifying components
- `App(f1, a1)` unifies with `App(f2, a2)` by unifying components
- `Tuple(elems1)` unifies with `Tuple(elems2)` if lengths match, by unifying elements
- `Nat(a)` unifies with `Nat(b)` only if `a == b`
- `Error` unifies with anything (error recovery)

### 8.3 Type Constructor Normalization

Surface type names are normalized to canonical forms:

| Surface Names | Canonical |
|---------------|-----------|
| `Array`, `Tensor`, `Ten` | `Tensor` |
| `Vec`, `Vector` | `Vec` |
| `Mat`, `Matrix` | `Mat` |
| `Sca`, `Scalar` | `Scalar` (identity: `Scalar F32` = `F32`) |
| `Options`, `Option` | `Option` |

### 8.4 Type Schemes

Polymorphic types are represented as schemes with quantified variables:

```
-- The scheme for `id : a -> a` is:
Scheme { vars: [0], ty: Arrow(Var(0), Var(0)) }
```

Each time a polymorphic name is used, its scheme is instantiated with fresh type variables, enabling type-safe reuse.

### 8.5 Constructor Types

Data type constructors are assigned types during registration:

```
data Option a = Some a | None

-- Some : a -> Option a
-- None : Option a
```

Record constructors take named fields:

```
data Point = Point { x : F32, y : F32 }
-- Point : (record) -> Point
```

---

## 9. Traits and Implementations

### 9.1 Trait Declaration

```
trait TraitName typeVar where
  methodName : type
  ...
```

Traits define interfaces with method signatures:

```
trait Add a where
  (+) : a -> a -> a
```

### 9.2 Trait Implementation

```
impl TraitName ConcreteType where
  methodName args = body
  ...
```

Example:

```
impl Add Fp64 where
  (+) a b =
    let s = twoSum a.high b.high
    in quickTwoSum (s.high, s.low + a.low + b.low)
```

### 9.3 Standalone Implementations

```
impl TypeName where
  methodName args = body
```

Standalone `impl` blocks define methods on a type without a trait:

```
impl Fp64 where
  collapse v = v.high + v.low
  neg a = Fp64 (-a.high) (-a.low)
```

### 9.4 Operator Overloading

**Arithmetic operator traits:** `Add` (`+`), `Sub` (`-`), `Mul` (`*`), `Div` (`/`), `Mod` (`%`).

**Bitwise operator traits:** `BitAnd` (`&`), `BitXor` (`^`), `Shl` (`<<`), `Shr` (`shr`), `BitNot` (`bitnot`), `Neg` (`negate`).

Operators on primitive types (I32, U32, F32) use native WGSL operators. Operators on user-defined types dispatch through trait implementations:

```
impl Add Fp64 where
  (+) a b = ...

-- Now x + y where x, y : Fp64 calls the trait method

impl BitAnd Mask where
  (&) a b = Mask { bits = a.bits & b.bits }

-- Now a & b where a, b : Mask calls bitand_Mask

impl BitNot Mask where
  bitnot a = Mask { bits = ~a.bits }

-- Now ~a where a : Mask calls bitnot_Mask

impl Neg Wrapper where
  negate a = Wrapper { val = -a.val }

-- Now -a where a : Wrapper calls negate_Wrapper
```

The operator method syntax uses parenthesized operator names: `(+)`, `(-)`, `(*)`, `(&)`, `(^)`, `(<<)`, etc. Non-operator trait methods like `shr`, `bitnot`, and `negate` use plain names.

### 9.5 Dispatch Mechanism

Trait dispatch is **fully static** — no vtables or runtime dispatch. Impl methods are compiled as regular functions with mangled names (e.g., `add_Fp64`). At trait resolution time, `Var(method)` is rewritten to `Var(mangled_name)` based on the resolved type of the operands.

---

## 10. Module System

### 10.1 Module Structure

Each file is a module. Directory structure defines namespaces:

```
src/
  Math/
    Fp64.fwgsl      -- module Math.Fp64
    Utils.fwgsl      -- module Math.Utils
  Main.fwgsl         -- module Main
```

### 10.2 Module Header

```
module Math.Fp64
```

Optional. If absent, the module name is derived from the file path relative to source roots.

### 10.3 Import Forms

| Syntax | Description |
|--------|-------------|
| `import Foo` | Import all public names from Foo |
| `import Foo (bar, baz)` | Import only `bar` and `baz` |
| `import Foo as F` | Qualified access: `F.bar` |
| `import Foo.*` | Import all sub-modules under Foo/ |
| `import Foo when cfg.debug` | Conditional import |

### 10.4 Visibility

Everything is **public by default**. There is a planned `private` keyword for module-local declarations (not yet implemented).

### 10.5 Module Resolution

The module resolver (`fwgsl_parser::module_resolver`) resolves imports to source files:

1. Builds a `ModuleGraph` from the root file's imports
2. Reads imported files via a `SourceReader` trait (supports filesystem or `VirtualFs`)
3. Performs topological sort with cycle detection
4. Merges modules in dependency order

### 10.6 Virtual Filesystem

`VirtualFs` provides an in-memory filesystem for browser/WASM/testing:

```rust
let mut vfs = VirtualFs::new();
vfs.add("Math/Fp64.fwgsl", source_code);
```

### 10.7 Bundle Format

For single-file multi-module embedding, `parse_bundle()` parses section markers:

```
--- module Math.Fp64 ---
... declarations ...

--- module Main ---
... declarations ...
```

---

## 11. Conditional Compilation

### 11.1 Feature Flags

Features are enabled via the CLI:

```
fwgsl compile file.fwgsl --feature debug --feature aa
```

Features are referenced in source as `cfg.name`.

### 11.2 Block Form

```
when cfg.debug
  debugLog : F32 -> ()
  debugLog x = ...
```

Declarations in the `when` body must be indented further than the `when` keyword.

**Limitation:** Each `when` block supports one type signature + one function definition pair. For multiple functions, use separate `when` blocks.

### 11.3 Else / Else-When

```
when cfg.aa
  calculateAA : F32 -> F32
  calculateAA x = ...
else
  calculateAA : F32 -> F32
  calculateAA x = 1.0
```

Chained:

```
when cfg.tier3
  maxLights : I32
  maxLights = 64
else when cfg.tier2
  maxLights : I32
  maxLights = 16
else
  maxLights : I32
  maxLights = 4
```

### 11.4 Conditional Imports

```
import Debug when cfg.debug
```

### 11.5 Predicate Combinators

| Syntax | Description |
|--------|-------------|
| `cfg.name` | Feature flag is set |
| `not pred` | Negation (tightest binding) |
| `pred && pred` | Conjunction |
| `pred \|\| pred` | Disjunction (loosest binding) |

Example:

```
when cfg.debug && cfg.msaa
  debugSamples : I32
  debugSamples = 4
```

### 11.6 Feature Evaluation

`FeatureSet::from_flags(&[String])` creates the feature set. `evaluate_features(&mut Program, &FeatureSet)` prunes the AST in-place after parsing and before module resolution — declarations in unfulfilled `when` branches are removed entirely from the AST.

---

## 12. Bitfields

Bitfields provide packed integer flag words with named fields.

### 12.1 Declaration

```
bitfield Name : BaseType = ConstructorName {
  field1 : kind1,
  field2 : kind2,
  ...
}
```

The constructor name appears after `=`, mirroring `data` record syntax. Fields support four forms:

| Syntax | Meaning | Accessor returns |
|---|---|---|
| `name : Type : N` | Typed field with explicit bit width | `Type` |
| `name : Bool` | Boolean field, always 1 bit | `Bool` |
| `name : EnumType` | Enum-typed, width inferred from variant count | `EnumType` |
| `name : N` | Bare integer width | base type (U32) |

Example:

```
bitfield CapFlags : U32 = CapFlags {
  endCap   : Bool,
  startCap : Bool,
  capButt  : Bool,
  capRound : Bool,
  capArrow : Bool,
}
```

### 12.2 Typed Fields

Fields can specify an explicit type and bit width:

```
bitfield LineFlags : U32 = LineFlags {
  capStart  : CapStyle : 2,    -- CapStyle enum stored in 2 bits
  capEnd    : CapStyle : 2,
  roughness : U32 : 5,         -- 5-bit unsigned integer
  visible   : Bool,            -- 1 bit (width implicit)
  mode      : BlendMode,       -- width inferred from enum variant count
}
```

For `Type : N` fields where `Type` is an enum, the compiler validates that `N >= ceil(log2(variant_count))`.

`Bool` always maps to width 1.

Bare integer widths (`name : N`) remain supported for quick prototyping.

### 12.3 Width Validation

The compiler checks at compile time that:
- The total bit width of all fields does not exceed the base type width (32 bits for U32)
- Typed fields have sufficient bits for their declared type

### 12.4 Field Access

```
flags.capRound      -- extract field via shift and mask: (flags >> offset) & mask
```

1-bit fields produce a boolean result: `(val >> offset & 1u) != 0u`.

### 12.5 Construction

```
CapFlags { endCap = 1, startCap = 0, capButt = 0, capRound = 1, capArrow = 0 }
```

Compiles to OR-chain of shifted masked values. For 1-bit fields with boolean values, uses `select(0u, 1u, val)`. For integer values, casts to U32 directly.

### 12.6 Functional Update

```
flags { capArrow = 1 }     -- clear field bits, then OR in new value
```

Compiles to: `(base & ~combined_mask) | ((new_val & field_mask) << offset)`.

### 12.7 Implementation Notes

- Bitfields are **not** registered as type aliases — they remain opaque so `FieldAccess` preserves the type name
- No WGSL structs are emitted for bitfields — they lower to plain integer operations
- Bitfield info is threaded through `LowerCtx` in HIR→MIR lowering
- `impl` blocks work on bitfield types (they are `Ty::Con("Name")` in the type system)

---

## 13. Entry Points and GPU Resources

### 13.1 Shader Stages

```
main : ComputeInput -> ()
@compute @workgroup_size(64, 1, 1)
main input = ...

vsMain : VertexInput -> VertexOutput
@vertex
vsMain input = ...

fsMain : VertexOutput -> Vec<4, F32>
@fragment
fsMain input = ...
```

### 13.2 Struct-Based I/O

Entry points use struct-based I/O. Input and output structs carry `@builtin` and `@location` attributes on their fields:

```
data ComputeInput = ComputeInput {
  @builtin(global_invocation_id) gid : Vec<3, U32>
}

data VertexOutput = VertexOutput {
  @builtin(position) clip_position : Vec<4, F32>,
  @location(0)       color         : Vec<4, F32>,
}
```

### 13.3 Return Type Annotations

For vertex/fragment entry points returning non-struct types (e.g., `Vec<4, F32>`), the codegen automatically emits `@location(0)` on the return type:

```wgsl
@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> { ... }
```

### 13.4 Resource Bindings

```
extern resource name : Uniform<T>                      @group G @binding B
extern resource name : Storage<ReadWrite, Array<T>>     @group G @binding B
```

Compiles to WGSL:

```wgsl
@group(G) @binding(B) var<uniform> name: T;
@group(G) @binding(B) var<storage, read_write> name: array<T>;
```

### 13.5 Resource Operations

| fwgsl | WGSL | Description |
|-------|------|-------------|
| `load resource` | `resource` (identity) | Read from uniform/storage |
| `writeAt resource index value` | `resource[index] = value` | Write to storage buffer |

---

## 14. Prelude and Builtins

The prelude (`prelude/prelude.fwgsl`) is loaded via `include_str!` and prepended to every compilation. It provides type signatures for built-in operations.

### 14.1 Prelude Data Types

```
data Option a = Some a | None
data Result a = Ok a | Err String
data Pair a b = Pair a b
```

### 14.2 Operator Traits

**Arithmetic:**
```
trait Add a where (+) : a -> a -> a
trait Sub a where (-) : a -> a -> a
trait Mul a where (*) : a -> a -> a
trait Div a where (/) : a -> a -> a
trait Mod a where (%) : a -> a -> a
```

**Bitwise:**
```
trait BitAnd a where (&) : a -> a -> a
trait BitXor a where (^) : a -> a -> a
trait Shl a where (<<) : a -> a -> a
trait Shr a where shr : a -> a -> a
trait BitNot a where bitnot : a -> a
trait Neg a where negate : a -> a
```

### 14.3 Arithmetic Operators

```
extern (+) : a -> a -> a
extern (-) : a -> a -> a
extern (*) : a -> a -> a
extern (/) : a -> a -> a
extern (%) : a -> a -> a
```

### 14.4 Comparison Operators

```
extern (==) : a -> a -> Bool
extern (/=) : a -> a -> Bool
extern (<)  : a -> a -> Bool
extern (>)  : a -> a -> Bool
extern (<=) : a -> a -> Bool
extern (>=) : a -> a -> Bool
```

### 14.5 Logical Operators

```
extern (&&) : Bool -> Bool -> Bool
extern (||) : Bool -> Bool -> Bool
```

### 14.6 Bitwise Operators

```
extern (&)  : a -> a -> a   -- bitwise AND
extern (^)  : a -> a -> a   -- bitwise XOR
extern (<<) : a -> a -> a   -- shift left
extern bor  : a -> a -> a   -- bitwise OR (| conflicts with pattern syntax)
extern shr  : a -> a -> a   -- shift right (>> works as infix with no space)
```

Prefix bitwise NOT (`~x`) is a built-in prefix operator; no `extern` needed.

### 14.7 Loop Combinators

```
extern foldRange : I32 -> I32 -> a -> (a -> I32 -> a) -> a
```

`foldRange start end init f` folds `f` over the integer range `[start, end)`, threading an accumulator. Compiles to an efficient WGSL `loop` with mutable variables.

### 14.8 Math Functions (Unary)

```
sin  cos  abs  fract  floor  sign  sqrt  log  log2  exp  ceil  round  trunc  negate
```

All have type `a -> a`.

### 14.9 Math Functions (Binary)

```
max  min  step  mod  pow  reflect  atan  atan2
```

All have type `a -> a -> a`.

### 14.10 Math Functions (Ternary)

```
clamp     : a -> a -> a -> a
mix       : a -> a -> b -> a
smoothstep : a -> a -> b -> a
```

### 14.11 Vector Operations

```
normalize : Vec<n, a> -> Vec<n, a>
length    : Vec<n, a> -> a
dot       : Vec<n, a> -> Vec<n, a> -> a
distance  : Vec<n, a> -> Vec<n, a> -> a
cross     : Vec<3, a> -> Vec<3, a> -> Vec<3, a>
select    : a -> a -> Bool -> a
```

### 14.12 Packing / Unpacking

```
unpack4x8unorm : U32 -> Vec<4, F32>
pack4x8unorm   : Vec<4, F32> -> U32
```

### 14.13 Vector Constructors

```
vec2 : a -> a -> Vec<2, a>
vec3 : a -> a -> a -> Vec<3, a>
vec4 : a -> a -> a -> a -> Vec<4, a>
splat2 : a -> Vec<2, a>
splat3 : a -> Vec<3, a>
splat4 : a -> Vec<4, a>
```

### 14.14 Fragment Shader Derivatives

```
dpdx  dpdy  dpdxCoarse  dpdxFine  dpdyCoarse  dpdyFine  fwidth  fwidthCoarse  fwidthFine
```

All have type `a -> a`.

### 14.15 Type Cast Builtins

| fwgsl | WGSL | Type |
|-------|------|------|
| `toF32 x` | `f32(x)` | `a -> F32` |
| `toI32 x` | `i32(x)` | `a -> I32` |
| `toU32 x` | `u32(x)` | `a -> U32` |
| `toBool x` | `bool(x)` | `a -> Bool` |

### 14.16 Resource Operations

| fwgsl | Behavior |
|-------|----------|
| `load x` | Identity (reads resource value) |
| `writeAt buf idx val` | `buf[idx] = val` |

---

## 15. Compilation Pipeline

```
Source → Parse → Feature Eval → Module Resolution → Module Merge
     → Semantic Analysis → AST→HIR Lowering → HIR→MIR Lowering
     → Dead Code Elimination → WGSL Code Generation
```

### 15.1 Parsing

- Lexer tokenizes source into `Token` stream
- Layout resolver inserts virtual indentation tokens
- Parser produces AST (`Program` with `Vec<Decl>`)

### 15.2 Feature Evaluation

`evaluate_features()` prunes `CfgDecl` nodes based on active `--feature` flags, removing dead branches from the AST.

### 15.3 Module Resolution

If the program has `import` declarations, the module resolver:
1. Discovers and parses imported files
2. Builds a dependency graph
3. Topologically sorts modules
4. Merges into a single flat `Program`

### 15.4 Semantic Analysis

The `SemanticAnalyzer`:
- Registers data types and their constructors
- Registers type aliases
- Registers traits and impls
- Performs name resolution
- Runs Hindley-Milner type inference on all expressions
- Validates type correctness of all match arms, guards, let bindings
- Desugars tuple parameters and call sites
- Handles method-call sugar resolution

### 15.5 AST → HIR Lowering

The `AstLowering` phase:
- Lowers AST expressions to typed HIR expressions
- Resolves trait methods to mangled concrete function names
- Resolves operator overloading (BinOp → App for user-defined types)
- Beta-reduces lambda applications
- Desugars pipeline `|>` to function application
- Desugars method-call syntax to function application
- Resolves bitfield operations
- Runs a finalization pass to apply all type substitutions

### 15.6 HIR → MIR Lowering

The MIR lowering phase:
- Converts functional expressions to imperative statements
- Lowers `let` bindings to `MirStmt::Let`
- Lowers `if/then/else` to `MirStmt::If`
- Lowers `match/case` to either native `switch/case` or if-else chains
- Lowers `loop` to `MirStmt::Loop` with `var` + `continue` + `break`
- Lowers bitfield access to shift/mask operations
- Lowers bitfield construction/update to bit manipulation
- Converts `toF32`/`toI32`/`toU32`/`toBool` to `MirExpr::Cast`
- Converts `load` to identity, `writeAt` to `MirStmt::IndexAssign`

### 15.7 Dead Code Elimination

`reachability::eliminate_dead_code()` removes functions not reachable from any entry point.

### 15.8 WGSL Code Generation

The codegen emits valid WGSL text:
- Structs with field types and attributes
- Global resource bindings with address spaces
- Module-level constants
- Functions with typed parameters and return types
- Entry points with stage attributes and workgroup sizes
- `@location(0)` on non-struct return types of vertex/fragment entry points
- Identifier sanitization (WGSL reserved words)

---

## 16. WGSL Code Generation

### 16.1 Type Mapping

| fwgsl | WGSL |
|-------|------|
| `I32` | `i32` |
| `U32` | `u32` |
| `F32` | `f32` |
| `Bool` | `bool` |
| `Vec<N, T>` | `vecN<T>` |
| `Mat<R, C, T>` | `matRxC<T>` |
| `Array<T, N>` | `array<T, N>` |
| `Array<T>` | `array<T>` |
| `()` | (no return type) |
| User struct | `StructName` |

### 16.2 ADT Encoding

**Pure enums** (all constructors have no fields):

```
data Color = Red | Green | Blue
-- Compiles to: u32 values (Red=0, Green=1, Blue=2)
```

**Sum types** with fields:

```
data Shape = Circle F32 | Rect F32 F32
-- Compiles to:
struct Shape {
  tag: u32,
  field0: f32,
  field1: f32,
}
```

### 16.3 Match Compilation

**Integer literal match** (no guards, all literal/or patterns): native WGSL `switch/case`:

```wgsl
switch (_scrut) {
  case 1i: { ... }
  case 2i, 3i: { ... }    // multi-value from or-patterns
  default: { ... }
}
```

**Other matches**: if-else chain with tag checks and field extraction.

**When-guards**: nested `if (guard) { body } else { fallthrough }` inside pattern match conditions.

### 16.4 Loop Compilation

```
loop go (i = 0) (acc = 0) in
  if i < n then go (acc + i) (i + 1) else acc
```

Compiles to:

```wgsl
var i: i32 = 0i;
var acc: i32 = 0i;
var _result: i32 = 0i;
loop {
  if (i < n) {
    let _tmp_0 = acc + i;
    let _tmp_1 = i + 1i;
    i = _tmp_0;    // temp lets avoid ordering issues
    acc = _tmp_1;
    continue;
  } else {
    _result = acc;
    break;
  }
}
```

---

## 17. Tooling

### 17.1 Crate Architecture

| Crate | Purpose |
|-------|---------|
| `fwgsl_syntax` | Token kinds and SyntaxKind enum |
| `fwgsl_span` | Source span tracking |
| `fwgsl_diagnostics` | Diagnostic infrastructure (errors, warnings) |
| `fwgsl_allocator` | Arena allocator |
| `fwgsl_cst` | Concrete syntax tree (placeholder) |
| `fwgsl_ast` | Abstract syntax tree (placeholder) |
| `fwgsl_parser` | Lexer, layout resolver, parser, module resolver, feature eval |
| `fwgsl_typechecker` | Type representation, unification, inference engine |
| `fwgsl_semantic` | Semantic analysis (name resolution, type inference) |
| `fwgsl_ast_lowering` | AST → HIR lowering |
| `fwgsl_hir` | High-level IR (typed, desugared) |
| `fwgsl_mir` | Mid-level IR (imperative, WGSL-close) + HIR→MIR lowering |
| `fwgsl_wgsl_codegen` | MIR → WGSL text emission |
| `fwgsl_ide` | IDE features (completions, hover, goto-def, references) |
| `fwgsl_formatter` | Token-stream based code formatter |
| `fwgsl_language_server` | LSP server (tower-lsp over stdin/stdout) |
| `fwgsl_wasm` | WASM compilation target (compile, parse, format, diagnostics, IDE) |
| `fwgsl_cli` | Command-line interface |
| `fwgsl_integration_tests` | End-to-end compiler tests |

### 17.2 CLI

```
fwgsl compile <file>    Compile .fwgsl to .wgsl (stdout)
fwgsl check <file>      Type-check without emitting
fwgsl fmt <file>        Format source code
fwgsl version           Print version
fwgsl help              Print help
```

**Options:**

| Flag | Description |
|------|-------------|
| `--emit-ast` | Print AST debug output |
| `--preserve-comments` | Preserve source comments in WGSL output |
| `--feature <name>` | Enable a compile-time feature flag (repeatable) |

### 17.3 Language Server (LSP)

The `fwgsl-lsp` binary provides a Language Server Protocol server over stdin/stdout (via tower-lsp + tokio).

**Supported features:**
- Diagnostics (parse errors, type errors)
- Completions (context-aware)
- Hover (type information)
- Go-to-definition
- Find references
- Semantic tokens
- Formatting

### 17.4 Formatter

The `fwgsl_formatter` crate provides token-stream based formatting:

```rust
format_default(source: &str) -> String
format(source: &str, config: &FormatConfig) -> String
```

`FormatConfig` supports configurable indentation width (default: 2).

### 17.5 WASM Target

The `fwgsl_wasm` crate compiles to `wasm32-unknown-unknown` and exports:

| Function | Description |
|----------|-------------|
| `compile(source)` | Full compilation to WGSL |
| `parse_ast(source)` | Parse and return AST debug string |
| `format(source)` | Format source code |
| `get_diagnostics(source)` | Return diagnostics as JSON |
| `editor_completions(source, line, col)` | Completions at position |
| `editor_hover(source, line, col)` | Hover info at position |
| `editor_definition(source, line, col)` | Go-to-definition |
| `editor_references(source, ...)` | Find references |

### 17.6 Editor Integrations

**VS Code** (`editors/vscode/`)
- TextMate grammar for syntax highlighting
- LSP client that spawns `fwgsl-lsp`

**Helix** (`editors/helix/`)
- `languages.toml` configuration
- Tree-sitter query files

**Zed** (`editors/zed/`)
- `extension.toml` + `config.toml`
- Query files for highlights, indents, outline, brackets

**Tree-sitter** (`tree-sitter-fwgsl/`)
- `grammar.js` with query files (highlights, locals, indents, textobjects)
- Excluded from workspace (separate build)
- Note: Tree-sitter has limitations with indentation-sensitive multi-line constructs (no external scanner)
- LSP semantic tokens provide the most accurate highlighting

---

## Appendix A: Complete Example

```fwgsl
-- A simple compute shader that scales a buffer of vec4s.

data ComputeInput = ComputeInput {
  @builtin(global_invocation_id) gid : Vec<3, U32>
}

extern resource input  : Storage<ReadWrite, Array<Vec<4, F32>>> @group 0 @binding 0
extern resource output : Storage<ReadWrite, Array<Vec<4, F32>>> @group 0 @binding 1

const SCALE : F32 = 2.0

scaleVec : Vec<4, F32> -> Vec<4, F32>
scaleVec v = vec4 (v.x * SCALE) (v.y * SCALE) (v.z * SCALE) v.w

main : ComputeInput -> ()
@compute @workgroup_size(64, 1, 1)
main input =
  let idx = toU32 input.gid.x
      v   = load (input[idx])
  in writeAt output idx (scaleVec v)
```

## Appendix B: Feature Summary

| Feature | Status |
|---------|--------|
| Hindley-Milner type inference | Implemented |
| Algebraic data types (records, enums, ADTs) | Implemented |
| Pattern matching with when-guards | Implemented |
| Native WGSL switch/case for integer patterns | Implemented |
| Multi-value or-patterns | Implemented |
| Named tail-recursive loops | Implemented |
| Traits with static dispatch | Implemented |
| Operator overloading | Implemented |
| Module system (file = module) | Implemented |
| Conditional compilation (`when cfg.x`) | Implemented |
| Bitfields with typed fields | Implemented |
| Pipeline operator (`\|>`) | Implemented |
| Lambda expressions (beta-reduced) | Implemented |
| Method-call syntax sugar | Implemented |
| Vec swizzle patterns | Implemented |
| Vec literal syntax (`[a, b, c]`) | Implemented |
| Tuple desugaring | Implemented |
| Struct-based entry point I/O | Implemented |
| Compute / vertex / fragment stages | Implemented |
| Dead code elimination | Implemented |
| LSP (diagnostics, completions, hover, goto-def, references) | Implemented |
| Code formatter | Implemented |
| WASM compilation target | Implemented |
| VS Code, Helix, Zed editor support | Implemented |
| Tree-sitter grammar | Implemented |
