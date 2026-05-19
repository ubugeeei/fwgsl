# Changelog

All notable changes to `fwgsl` are recorded here.

## [Unreleased]

### CLI

- `fwgsl` now does real argv-driven dispatch (#47). Subcommands:
  `compile`, `check`, `fmt`, `version`, `help` (with `-h` / `-V`
  aliases). Inputs come from `<file>`, `--source <text>`, or
  `--stdin`. The library `run_command(cmd, source, label) ->
  (String, Int)` returns ready-to-print stdout plus an exit code
  (`0` ok, `1` for any error-severity diagnostic, `2` for usage
  errors).
- `fwgsl check` now surfaces lint warnings alongside parser and
  semantic diagnostics, ordered errors-first (#65 → #66).

### Language Server / IDE

- **Hover** now renders the inferred type scheme for identifiers
  and constructors (`add : I32 -> I32 -> I32`, `id : a -> a`,
  `Some : a -> Option a`) instead of the lexer-only label (#49).
- **Completions** now include in-scope value bindings (Function /
  Variable, classified by whether the scheme is an arrow),
  constructors (`Some`, `Ok`, user `data` decls), and data type
  names, alongside the existing keyword list (#57 → #58).
- **Semantic tokens** now distinguish types, functions, variables,
  decorators (`@compute`), numbers, and strings rather than only
  keyword / operator / comment (#59 → #60).
- **Inlay hints** for top-level decls without an explicit type
  signature, anchored just after the function name (#61 → #62).
- **Code actions**: first action `Add type signature for X`, with
  preserved indentation (#63 → #64).
- **Document symbols** outline (`textDocument/documentSymbol`)
  with `Function` / `Struct` / `EnumMember` / `TypeParameter`
  kinds; constructors nest under their data declaration (#67 →
  #68).
- `compute_diagnostics` includes lint warnings on top of parser /
  semantic errors (#65 → #66).

### Linter

- New `moonbit/linter` package (#52 → #53). `LintRuleId` enum
  (`UnusedLet`, `UnusedWhere`, `ShadowedBinding`,
  `RedundantWildcard`) with kebab-case `code()` names. `lint`
  re-parses; `lint_program` works on an already-parsed AST.
- Rules: `unused-let`, `unused-where`, `redundant-wildcard` (#52 →
  #53), `shadowed-binding` (#69 → #70). Identifiers with a
  leading `_` opt out of the unused / shadow checks.

### Diagnostics

- `render` now embeds a `miette`-style source-line excerpt with a
  caret pointing at the offending span (#75). Multi-token spans
  get a wider caret. The gutter width adapts to the line number.
- Nine new end-to-end tests pin down span accuracy, help-text
  presence, type-mismatch reporting, parser-error shape, multi-
  error pipelines, error labels, and clean-source silence (#50 →
  #51).

### Housekeeping

- `not(expr)` -> `!expr` across every package: 66 call sites in
  #54 → #55, plus the three stragglers in the new linter package
  in #71 → #72.
- `String::substring(start=, end=)` -> `String::unsafe_substring`
  across every package: ~36 call sites (#71 → #72).
- README and `examples/README.md` scrub stale `crates/fwgsl_*`
  paths and absolute developer-machine paths (#56, #74).

## [0.1.0-moonbit] — 2026-05-19

### Changed

- The entire compiler is now written in [MoonBit](https://www.moonbitlang.com/).
  The original Rust workspace is gone; every package lives under `moonbit/`
  and targets `wasm-gc`. Migration tracker: [#4](https://github.com/ubugeeei/fwgsl/issues/4).

### Compiler

- **`moonbit/span`** — `Span`, `Atom`, `Spanned[T]` with `merge`,
  `contains`, `source_text`.
- **`moonbit/diagnostics`** — `Diagnostic`, `Label`, `DiagnosticSink`
  builder API and a `render` function that replaces the Rust `miette`
  integration.
- **`moonbit/syntax`** — `SyntaxKind` with `is_trivia` / `is_keyword`
  / `is_operator` / `is_literal` classifiers and a `keyword_from_str`
  resolver. `pub impl Show for SyntaxKind`.
- **`moonbit/cst`** — `GreenToken`, `GreenNode`, `GreenChild`.
- **`moonbit/ast`** — `Lit`, `Type`, `Pat`, `Expr`, `DoStmt`,
  `ConFields`, `ConDecl`, `Attribute`, `Decl`, `Program`, with
  per-node `span()` accessors.
- **`moonbit/parser`** — Hand-written lexer, indentation-sensitive
  layout resolver, and a recursive-descent parser with Pratt
  expression handling. Drives diagnostics through the same sink.
- **`moonbit/typechecker`** — `Ty`, `Scheme`, `Substitution`,
  `TypeEnv`, `InferEngine` (`fresh_var`, `unify`, `instantiate`,
  `generalize`, `finalize`), `ConstructorInfo`, `ConstructorFields`,
  `WgslType`, `ty_to_wgsl`, `normalize_type_aliases`.
- **`moonbit/semantic`** — Full builtin prelude (`id`, `const`,
  `flip`, `compose`, `pure`, `bind`, `fmap` / `map` / `$map`,
  `filter`, `fold` / `foldr`, `flat` / `flatMap`, `zip`, `all` /
  `any`, `fst` / `snd` / `swap`, `unwrapOr`, the unary / binary
  numeric builtins, `$normalize`, `$length`, `$dot`,
  `$vec2`/`$vec3`/`$vec4`, `$splat*`, `$vecX/Y/Z/W`) plus
  Algorithm-W inference over the parser AST. `is_swizzle`,
  `validate_swizzle`, `extract_vec_type` exposed for downstream use.
- **`moonbit/ast_lowering`** — Full AST -> HIR lowering with HM
  inference and finalisation.
- **`moonbit/hir`** — Typed `HirProgram`, `HirFunction`,
  `HirEntryPoint`, `HirExpr` with `ty()`, `BinOp` with
  `parse` / `to_wgsl_str`.
- **`moonbit/mir`** — WGSL-shaped IR (`MirType`, `MirStmt`,
  `MirExpr`, `MirEntryPoint`) plus the HIR -> MIR lowering pass
  (curried-app flattening, if/case desugaring, `negate` / `$mod` /
  `$splat*` / `$vecX..` builtin rewrites, compute-shader `_gid`
  binding).
- **`moonbit/wgsl_codegen`** — Tree-walk emitter that produces
  WGSL text; `sanitize_identifier`, WGSL reserved-word handling,
  `vec3<f32>(...)` constructor inference.
- **`moonbit/ide`** — `CompletionItem`, `HoverInfo`, `GotoTarget`,
  `SemanticToken`, semantic-token producer running off the real
  lexer.
- **`moonbit/language_server`** — `DocumentStore` lifecycle,
  `compute_diagnostics` returning `LspDiagnostic` records, JSON
  serialisers ready for a JSON-RPC transport
  (`diagnostic_to_json`, `diagnostics_json_for`), and proxies
  through to the `ide` library.
- **`moonbit/wasm`** — Pure MoonBit `wasm_check` / `wasm_compile`
  entry points that return JSON. Replaces the `wasm-bindgen` glue
  layer entirely.
- **`moonbit/cli`** — `check_source`, `compile_source` returning a
  `CompileResult` (`program`, `wgsl`, `diagnostics`, `has_errors`),
  `format_diagnostic` / `format_diagnostics`, `wrap_wgsl`,
  `print_help`. The binary lives at `moonbit/cli/cmd/main`.
- **`moonbit/integration_tests`** — End-to-end pipeline tests
  asserting actual WGSL output for the `hello`, `adt match`,
  `where`, compute / vertex / fragment entry points, `if`,
  operator precedence, and generic-data sample programs.

### CI

- `Check` and `WASM Build` jobs are gone; `MoonBit Check` is the
  only job and runs without `continue-on-error`.
- `.gitignore` now excludes `_build/`, `.mooncakes/`, and
  `playground/pkg/`.

### Removed

- `crates/` and the workspace `Cargo.toml` (#34).
- `rustfmt.toml` and the `cargo` / `clippy` / `wasm-pack` mise tasks (#39).
