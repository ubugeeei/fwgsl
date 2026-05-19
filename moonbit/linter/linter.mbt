///|
/// Linter for fwgsl.
///
/// Runs static checks on the AST that the type-checker doesn't catch on
/// its own — unused bindings, shadowed names, redundant match arms — and
/// reports them as warnings the LSP / CLI can surface alongside parser
/// and semantic diagnostics.
///
/// Each rule is a pure function over `@ast.Program` that pushes
/// diagnostics into a sink. `lint(source)` is the top-level entry point
/// that ties them together.

///|
/// Identifier for a lint rule. Used in the diagnostic `code` field so
/// `// fwgsl: allow(unused-let)`-style suppressions can be added later.
pub(all) enum LintRuleId {
  UnusedLet
  UnusedWhere
  ShadowedBinding
  RedundantWildcard
} derive(Eq, Show)

///|
/// Human-readable name of a lint rule. Embedded in the diagnostic code
/// (`unused-let`, `shadowed-binding`, etc.).
pub fn LintRuleId::code(self : LintRuleId) -> String {
  match self {
    UnusedLet => "unused-let"
    UnusedWhere => "unused-where"
    ShadowedBinding => "shadowed-binding"
    RedundantWildcard => "redundant-wildcard"
  }
}

///|
/// Run every enabled lint rule on `source`. Returns the combined list of
/// warning diagnostics, ordered by source position. The list is empty
/// when the source is clean.
pub fn lint(source : String) -> Array[@diagnostics.Diagnostic] {
  let parser = @parser.Parser::new(source)
  let program = parser.parse_program()
  lint_program(program)
}

///|
/// Lower-level entry point used by the CLI / LSP which already have a
/// parsed `Program` they don't want to re-parse.
pub fn lint_program(program : @ast.Program) -> Array[@diagnostics.Diagnostic] {
  let diags : Array[@diagnostics.Diagnostic] = Array::new()
  for decl in program.decls {
    lint_decl(decl, diags)
  }
  sort_by_span(diags)
  diags
}

///|
fn lint_decl(
  decl : @ast.Decl,
  diags : Array[@diagnostics.Diagnostic]
) -> Unit {
  match decl {
    FunDecl(_, params, body, where_binds, _) => {
      lint_unused_where(where_binds, body, diags)
      lint_expr(body, diags)
      for p in params {
        lint_pattern(p, diags)
      }
      for w in where_binds {
        let (_, w_body) = w
        lint_expr(w_body, diags)
      }
    }
    EntryPoint(_, _, params, body, _) => {
      lint_expr(body, diags)
      for p in params {
        lint_pattern(p, diags)
      }
    }
    _ => ()
  }
}

///|
/// Walk an expression for lint findings. Currently visits every nested
/// `let`, `case`, `lambda`, `if`, and binary expression.
fn lint_expr(
  expr : @ast.Expr,
  diags : Array[@diagnostics.Diagnostic]
) -> Unit {
  match expr {
    Let(binds, body, _) => {
      lint_unused_let(binds, body, diags)
      for b in binds {
        let (_, e) = b
        lint_expr(e, diags)
      }
      lint_expr(body, diags)
    }
    Case(scrut, arms, _) => {
      lint_expr(scrut, diags)
      lint_redundant_wildcard(arms, diags)
      for a in arms {
        let (pat, body) = a
        lint_pattern(pat, diags)
        lint_expr(body, diags)
      }
    }
    Lambda(params, body, _) => {
      for p in params {
        lint_pattern(p, diags)
      }
      lint_expr(body, diags)
    }
    If(c, t, e, _) => {
      lint_expr(c, diags)
      lint_expr(t, diags)
      lint_expr(e, diags)
    }
    App(f, a, _) => {
      lint_expr(f, diags)
      lint_expr(a, diags)
    }
    Infix(l, _, r, _) => {
      lint_expr(l, diags)
      lint_expr(r, diags)
    }
    Paren(inner, _) | Neg(inner, _) => lint_expr(inner, diags)
    FieldAccess(inner, _, _) => lint_expr(inner, diags)
    Tuple(es, _) | VecLit(es, _) =>
      for e in es {
        lint_expr(e, diags)
      }
    Record(fields, _) =>
      for f in fields {
        let (_, e) = f
        lint_expr(e, diags)
      }
    Do(stmts, _) =>
      for s in stmts {
        match s {
          Bind(_, e, _) | ExprStmt(e, _) | LetStmt(_, e, _) =>
            lint_expr(e, diags)
        }
      }
    Lit(_, _) | Var(_, _) | Con(_, _) | OpSection(_, _) => ()
  }
}

///|
fn lint_pattern(
  pat : @ast.Pat,
  diags : Array[@diagnostics.Diagnostic]
) -> Unit {
  match pat {
    Con(_, sub, _) =>
      for p in sub {
        lint_pattern(p, diags)
      }
    Paren(p, _) | As(_, p, _) => lint_pattern(p, diags)
    Tuple(ps, _) =>
      for p in ps {
        lint_pattern(p, diags)
      }
    Record(_, fields, _) =>
      for f in fields {
        let (_, pat_opt) = f
        match pat_opt {
          Some(p) => lint_pattern(p, diags)
          None => ()
        }
      }
    Wild(_) | Var(_, _) | Lit(_, _) => ()
  }
}

// ---------------------------------------------------------------------------
// Rule: unused-let
// ---------------------------------------------------------------------------

///|
/// Flag `let x = e in body` when `x` is never mentioned in `body`.
fn lint_unused_let(
  binds : Array[(String, @ast.Expr)],
  body : @ast.Expr,
  diags : Array[@diagnostics.Diagnostic]
) -> Unit {
  for b in binds {
    let (name, e) = b
    if is_ignored(name) {
      continue
    }
    if !expr_references(body, name) && !any_binding_references(binds, name, e) {
      diags.push(
        @diagnostics.Diagnostic::warning(
          "Unused let binding `" + name + "`",
        )
          .with_code(LintRuleId::UnusedLet.code())
          .with_label(
            @diagnostics.Label::primary(e.span(), "bound but never used"),
          )
          .with_help(
            "remove the binding, rename it with a leading `_`, or use it in the body",
          ),
      )
    }
  }
}

///|
fn any_binding_references(
  binds : Array[(String, @ast.Expr)],
  name : String,
  exclude : @ast.Expr
) -> Bool {
  for b in binds {
    let (_, e) = b
    if same_expr(e, exclude) {
      continue
    }
    if expr_references(e, name) {
      return true
    }
  }
  false
}

// ---------------------------------------------------------------------------
// Rule: unused-where
// ---------------------------------------------------------------------------

///|
/// Flag a `where x = e` binding when neither the function body nor any
/// of its sibling `where` bindings reference `x`.
fn lint_unused_where(
  where_binds : Array[(String, @ast.Expr)],
  body : @ast.Expr,
  diags : Array[@diagnostics.Diagnostic]
) -> Unit {
  for b in where_binds {
    let (name, e) = b
    if is_ignored(name) {
      continue
    }
    if !expr_references(body, name) &&
      !any_binding_references(where_binds, name, e) {
      diags.push(
        @diagnostics.Diagnostic::warning(
          "Unused where binding `" + name + "`",
        )
          .with_code(LintRuleId::UnusedWhere.code())
          .with_label(
            @diagnostics.Label::primary(e.span(), "bound but never used"),
          )
          .with_help(
            "remove the binding, rename it with a leading `_`, or use it in the body",
          ),
      )
    }
  }
}

// ---------------------------------------------------------------------------
// Rule: redundant-wildcard
// ---------------------------------------------------------------------------

///|
/// Flag a match arm that comes after a wildcard `_` — every later arm is
/// unreachable.
fn lint_redundant_wildcard(
  arms : Array[(@ast.Pat, @ast.Expr)],
  diags : Array[@diagnostics.Diagnostic]
) -> Unit {
  let mut saw_wildcard = false
  for a in arms {
    let (pat, _) = a
    if saw_wildcard {
      diags.push(
        @diagnostics.Diagnostic::warning(
          "Unreachable match arm after a wildcard",
        )
          .with_code(LintRuleId::RedundantWildcard.code())
          .with_label(
            @diagnostics.Label::primary(pat.span(), "unreachable"),
          )
          .with_help(
            "move this arm above the wildcard, or remove the wildcard",
          ),
      )
    }
    if pat_is_wildcard(pat) {
      saw_wildcard = true
    }
  }
}

///|
fn pat_is_wildcard(pat : @ast.Pat) -> Bool {
  match pat {
    Wild(_) => true
    Paren(inner, _) => pat_is_wildcard(inner)
    _ => false
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

///|
/// Does `expr` contain a free reference to `name`? Conservative: doesn't
/// track shadowing — that's the job of the `shadowed-binding` rule.
fn expr_references(expr : @ast.Expr, name : String) -> Bool {
  match expr {
    Var(n, _) => n == name
    Con(_, _) | Lit(_, _) | OpSection(_, _) => false
    App(f, a, _) => expr_references(f, name) || expr_references(a, name)
    Infix(l, op, r, _) =>
      op == name ||
      expr_references(l, name) ||
      expr_references(r, name)
    Lambda(_, body, _) => expr_references(body, name)
    Let(binds, body, _) => {
      for b in binds {
        let (_, e) = b
        if expr_references(e, name) {
          return true
        }
      }
      expr_references(body, name)
    }
    Case(scrut, arms, _) => {
      if expr_references(scrut, name) {
        return true
      }
      for a in arms {
        let (_, e) = a
        if expr_references(e, name) {
          return true
        }
      }
      false
    }
    If(c, t, e, _) =>
      expr_references(c, name) ||
      expr_references(t, name) ||
      expr_references(e, name)
    Paren(inner, _) | Neg(inner, _) => expr_references(inner, name)
    FieldAccess(inner, _, _) => expr_references(inner, name)
    Tuple(es, _) | VecLit(es, _) => {
      for e in es {
        if expr_references(e, name) {
          return true
        }
      }
      false
    }
    Record(fields, _) => {
      for f in fields {
        let (_, e) = f
        if expr_references(e, name) {
          return true
        }
      }
      false
    }
    Do(stmts, _) => {
      for s in stmts {
        match s {
          Bind(_, e, _) | ExprStmt(e, _) | LetStmt(_, e, _) =>
            if expr_references(e, name) {
              return true
            }
        }
      }
      false
    }
  }
}

///|
/// Identifiers conventionally suppressed: leading underscore (`_x`) and
/// the wildcard placeholder (`_`).
fn is_ignored(name : String) -> Bool {
  if name.length() == 0 {
    return true
  }
  name.unsafe_charcode_at(0) == 0x5F // '_'
}

///|
/// Cheap pointer-style identity check on AST expressions: two exprs are
/// "the same" if their spans match exactly. Used to skip the `e` that
/// produced a binding when checking whether a later binding references
/// the same name (a real shadow vs. reference disambiguation).
fn same_expr(a : @ast.Expr, b : @ast.Expr) -> Bool {
  let sa = a.span()
  let sb = b.span()
  sa.start == sb.start && sa.end == sb.end
}

///|
/// Insertion-sort the diagnostics by the start offset of their first
/// label. Stable enough for the small lists the linter typically
/// produces and keeps reports deterministic across runs.
fn sort_by_span(diags : Array[@diagnostics.Diagnostic]) -> Unit {
  let n = diags.length()
  let mut i = 1
  while i < n {
    let cur = diags[i]
    let cur_off = first_label_offset(cur)
    let mut j = i - 1
    while j >= 0 && first_label_offset(diags[j]) > cur_off {
      diags[j + 1] = diags[j]
      j = j - 1
    }
    diags[j + 1] = cur
    i = i + 1
  }
}

///|
fn first_label_offset(d : @diagnostics.Diagnostic) -> UInt {
  if d.labels.length() == 0 {
    return 0U
  }
  d.labels[0].span.start
}
