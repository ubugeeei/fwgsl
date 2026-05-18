///|
/// Type inference engine.
pub struct InferEngine {
  mut next_var : TyVarId
  subst : Substitution
  diagnostics : @diagnostics.DiagnosticSink
}

///|
pub fn InferEngine::new() -> InferEngine {
  InferEngine::{
    next_var: 0,
    subst: Substitution::new(),
    diagnostics: @diagnostics.DiagnosticSink::new(),
  }
}

///|
pub fn InferEngine::subst(self : InferEngine) -> Substitution {
  self.subst
}

///|
pub fn InferEngine::diagnostics(self : InferEngine) -> @diagnostics.DiagnosticSink {
  self.diagnostics
}

///|
/// Generate a fresh type variable.
pub fn InferEngine::fresh_var(self : InferEngine) -> Ty {
  let v = self.next_var
  self.next_var = self.next_var + 1
  Var(v)
}

///|
pub fn InferEngine::reserve_above(
  self : InferEngine,
  next_var : TyVarId
) -> Unit {
  if next_var > self.next_var {
    self.next_var = next_var
  }
}

///|
/// Unify two types.
pub fn InferEngine::unify(
  self : InferEngine,
  a : Ty,
  b : Ty,
  span : @span.Span
) -> Unit {
  let a = normalize_type_aliases(a.apply_subst(self.subst))
  let b = normalize_type_aliases(b.apply_subst(self.subst))
  match (a, b) {
    (Error, _) => ()
    (_, Error) => ()
    (Var(v1), Var(v2)) =>
      if v1 == v2 {
        ()
      } else {
        self.subst.insert(v1, Var(v2))
      }
    (Var(v), ty) => self.bind_var(v, ty, span)
    (ty, Var(v)) => self.bind_var(v, ty, span)
    (Con(x), Con(y)) =>
      if x != y {
        self.report_mismatch(Con(x), Con(y), span)
      }
    (Nat(x), Nat(y)) =>
      if x != y {
        self.report_mismatch(Nat(x), Nat(y), span)
      }
    (Arrow(a1, a2), Arrow(b1, b2)) => {
      self.unify(a1, b1, span)
      self.unify(a2, b2, span)
    }
    (App(f1, a1), App(f2, a2)) => {
      self.unify(f1, f2, span)
      self.unify(a1, a2, span)
    }
    (x, y) => self.report_mismatch(x, y, span)
  }
}

///|
fn InferEngine::bind_var(
  self : InferEngine,
  v : TyVarId,
  ty : Ty,
  span : @span.Span
) -> Unit {
  if ty.contains_var(v) {
    self.diagnostics.push(
      @diagnostics.Diagnostic::error(
        "Infinite type: t" + v.to_string() + " ~ " + ty.display(),
      )
        .with_label(@diagnostics.Label::primary(span, "occurs here"))
        .with_help(
          "introduce a concrete argument or result type to break the self-reference",
        ),
    )
  } else {
    self.subst.insert(v, ty)
  }
}

///|
fn InferEngine::report_mismatch(
  self : InferEngine,
  a : Ty,
  b : Ty,
  span : @span.Span
) -> Unit {
  self.diagnostics.push(
    @diagnostics.Diagnostic::error(
      "Type mismatch: " + a.display() + " vs " + b.display(),
    )
      .with_label(@diagnostics.Label::primary(span, "type mismatch"))
      .with_help(
        "add a type annotation or convert one side so both types agree",
      ),
  )
}

///|
/// Instantiate a type scheme with fresh variables.
pub fn InferEngine::instantiate(self : InferEngine, scheme : Scheme) -> Ty {
  let subst = Substitution::new()
  for v in scheme.vars {
    subst.insert(v, self.fresh_var())
  }
  scheme.ty.apply_subst(subst)
}

///|
/// Generalize a type over variables not free in the environment.
pub fn InferEngine::generalize(
  self : InferEngine,
  env : TypeEnv,
  ty : Ty
) -> Scheme {
  let ty = ty.apply_subst(self.subst)
  let env_vars = env.free_vars()
  let ty_vars = ty.free_vars()
  let gen_vars : Array[TyVarId] = Array::new()
  for v in ty_vars {
    if not(contains_var_id(env_vars, v)) {
      gen_vars.push(v)
    }
  }
  Scheme::poly(gen_vars, ty)
}

///|
/// Finalize a type by applying all accumulated substitutions.
pub fn InferEngine::finalize(self : InferEngine, ty : Ty) -> Ty {
  ty.apply_subst(self.subst)
}

// ---------------------------------------------------------------------------
// Constructor info
// ---------------------------------------------------------------------------

///|
pub(all) struct ConstructorInfo {
  type_name : String
  tag : UInt
  scheme_vars : Array[TyVarId]
  fields : ConstructorFields
  result_ty : Ty
}

///|
pub fn ConstructorInfo::instantiate(
  self : ConstructorInfo,
  engine : InferEngine
) -> ConstructorInfo {
  let subst = Substitution::new()
  for v in self.scheme_vars {
    subst.insert(v, engine.fresh_var())
  }
  ConstructorInfo::{
    type_name: self.type_name,
    tag: self.tag,
    scheme_vars: Array::new(),
    fields: self.fields.apply_subst(subst),
    result_ty: self.result_ty.apply_subst(subst),
  }
}

///|
pub(all) enum ConstructorFields {
  Positional(Array[Ty])
  RecordFields(Array[(String, Ty)])
  Empty
}

///|
pub fn ConstructorFields::apply_subst(
  self : ConstructorFields,
  subst : Substitution
) -> ConstructorFields {
  match self {
    Positional(fields) => {
      let out : Array[Ty] = Array::new()
      for ty in fields {
        out.push(ty.apply_subst(subst))
      }
      Positional(out)
    }
    RecordFields(fields) => {
      let out : Array[(String, Ty)] = Array::new()
      for pair in fields {
        let (name, ty) = pair
        out.push((name, ty.apply_subst(subst)))
      }
      RecordFields(out)
    }
    Empty => Empty
  }
}
