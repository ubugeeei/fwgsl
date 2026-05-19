///|
/// AST -> HIR lowering for fwgsl.
///
/// Walks the parser AST and produces a fully type-annotated `HirProgram`
/// by re-running Hindley-Milner inference while simultaneously building
/// HIR nodes. Requires a `SemanticAnalyzer` whose `analyze()` has already
/// populated constructor and data-type information.

///|
pub struct AstLowering {
  env : @typechecker.TypeEnv
  engine : @typechecker.InferEngine
  constructors : Map[String, @typechecker.ConstructorInfo]
  data_types : Map[String, @semantic.DataTypeInfo]
}

///|
pub fn AstLowering::new(sa : @semantic.SemanticAnalyzer) -> AstLowering {
  let engine = @typechecker.InferEngine::new()
  match sa.engine().subst() {
    _ => ()
  }
  // Snapshot the analyzer's environment and constructor tables.
  let env = clone_env(sa.env())
  let constructors = clone_ctors(sa.constructors())
  let data_types = clone_data_types(sa.data_types())
  // Make sure freshly-allocated tyvars don't collide with anything the
  // semantic analyzer already produced.
  match sa.env().max_var_id() {
    Some(max) => engine.reserve_above(max + 1)
    None => ()
  }
  let lowering = AstLowering::{ env, engine, constructors, data_types }
  lowering.add_builtins()
  lowering
}

///|
pub fn AstLowering::diagnostics(self : AstLowering) -> @diagnostics.DiagnosticSink {
  self.engine.diagnostics()
}

///|
pub fn AstLowering::has_errors(self : AstLowering) -> Bool {
  self.engine.diagnostics().has_errors()
}

///|
fn clone_env(src : @typechecker.TypeEnv) -> @typechecker.TypeEnv {
  let dst = @typechecker.TypeEnv::new()
  src.bindings.each(fn(k, v) { dst.insert(k, v) })
  dst
}

///|
fn clone_ctors(
  src : Map[String, @typechecker.ConstructorInfo]
) -> Map[String, @typechecker.ConstructorInfo] {
  let dst : Map[String, @typechecker.ConstructorInfo] = Map::new()
  src.each(fn(k, v) { dst[k] = v })
  dst
}

///|
fn clone_data_types(
  src : Map[String, @semantic.DataTypeInfo]
) -> Map[String, @semantic.DataTypeInfo] {
  let dst : Map[String, @semantic.DataTypeInfo] = Map::new()
  src.each(fn(k, v) { dst[k] = v })
  dst
}

///|
fn AstLowering::add_builtins(self : AstLowering) -> Unit {
  // Re-register operator schemes so they get fresh tyvars allocated by
  // *this* engine rather than the semantic analyzer's.
  let a1 = fresh_var_id(self.engine)
  let numeric_binop = @typechecker.Scheme::poly(
    [a1],
    @typechecker.Ty::arrow(
      @typechecker.Ty::Var(a1),
      @typechecker.Ty::arrow(@typechecker.Ty::Var(a1), @typechecker.Ty::Var(a1)),
    ),
  )
  for op in ["+", "-", "*", "/", "%"] {
    self.env.insert(op, numeric_binop)
  }
  let a2 = fresh_var_id(self.engine)
  let cmp_binop = @typechecker.Scheme::poly(
    [a2],
    @typechecker.Ty::arrow(
      @typechecker.Ty::Var(a2),
      @typechecker.Ty::arrow(@typechecker.Ty::Var(a2), @typechecker.Ty::bool()),
    ),
  )
  for op in ["==", "/=", "<", ">", "<=", ">="] {
    self.env.insert(op, cmp_binop)
  }
  let bool_binop = @typechecker.Scheme::mono(
    @typechecker.Ty::arrow(
      @typechecker.Ty::bool(),
      @typechecker.Ty::arrow(@typechecker.Ty::bool(), @typechecker.Ty::bool()),
    ),
  )
  self.env.insert("&&", bool_binop)
  self.env.insert("||", bool_binop)
}

///|
fn fresh_var_id(engine : @typechecker.InferEngine) -> UInt {
  match engine.fresh_var() {
    Var(v) => v
    _ => 0
  }
}

///|
/// Lower a complete `Program` to a `HirProgram`.
pub fn AstLowering::lower_program(
  self : AstLowering,
  prog : @ast.Program
) -> @hir.HirProgram {
  // Pass 1: register data-type constructor schemes into the env.
  for d in prog.decls {
    match d {
      DataDecl(name, type_params, cons, _) =>
        self.register_data_constructor_schemes(name, type_params, cons)
      _ => ()
    }
  }
  // Pass 2: collect explicit type signatures.
  for d in prog.decls {
    match d {
      TypeSig(name, ty, _) => {
        let scheme = self.convert_syntax_type_scheme(ty)
        self.env.insert(name, scheme)
      }
      _ => ()
    }
  }
  let functions : Array[@hir.HirFunction] = Array::new()
  let data_types : Array[@hir.HirDataType] = Array::new()
  let entry_points : Array[@hir.HirEntryPoint] = Array::new()
  for d in prog.decls {
    match d {
      FunDecl(name, params, body, where_binds, span) =>
        match self.lower_fun_decl(name, params, body, where_binds, span) {
          Some(f) => functions.push(f)
          None => ()
        }
      EntryPoint(attrs, name, params, body, span) =>
        match self.lower_entry_point(attrs, name, params, body, span) {
          Some(ep) => entry_points.push(ep)
          None => ()
        }
      DataDecl(name, _, ctors, _) =>
        data_types.push(self.lower_data_decl(name, ctors))
      TypeSig(_, _, _) | TypeAlias(_, _, _, _) => ()
    }
  }
  @hir.HirProgram::{ functions, data_types, entry_points }
}

///|
fn AstLowering::register_data_constructor_schemes(
  self : AstLowering,
  name : String,
  type_params : Array[String],
  cons : Array[@ast.ConDecl]
) -> Unit {
  let scope : Map[String, UInt] = Map::new()
  for p in type_params {
    scope[p] = fresh_var_id(self.engine)
  }
  let scheme_vars = scope_var_ids(scope, type_params)
  let result_ty = apply_type_params(name, type_params, scope)
  for con in cons {
    let con_ty = match con.fields {
      Empty => result_ty
      Positional(fields) => {
        let mut ty = result_ty
        let mut i = fields.length() - 1
        while i >= 0 {
          let ft = self.convert_syntax_type_with_scope(fields[i], scope)
          ty = @typechecker.Ty::arrow(ft, ty)
          i = i - 1
        }
        ty
      }
      RecordFields(fields) => {
        let mut ty = result_ty
        let mut i = fields.length() - 1
        while i >= 0 {
          let (_, field_ty) = fields[i]
          let ft = self.convert_syntax_type_with_scope(field_ty, scope)
          ty = @typechecker.Ty::arrow(ft, ty)
          i = i - 1
        }
        ty
      }
    }
    self.env.insert(con.name, @typechecker.Scheme::poly(scheme_vars, con_ty))
  }
}

///|
fn scope_var_ids(
  scope : Map[String, UInt],
  type_params : Array[String]
) -> Array[UInt] {
  let out : Array[UInt] = Array::new()
  for p in type_params {
    match scope.get(p) {
      Some(v) => out.push(v)
      None => ()
    }
  }
  out
}

///|
fn apply_type_params(
  name : String,
  type_params : Array[String],
  scope : Map[String, UInt]
) -> @typechecker.Ty {
  let mut ty : @typechecker.Ty = @typechecker.Ty::Con(name)
  for p in type_params {
    match scope.get(p) {
      Some(v) => ty = @typechecker.Ty::App(ty, @typechecker.Ty::Var(v))
      None => ()
    }
  }
  ty
}

///|
fn AstLowering::convert_syntax_type_scheme(
  self : AstLowering,
  ty : @ast.Type
) -> @typechecker.Scheme {
  let scope : Map[String, UInt] = Map::new()
  let t = self.convert_syntax_type_with_scope(ty, scope)
  let vars : Array[UInt] = Array::new()
  scope.each(fn(_, v) { vars.push(v) })
  @typechecker.Scheme::poly(vars, t)
}

///|
fn AstLowering::convert_syntax_type_with_scope(
  self : AstLowering,
  ty : @ast.Type,
  scope : Map[String, UInt]
) -> @typechecker.Ty {
  let raw = match ty {
    Con(name, _) => @typechecker.Ty::Con(name)
    Var(name, _) => {
      let id = match scope.get(name) {
        Some(v) => v
        None => {
          let v = fresh_var_id(self.engine)
          scope[name] = v
          v
        }
      }
      @typechecker.Ty::Var(id)
    }
    Nat(n, _) => @typechecker.Ty::Nat(n)
    Arrow(a, b, _) =>
      @typechecker.Ty::arrow(
        self.convert_syntax_type_with_scope(a, scope),
        self.convert_syntax_type_with_scope(b, scope),
      )
    App(f, a, _) =>
      @typechecker.Ty::App(
        self.convert_syntax_type_with_scope(f, scope),
        self.convert_syntax_type_with_scope(a, scope),
      )
    Paren(inner, _) => self.convert_syntax_type_with_scope(inner, scope)
    Tuple(elems, _) =>
      if elems.length() == 0 {
        @typechecker.Ty::unit()
      } else {
        self.convert_syntax_type_with_scope(elems[0], scope)
      }
    Unit(_) => @typechecker.Ty::unit()
  }
  @typechecker.normalize_type_aliases(raw)
}

///|
fn AstLowering::convert_syntax_type_pure(
  self : AstLowering,
  ty : @ast.Type
) -> @typechecker.Ty {
  let _ = self
  convert_syntax_type_pure_inner(ty)
}

///|
fn convert_syntax_type_pure_inner(ty : @ast.Type) -> @typechecker.Ty {
  let raw = match ty {
    Con(name, _) => @typechecker.Ty::Con(name)
    Var(name, _) => @typechecker.Ty::Con(name)
    Nat(n, _) => @typechecker.Ty::Nat(n)
    Arrow(a, b, _) =>
      @typechecker.Ty::arrow(
        convert_syntax_type_pure_inner(a),
        convert_syntax_type_pure_inner(b),
      )
    App(f, a, _) =>
      @typechecker.Ty::App(
        convert_syntax_type_pure_inner(f),
        convert_syntax_type_pure_inner(a),
      )
    Paren(inner, _) => convert_syntax_type_pure_inner(inner)
    Tuple(elems, _) =>
      if elems.length() == 0 {
        @typechecker.Ty::unit()
      } else {
        convert_syntax_type_pure_inner(elems[0])
      }
    Unit(_) => @typechecker.Ty::unit()
  }
  @typechecker.normalize_type_aliases(raw)
}

///|
fn AstLowering::lower_data_decl(
  self : AstLowering,
  name : String,
  cons : Array[@ast.ConDecl]
) -> @hir.HirDataType {
  let hir_cons : Array[@hir.HirConstructor] = Array::new()
  let mut tag : UInt = 0
  for con in cons {
    let fields : Array[(String, @typechecker.Ty)] = Array::new()
    match con.fields {
      Empty => ()
      Positional(tys) => {
        let mut i = 0
        for t in tys {
          fields.push(("field" + i.to_string(), self.convert_syntax_type_pure(t)))
          i = i + 1
        }
      }
      RecordFields(rs) =>
        for r in rs {
          let (n, t) = r
          fields.push((n, self.convert_syntax_type_pure(t)))
        }
    }
    hir_cons.push(@hir.HirConstructor::{ name: con.name, tag, fields })
    tag = tag + 1
  }
  @hir.HirDataType::{ name, constructors: hir_cons }
}

///|
fn AstLowering::lower_fun_decl(
  self : AstLowering,
  name : String,
  params : Array[@ast.Pat],
  body : @ast.Expr,
  where_binds : Array[(String, @ast.Expr)],
  span : @span.Span
) -> @hir.HirFunction? {
  let local_env = clone_env(self.env)
  let hir_params : Array[(String, @typechecker.Ty)] = Array::new()
  let param_types : Array[@typechecker.Ty] = Array::new()
  for pat in params {
    let ty = self.engine.fresh_var()
    let pname = pat_name(pat)
    self.bind_pattern(pat, ty, local_env)
    param_types.push(ty)
    hir_params.push((pname, ty))
  }
  let body_ast = desugar_where(body, where_binds, span)
  let (hir_body, body_ty) = self.lower_expr(body_ast, local_env)
  let mut fun_ty = body_ty
  let mut i = param_types.length() - 1
  while i >= 0 {
    fun_ty = @typechecker.Ty::arrow(param_types[i], fun_ty)
    i = i - 1
  }
  match self.env.lookup(name) {
    Some(scheme) => {
      let declared = self.engine.instantiate(scheme)
      self.engine.unify(fun_ty, declared, span)
    }
    None => ()
  }
  let finalized_params : Array[(String, @typechecker.Ty)] = Array::new()
  for pair in hir_params {
    let (pname, pty) = pair
    finalized_params.push((pname, self.engine.finalize(pty)))
  }
  let return_ty = self.engine.finalize(body_ty)
  let body = self.finalize_expr(hir_body)
  Some(
    @hir.HirFunction::{
      name,
      params: finalized_params,
      return_ty,
      body,
      span,
    },
  )
}

///|
fn AstLowering::lower_entry_point(
  self : AstLowering,
  attributes : Array[@ast.Attribute],
  name : String,
  params : Array[@ast.Pat],
  body : @ast.Expr,
  span : @span.Span
) -> @hir.HirEntryPoint? {
  let local_env = clone_env(self.env)
  let hir_params : Array[(String, @typechecker.Ty)] = Array::new()
  let param_types : Array[@typechecker.Ty] = Array::new()
  for pat in params {
    let ty = self.engine.fresh_var()
    let pname = pat_name(pat)
    self.bind_pattern(pat, ty, local_env)
    param_types.push(ty)
    hir_params.push((pname, ty))
  }
  let (hir_body, body_ty) = self.lower_expr(body, local_env)
  let mut fun_ty = body_ty
  let mut i = param_types.length() - 1
  while i >= 0 {
    fun_ty = @typechecker.Ty::arrow(param_types[i], fun_ty)
    i = i - 1
  }
  match self.env.lookup(name) {
    Some(scheme) => {
      let declared = self.engine.instantiate(scheme)
      self.engine.unify(fun_ty, declared, span)
    }
    None => ()
  }
  let finalized_params : Array[(String, @typechecker.Ty)] = Array::new()
  for pair in hir_params {
    let (pname, pty) = pair
    finalized_params.push((pname, self.engine.finalize(pty)))
  }
  let return_ty = self.engine.finalize(body_ty)
  let body = self.finalize_expr(hir_body)
  let hir_attrs : Array[@hir.HirAttribute] = Array::new()
  for a in attributes {
    hir_attrs.push(@hir.HirAttribute::{ name: a.name, args: a.args })
  }
  Some(
    @hir.HirEntryPoint::{
      name,
      attributes: hir_attrs,
      params: finalized_params,
      return_ty,
      body,
      span,
    },
  )
}

///|
/// Lower an expression and return (HIR expression, inferred type).
fn AstLowering::lower_expr(
  self : AstLowering,
  expr : @ast.Expr,
  env : @typechecker.TypeEnv
) -> (@hir.HirExpr, @typechecker.Ty) {
  match expr {
    Lit(lit, span) => {
      let (hir_lit, ty) = self.lower_lit(lit)
      (@hir.HirExpr::Lit(hir_lit, ty, span), ty)
    }
    Var(name, span) => {
      let ty : @typechecker.Ty = match env.lookup(name) {
        Some(scheme) => self.engine.instantiate(scheme)
        None => @typechecker.Ty::Error
      }
      (@hir.HirExpr::Var(name, ty, span), ty)
    }
    Con(name, span) => {
      let ty : @typechecker.Ty = match env.lookup(name) {
        Some(scheme) => self.engine.instantiate(scheme)
        None => @typechecker.Ty::Error
      }
      match self.constructors.get(name) {
        Some(info) =>
          match info.fields {
            Empty =>
              (
                @hir.HirExpr::ConstructorCall(
                  name,
                  info.tag,
                  Array::new(),
                  ty,
                  span,
                ),
                ty,
              )
            _ => (@hir.HirExpr::Var(name, ty, span), ty)
          }
        None => (@hir.HirExpr::Var(name, ty, span), ty)
      }
    }
    App(func, arg, span) => {
      let (hir_func, func_ty) = self.lower_expr(func, env)
      let (hir_arg, arg_ty) = self.lower_expr(arg, env)
      let ret_ty = self.engine.fresh_var()
      let expected = @typechecker.Ty::arrow(arg_ty, ret_ty)
      self.engine.unify(func_ty, expected, span)
      (@hir.HirExpr::App(hir_func, hir_arg, ret_ty, span), ret_ty)
    }
    Infix(lhs, op, rhs, span) =>
      match @hir.BinOp::parse(op) {
        Some(binop) => {
          let (hir_lhs, lhs_ty) = self.lower_expr(lhs, env)
          let (hir_rhs, rhs_ty) = self.lower_expr(rhs, env)
          let op_ty : @typechecker.Ty = match env.lookup(op) {
            Some(scheme) => self.engine.instantiate(scheme)
            None => @typechecker.Ty::Error
          }
          let ret_ty = self.engine.fresh_var()
          self.engine.unify(
            op_ty,
            @typechecker.Ty::arrow(
              lhs_ty,
              @typechecker.Ty::arrow(rhs_ty, ret_ty),
            ),
            span,
          )
          (@hir.HirExpr::BinOpE(binop, hir_lhs, hir_rhs, ret_ty, span), ret_ty)
        }
        None => {
          let op_ty : @typechecker.Ty = match env.lookup(op) {
            Some(scheme) => self.engine.instantiate(scheme)
            None => @typechecker.Ty::Error
          }
          let (hir_lhs, lhs_ty) = self.lower_expr(lhs, env)
          let (hir_rhs, rhs_ty) = self.lower_expr(rhs, env)
          let ret_ty = self.engine.fresh_var()
          self.engine.unify(
            op_ty,
            @typechecker.Ty::arrow(
              lhs_ty,
              @typechecker.Ty::arrow(rhs_ty, ret_ty),
            ),
            span,
          )
          let op_expr = @hir.HirExpr::Var(op, op_ty, span)
          let app1_ty = self.engine.fresh_var()
          let app1 = @hir.HirExpr::App(op_expr, hir_lhs, app1_ty, span)
          (@hir.HirExpr::App(app1, hir_rhs, ret_ty, span), ret_ty)
        }
      }
    Lambda(pats, body, _) => {
      let local = clone_env(env)
      let param_types : Array[@typechecker.Ty] = Array::new()
      for pat in pats {
        let ty = self.engine.fresh_var()
        self.bind_pattern(pat, ty, local)
        param_types.push(ty)
      }
      let (hir_body, body_ty) = self.lower_expr(body, local)
      let mut result_ty = body_ty
      let mut i = param_types.length() - 1
      while i >= 0 {
        result_ty = @typechecker.Ty::arrow(param_types[i], result_ty)
        i = i - 1
      }
      (hir_body, result_ty)
    }
    Let(binds, body, span) => {
      let local = clone_env(env)
      let hir_binds : Array[(String, @hir.HirExpr)] = Array::new()
      for pair in binds {
        let (name, e) = pair
        let (hir_e, ty) = self.lower_expr(e, local)
        local.insert(name, @typechecker.Scheme::mono(ty))
        hir_binds.push((name, hir_e))
      }
      let (hir_body, body_ty) = self.lower_expr(body, local)
      (@hir.HirExpr::Let(hir_binds, hir_body, body_ty, span), body_ty)
    }
    Case(scrutinee, arms, span) => {
      let (hir_scrut, scrut_ty) = self.lower_expr(scrutinee, env)
      let result_ty = self.engine.fresh_var()
      let hir_arms : Array[@hir.HirCaseArm] = Array::new()
      for arm in arms {
        let (pat, body) = arm
        let arm_env = clone_env(env)
        self.bind_pattern(pat, scrut_ty, arm_env)
        let hir_pattern = self.lower_pattern(pat, scrut_ty)
        let (hir_body, body_ty) = self.lower_expr(body, arm_env)
        self.engine.unify(result_ty, body_ty, span)
        hir_arms.push(@hir.HirCaseArm::{ pattern: hir_pattern, body: hir_body })
      }
      (@hir.HirExpr::Case(hir_scrut, hir_arms, result_ty, span), result_ty)
    }
    If(cond, then_expr, else_expr, span) => {
      let (hir_cond, cond_ty) = self.lower_expr(cond, env)
      self.engine.unify(cond_ty, @typechecker.Ty::bool(), span)
      let (hir_then, then_ty) = self.lower_expr(then_expr, env)
      let (hir_else, else_ty) = self.lower_expr(else_expr, env)
      self.engine.unify(then_ty, else_ty, span)
      (
        @hir.HirExpr::If(hir_cond, hir_then, hir_else, then_ty, span),
        then_ty,
      )
    }
    Paren(inner, _) => self.lower_expr(inner, env)
    Tuple(elems, span) =>
      if elems.length() == 0 {
        (
          @hir.HirExpr::Lit(@hir.HirLit::Int(0L), @typechecker.Ty::unit(), span),
          @typechecker.Ty::unit(),
        )
      } else {
        self.lower_expr(elems[0], env)
      }
    Record(fields, span) =>
      if fields.length() > 0 {
        let (_, e) = fields[0]
        self.lower_expr(e, env)
      } else {
        (
          @hir.HirExpr::Lit(@hir.HirLit::Int(0L), @typechecker.Ty::unit(), span),
          @typechecker.Ty::unit(),
        )
      }
    FieldAccess(inner, field, span) => {
      let (hir_inner, _inner_ty) = self.lower_expr(inner, env)
      let result_ty = self.engine.fresh_var()
      (@hir.HirExpr::FieldAccess(hir_inner, field, result_ty, span), result_ty)
    }
    VecLit(elems, span) => {
      let scalar_ty = self.engine.fresh_var()
      let mut total : UInt64 = 0UL
      let hir_args : Array[@hir.HirExpr] = Array::new()
      for elem in elems {
        let (hir_elem, elem_ty) = self.lower_expr(elem, env)
        total = total + 1UL
        self.engine.unify(scalar_ty, elem_ty, span)
        hir_args.push(hir_elem)
      }
      let n : UInt64 = if total < 2UL { 2UL } else if total > 4UL { 4UL } else { total }
      let vec_name = "$vec" + n.to_string()
      let result_ty = @typechecker.Ty::app(
        @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Nat(n)),
        scalar_ty,
      )
      let mut e : @hir.HirExpr = @hir.HirExpr::Var(vec_name, result_ty, span)
      for arg in hir_args {
        e = @hir.HirExpr::App(e, arg, result_ty, span)
      }
      (e, result_ty)
    }
    OpSection(op, span) => {
      let ty : @typechecker.Ty = match env.lookup(op) {
        Some(scheme) => self.engine.instantiate(scheme)
        None => @typechecker.Ty::Error
      }
      (@hir.HirExpr::Var(op, ty, span), ty)
    }
    Neg(inner, span) => {
      let (hir_inner, inner_ty) = self.lower_expr(inner, env)
      let zero = @hir.HirExpr::Lit(@hir.HirLit::Int(0L), inner_ty, span)
      (
        @hir.HirExpr::BinOpE(@hir.BinOp::Sub, zero, hir_inner, inner_ty, span),
        inner_ty,
      )
    }
    Do(stmts, span) => {
      let local = clone_env(env)
      let hir_binds : Array[(String, @hir.HirExpr)] = Array::new()
      let mut last_expr : @hir.HirExpr? = None
      for stmt in stmts {
        match stmt {
          ExprStmt(e, _) => {
            let (he, _) = self.lower_expr(e, local)
            last_expr = Some(he)
          }
          Bind(name, e, _) => {
            let (he, _) = self.lower_expr(e, local)
            let inner_ty = self.engine.fresh_var()
            local.insert(name, @typechecker.Scheme::mono(inner_ty))
            hir_binds.push((name, he))
            last_expr = None
          }
          LetStmt(name, e, _) => {
            let (he, ty) = self.lower_expr(e, local)
            local.insert(name, @typechecker.Scheme::mono(ty))
            hir_binds.push((name, he))
            last_expr = None
          }
        }
      }
      let body : @hir.HirExpr = match last_expr {
        Some(e) => e
        None => @hir.HirExpr::Lit(@hir.HirLit::Int(0L), @typechecker.Ty::unit(), span)
      }
      let body_ty = body.ty()
      if hir_binds.length() == 0 {
        (body, body_ty)
      } else {
        (@hir.HirExpr::Let(hir_binds, body, body_ty, span), body_ty)
      }
    }
  }
}

///|
fn AstLowering::lower_pattern(
  self : AstLowering,
  pat : @ast.Pat,
  scrut_ty : @typechecker.Ty
) -> @hir.HirPattern {
  match pat {
    Wild(_) => @hir.HirPattern::Wild
    Var(name, _) => @hir.HirPattern::Var(name, self.engine.finalize(scrut_ty))
    Con(name, sub_pats, _) =>
      match self.constructors.get(name) {
        Some(info) => {
          let inst = info.instantiate(self.engine)
          let sub_hir : Array[@hir.HirPattern] = Array::new()
          match inst.fields {
            Positional(field_tys) => {
              let n = if field_tys.length() < sub_pats.length() {
                field_tys.length()
              } else {
                sub_pats.length()
              }
              let mut i = 0
              while i < n {
                sub_hir.push(self.lower_pattern(sub_pats[i], field_tys[i]))
                i = i + 1
              }
            }
            RecordFields(fields) => {
              let n = if fields.length() < sub_pats.length() {
                fields.length()
              } else {
                sub_pats.length()
              }
              let mut i = 0
              while i < n {
                let (_, fty) = fields[i]
                sub_hir.push(self.lower_pattern(sub_pats[i], fty))
                i = i + 1
              }
            }
            Empty => ()
          }
          @hir.HirPattern::Constructor(name, inst.tag, sub_hir)
        }
        None => @hir.HirPattern::Wild
      }
    Lit(lit, _) => {
      let (hir_lit, _) = self.lower_lit(lit)
      @hir.HirPattern::PatLit(hir_lit)
    }
    Paren(inner, _) => self.lower_pattern(inner, scrut_ty)
    Tuple(pats, _) =>
      if pats.length() > 0 {
        self.lower_pattern(pats[0], scrut_ty)
      } else {
        @hir.HirPattern::Wild
      }
    Record(con_name, fields, _) =>
      match self.constructors.get(con_name) {
        Some(info) => {
          let inst = info.instantiate(self.engine)
          let sub_pats : Array[@hir.HirPattern] = Array::new()
          match inst.fields {
            RecordFields(con_fields) =>
              for field in fields {
                let (fname, maybe_pat) = field
                let field_ty = lookup_field_ty(con_fields, fname)
                match maybe_pat {
                  Some(p) => sub_pats.push(self.lower_pattern(p, field_ty))
                  None =>
                    sub_pats.push(
                      @hir.HirPattern::Var(fname, self.engine.finalize(field_ty)),
                    )
                }
              }
            _ => ()
          }
          @hir.HirPattern::Constructor(con_name, inst.tag, sub_pats)
        }
        None => @hir.HirPattern::Wild
      }
    As(_, inner, _) => self.lower_pattern(inner, scrut_ty)
  }
}

///|
fn lookup_field_ty(
  fields : Array[(String, @typechecker.Ty)],
  name : String
) -> @typechecker.Ty {
  for f in fields {
    let (n, ty) = f
    if n == name {
      return ty
    }
  }
  @typechecker.Ty::Error
}

///|
fn AstLowering::lower_lit(
  self : AstLowering,
  lit : @ast.Lit
) -> (@hir.HirLit, @typechecker.Ty) {
  let _ = self
  match lit {
    Int(v) => (@hir.HirLit::Int(v), @typechecker.Ty::i32())
    Float(v) => (@hir.HirLit::Float(v), @typechecker.Ty::f32())
    StringLit(_) => (@hir.HirLit::Int(0L), @typechecker.Ty::Con("String"))
    Char(_) => (@hir.HirLit::Int(0L), @typechecker.Ty::Con("Char"))
  }
}

///|
fn AstLowering::bind_pattern(
  self : AstLowering,
  pat : @ast.Pat,
  ty : @typechecker.Ty,
  env : @typechecker.TypeEnv
) -> Unit {
  match pat {
    Var(name, _) => env.insert(name, @typechecker.Scheme::mono(ty))
    Wild(_) => ()
    Con(name, sub_pats, span) =>
      match self.constructors.get(name) {
        Some(info) => {
          let inst = info.instantiate(self.engine)
          self.engine.unify(ty, inst.result_ty, span)
          match inst.fields {
            Positional(field_tys) => {
              let n = if field_tys.length() < sub_pats.length() {
                field_tys.length()
              } else {
                sub_pats.length()
              }
              let mut i = 0
              while i < n {
                self.bind_pattern(sub_pats[i], field_tys[i], env)
                i = i + 1
              }
            }
            Empty => ()
            RecordFields(fields) => {
              let n = if fields.length() < sub_pats.length() {
                fields.length()
              } else {
                sub_pats.length()
              }
              let mut i = 0
              while i < n {
                let (_, fty) = fields[i]
                self.bind_pattern(sub_pats[i], fty, env)
                i = i + 1
              }
            }
          }
        }
        None => ()
      }
    Lit(lit, span) => {
      let lit_ty : @typechecker.Ty = match lit {
        Int(_) => @typechecker.Ty::i32()
        Float(_) => @typechecker.Ty::f32()
        StringLit(_) => @typechecker.Ty::Con("String")
        Char(_) => @typechecker.Ty::Con("Char")
      }
      self.engine.unify(ty, lit_ty, span)
    }
    Paren(inner, _) => self.bind_pattern(inner, ty, env)
    Tuple(pats, _) =>
      if pats.length() > 0 {
        self.bind_pattern(pats[0], ty, env)
      }
    Record(con_name, fields, span) =>
      match self.constructors.get(con_name) {
        Some(info) => {
          let inst = info.instantiate(self.engine)
          self.engine.unify(ty, inst.result_ty, span)
          match inst.fields {
            RecordFields(con_fields) =>
              for field in fields {
                let (fname, maybe_pat) = field
                let field_ty = lookup_field_ty(con_fields, fname)
                match maybe_pat {
                  Some(p) => self.bind_pattern(p, field_ty, env)
                  None =>
                    env.insert(fname, @typechecker.Scheme::mono(field_ty))
                }
              }
            _ => ()
          }
        }
        None => ()
      }
    As(name, inner, _) => {
      env.insert(name, @typechecker.Scheme::mono(ty))
      self.bind_pattern(inner, ty, env)
    }
  }
}

///|
fn desugar_where(
  body : @ast.Expr,
  where_binds : Array[(String, @ast.Expr)],
  span : @span.Span
) -> @ast.Expr {
  if where_binds.length() == 0 {
    body
  } else {
    @ast.Expr::Let(where_binds, body, span)
  }
}

///|
fn pat_name(pat : @ast.Pat) -> String {
  match pat {
    Var(name, _) => name
    Wild(_) => "_"
    Paren(inner, _) => pat_name(inner)
    As(name, _, _) => name
    _ => "_"
  }
}

///|
/// Walk an HIR expression and apply the engine's substitution to every
/// embedded type, so the tree downstream consumers see is fully
/// resolved.
fn AstLowering::finalize_expr(
  self : AstLowering,
  expr : @hir.HirExpr
) -> @hir.HirExpr {
  match expr {
    Lit(lit, ty, span) => @hir.HirExpr::Lit(lit, self.engine.finalize(ty), span)
    Var(name, ty, span) =>
      @hir.HirExpr::Var(name, self.engine.finalize(ty), span)
    App(func, arg, ty, span) =>
      @hir.HirExpr::App(
        self.finalize_expr(func),
        self.finalize_expr(arg),
        self.engine.finalize(ty),
        span,
      )
    Let(binds, body, ty, span) => {
      let fb : Array[(String, @hir.HirExpr)] = Array::new()
      for pair in binds {
        let (n, e) = pair
        fb.push((n, self.finalize_expr(e)))
      }
      @hir.HirExpr::Let(fb, self.finalize_expr(body), self.engine.finalize(ty), span)
    }
    Case(scrut, arms, ty, span) => {
      let farms : Array[@hir.HirCaseArm] = Array::new()
      for arm in arms {
        farms.push(
          @hir.HirCaseArm::{
            pattern: self.finalize_pattern(arm.pattern),
            body: self.finalize_expr(arm.body),
          },
        )
      }
      @hir.HirExpr::Case(
        self.finalize_expr(scrut),
        farms,
        self.engine.finalize(ty),
        span,
      )
    }
    If(c, t, e, ty, span) =>
      @hir.HirExpr::If(
        self.finalize_expr(c),
        self.finalize_expr(t),
        self.finalize_expr(e),
        self.engine.finalize(ty),
        span,
      )
    BinOpE(op, lhs, rhs, ty, span) =>
      @hir.HirExpr::BinOpE(
        op,
        self.finalize_expr(lhs),
        self.finalize_expr(rhs),
        self.engine.finalize(ty),
        span,
      )
    ConstructorCall(name, tag, args, ty, span) => {
      let fa : Array[@hir.HirExpr] = Array::new()
      for a in args {
        fa.push(self.finalize_expr(a))
      }
      @hir.HirExpr::ConstructorCall(name, tag, fa, self.engine.finalize(ty), span)
    }
    FieldAccess(inner, field, ty, span) =>
      @hir.HirExpr::FieldAccess(
        self.finalize_expr(inner),
        field,
        self.engine.finalize(ty),
        span,
      )
  }
}

///|
fn AstLowering::finalize_pattern(
  self : AstLowering,
  pat : @hir.HirPattern
) -> @hir.HirPattern {
  match pat {
    Wild => @hir.HirPattern::Wild
    Var(name, ty) => @hir.HirPattern::Var(name, self.engine.finalize(ty))
    Constructor(name, tag, subs) => {
      let fs : Array[@hir.HirPattern] = Array::new()
      for s in subs {
        fs.push(self.finalize_pattern(s))
      }
      @hir.HirPattern::Constructor(name, tag, fs)
    }
    PatLit(lit) => @hir.HirPattern::PatLit(lit)
  }
}
