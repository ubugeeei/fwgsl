///|
/// HIR -> MIR lowering.
///
/// Converts the type-annotated HIR into MIR, which maps closely to WGSL.
/// Key transformations:
/// - Curried applications `App(App(f, a), b)` flatten to `Call(f, [a, b])`.
/// - If-expressions become `var tmp; if { tmp = a } else { tmp = b }` blocks.
/// - Let-expressions become `let` statements followed by the body.
/// - Case-expressions become chains of if-else on constructor tags.
/// - Nullary enum constructors become `u32` tag literals.
/// - Record constructors emit `MirStruct` definitions plus
///   `ConstructStruct` expressions.

///|
/// Lower a complete HIR program to MIR.
pub fn lower_hir_to_mir(
  hir : @hir.HirProgram
) -> Result[MirProgram, Array[String]] {
  let errors : Array[String] = Array::new()
  let structs : Array[MirStruct] = Array::new()
  let functions : Array[MirFunction] = Array::new()
  let entry_points : Array[MirEntryPoint] = Array::new()

  // Lower record-shaped data type constructors into MIR structs.
  for dt in hir.data_types {
    for con in dt.constructors {
      if con.fields.length() > 0 {
        let fields : Array[MirField] = Array::new()
        for pair in con.fields {
          let (name, ty) = pair
          let mir_ty = ty_to_mir_type_or_default(ty)
          fields.push(MirField::{ name, ty: mir_ty })
        }
        structs.push(MirStruct::{ name: con.name, fields })
      }
    }
  }

  // Lower functions.
  for f in hir.functions {
    match lower_hir_function(f) {
      Ok(mir_f) => functions.push(mir_f)
      Err(e) => errors.push(e)
    }
  }

  // Lower entry points.
  for ep in hir.entry_points {
    match lower_hir_entry_point(ep) {
      Ok(mir_ep) => entry_points.push(mir_ep)
      Err(e) => errors.push(e)
    }
  }
  if errors.length() == 0 {
    Ok(MirProgram::{ structs, functions, entry_points })
  } else {
    Err(errors)
  }
}

///|
/// Convert a `Ty` to `MirType`. Returns `Err` for types WGSL cannot
/// represent.
pub fn ty_to_mir_type(ty : @typechecker.Ty) -> Result[MirType, String] {
  let ty = @typechecker.normalize_type_aliases(ty)
  match ty {
    Con(name) =>
      match name {
        "I32" => Ok(I32)
        "U32" => Ok(U32)
        "F32" => Ok(F32)
        "Bool" => Ok(Bool)
        "()" => Ok(Unit)
        _ => Ok(MirStructTy(name))
      }
    App(f, arg) => app_to_mir(f, arg, ty)
    Arrow(_, _) => Err("Function types cannot be represented in WGSL")
    Var(_) | Error => Ok(I32)
    _ => Err("Cannot convert to MIR type: " + ty.display())
  }
}

///|
fn app_to_mir(
  f : @typechecker.Ty,
  arg : @typechecker.Ty,
  original : @typechecker.Ty
) -> Result[MirType, String] {
  match f {
    App(ff, n) =>
      match (ff, n) {
        // Mat N M Scalar
        (App(fff, nn), Nat(m)) =>
          match (fff, nn) {
            (Con(name), Nat(nv)) =>
              if name == "Mat" {
                match ty_to_mir_type(arg) {
                  Ok(elem) => Ok(MirMat(nv.to_uint(), m.to_uint(), elem))
                  Err(e) => Err(e)
                }
              } else {
                Err("Cannot convert to MIR type: " + original.display())
              }
            _ => Err("Cannot convert to MIR type: " + original.display())
          }
        // Vec N Scalar or Tensor N Elem
        (Con(name), Nat(nv)) =>
          if name == "Vec" {
            match ty_to_mir_type(arg) {
              Ok(elem) => Ok(MirVec(nv.to_uint(), elem))
              Err(e) => Err(e)
            }
          } else if name == "Tensor" {
            match ty_to_mir_type(arg) {
              Ok(elem) => Ok(MirArray(elem, nv.to_uint()))
              Err(e) => Err(e)
            }
          } else {
            Err("Cannot convert to MIR type: " + original.display())
          }
        _ => Err("Cannot convert to MIR type: " + original.display())
      }
    _ => Err("Cannot convert to MIR type: " + original.display())
  }
}

///|
fn ty_to_mir_type_or_default(ty : @typechecker.Ty) -> MirType {
  match ty_to_mir_type(ty) {
    Ok(t) => t
    Err(_) => I32
  }
}

///|
fn lower_hir_function(
  f : @hir.HirFunction
) -> Result[MirFunction, String] {
  let params : Array[MirParam] = Array::new()
  for pair in f.params {
    let (name, ty) = pair
    params.push(MirParam::{ name, ty: ty_to_mir_type_or_default(ty) })
  }
  let return_ty = ty_to_mir_type_or_default(f.return_ty)
  let body_result = lower_hir_expr_to_body(f.body)
  match body_result {
    Err(e) => Err(e)
    Ok(pair) => {
      let (stmts, return_expr) = pair
      Ok(MirFunction::{
        name: f.name,
        params,
        return_ty,
        body: stmts,
        return_expr: Some(return_expr),
      })
    }
  }
}

///|
fn lower_hir_entry_point(
  ep : @hir.HirEntryPoint
) -> Result[MirEntryPoint, String] {
  let mut stage : ShaderStage = Compute
  let mut workgroup_size : (UInt, UInt, UInt)? = None
  for attr in ep.attributes {
    match attr.name {
      "compute" => stage = Compute
      "vertex" => stage = Vertex
      "fragment" => stage = Fragment
      "workgroup_size" => {
        let mut x : UInt = 1
        let mut y : UInt = 1
        let mut z : UInt = 1
        let n = attr.args.length()
        if n >= 1 {
          x = parse_uint_or(attr.args[0], 1)
        }
        if n >= 2 {
          y = parse_uint_or(attr.args[1], 1)
        }
        if n >= 3 {
          z = parse_uint_or(attr.args[2], 1)
        }
        workgroup_size = Some((x, y, z))
      }
      _ => ()
    }
  }
  let body_result = lower_hir_expr_to_body(ep.body)
  match body_result {
    Err(e) => Err(e)
    Ok(pair) => {
      let (initial_stmts, return_expr) = pair
      let stmts : Array[MirStmt] = Array::new()
      // We rebuild stmts so we can insert at the front for compute shaders.
      for s in initial_stmts {
        stmts.push(s)
      }
      let builtins : Array[(String, BuiltinBinding, MirType)] = Array::new()
      let params : Array[MirParam] = Array::new()
      if stage == Compute {
        let gid_ty = MirVec(3, U32)
        builtins.push(("_gid", GlobalInvocationId, gid_ty))
        if ep.params.length() > 0 {
          let (name, _ty) = ep.params[0]
          let gid_x = MirExpr::FieldAccess(
            MirExpr::VarExpr("_gid", MirVec(3, U32)),
            "x",
            U32,
          )
          let cast_to_i32 = MirExpr::Cast(gid_x, I32)
          let prepend = MirStmt::Let(name, I32, cast_to_i32)
          let rebuilt : Array[MirStmt] = [prepend]
          for s in stmts {
            rebuilt.push(s)
          }
          while stmts.length() > 0 {
            let _ = stmts.pop()
          }
          for s in rebuilt {
            stmts.push(s)
          }
        }
      } else {
        for pair in ep.params {
          let (name, ty) = pair
          params.push(MirParam::{ name, ty: ty_to_mir_type_or_default(ty) })
        }
      }
      let return_ty : MirType = if stage == Compute {
        Unit
      } else {
        ty_to_mir_type_or_default(ep.return_ty)
      }
      let final_return_expr : MirExpr? = if return_ty == Unit {
        None
      } else {
        Some(return_expr)
      }
      Ok(MirEntryPoint::{
        name: ep.name,
        stage,
        workgroup_size,
        params,
        builtins,
        return_ty,
        body: stmts,
        return_expr: final_return_expr,
      })
    }
  }
}

///|
fn parse_uint_or(text : String, default_value : UInt) -> UInt {
  let len = text.length()
  let mut i = 0
  let mut acc : UInt = 0
  let mut saw_digit = false
  while i < len {
    let c = text.charcode_at(i)
    if c < 0x30 || c > 0x39 {
      break
    }
    saw_digit = true
    acc = acc * 10 + (c - 0x30).reinterpret_as_uint()
    i = i + 1
  }
  if saw_digit {
    acc
  } else {
    default_value
  }
}

///|
fn lower_hir_expr_to_body(
  expr : @hir.HirExpr
) -> Result[(Array[MirStmt], MirExpr), String] {
  lower_hir_expr_to_stmts(expr)
}

///|
fn lower_hir_expr_to_stmts(
  expr : @hir.HirExpr
) -> Result[(Array[MirStmt], MirExpr), String] {
  match expr {
    Let(binds, body, _, _) => {
      let stmts : Array[MirStmt] = Array::new()
      for pair in binds {
        let (name, bind_expr) = pair
        match lower_hir_expr_to_stmts(bind_expr) {
          Err(e) => return Err(e)
          Ok(bp) => {
            let (bind_stmts, bind_val) = bp
            for s in bind_stmts {
              stmts.push(s)
            }
            let bind_ty = ty_to_mir_type_or_default(bind_expr.ty())
            stmts.push(MirStmt::Let(name, bind_ty, bind_val))
          }
        }
      }
      match lower_hir_expr_to_stmts(body) {
        Err(e) => Err(e)
        Ok(bp) => {
          let (body_stmts, body_expr) = bp
          for s in body_stmts {
            stmts.push(s)
          }
          Ok((stmts, body_expr))
        }
      }
    }
    If(cond, then_expr, else_expr, ty, span) => {
      let cond_mir_r = lower_hir_expr(cond)
      match cond_mir_r {
        Err(e) => Err(e)
        Ok(cond_mir) => {
          let result_ty = ty_to_mir_type_or_default(ty)
          let tmp_name = "_if_tmp_" + span.start.to_string()
          match lower_hir_expr_to_stmts(then_expr) {
            Err(e) => Err(e)
            Ok(tp) => {
              let (then_stmts, then_val) = tp
              then_stmts.push(MirStmt::Assign(tmp_name, then_val))
              match lower_hir_expr_to_stmts(else_expr) {
                Err(e) => Err(e)
                Ok(ep_pair) => {
                  let (else_stmts, else_val) = ep_pair
                  else_stmts.push(MirStmt::Assign(tmp_name, else_val))
                  let stmts : Array[MirStmt] = [
                    MirStmt::VarDecl(
                      tmp_name,
                      result_ty,
                      default_expr_for_type(result_ty),
                    ),
                    MirStmt::IfStmt(cond_mir, then_stmts, else_stmts),
                  ]
                  Ok((stmts, MirExpr::VarExpr(tmp_name, result_ty)))
                }
              }
            }
          }
        }
      }
    }
    Case(scrutinee, arms, ty, span) => {
      let result_ty = ty_to_mir_type_or_default(ty)
      let tmp_name = "_case_tmp_" + span.start.to_string()
      match lower_hir_expr(scrutinee) {
        Err(e) => Err(e)
        Ok(scrut_mir) => {
          let scrut_ty = ty_to_mir_type_or_default(scrutinee.ty())
          let scrut_name = "_scrut_" + span.start.to_string()
          let stmts : Array[MirStmt] = [
            MirStmt::Let(scrut_name, scrut_ty, scrut_mir),
            MirStmt::VarDecl(
              tmp_name,
              result_ty,
              default_expr_for_type(result_ty),
            ),
          ]
          match lower_case_arms(scrut_name, scrut_ty, arms, tmp_name) {
            Err(e) => Err(e)
            Ok(Some(stmt)) => {
              stmts.push(stmt)
              Ok((stmts, MirExpr::VarExpr(tmp_name, result_ty)))
            }
            Ok(None) => Ok((stmts, MirExpr::VarExpr(tmp_name, result_ty)))
          }
        }
      }
    }
    _ =>
      match lower_hir_expr(expr) {
        Err(e) => Err(e)
        Ok(m) => Ok((Array::new(), m))
      }
  }
}

///|
fn lower_hir_expr(expr : @hir.HirExpr) -> Result[MirExpr, String] {
  match expr {
    Lit(lit, ty, _) => Ok(MirExpr::Lit(lower_hir_lit(lit, ty)))
    Var(name, ty, _) => Ok(MirExpr::VarExpr(name, ty_to_mir_type_or_default(ty)))
    BinOpE(op, lhs, rhs, ty, _) => {
      let lhs_r = lower_hir_expr(lhs)
      let rhs_r = lower_hir_expr(rhs)
      match (lhs_r, rhs_r) {
        (Ok(mlhs), Ok(mrhs)) =>
          Ok(
            MirExpr::BinOpE(
              lower_binop(op),
              mlhs,
              mrhs,
              ty_to_mir_type_or_default(ty),
            ),
          )
        (Err(e), _) => Err(e)
        (_, Err(e)) => Err(e)
      }
    }
    App(_, _, ty, _) => lower_app(expr, ty)
    ConstructorCall(name, tag, args, _, _) =>
      if args.length() == 0 {
        Ok(MirExpr::Lit(MirLit::U32Lit(tag)))
      } else {
        let mir_args : Array[MirExpr] = Array::new()
        for a in args {
          match lower_hir_expr(a) {
            Ok(m) => mir_args.push(m)
            Err(e) => return Err(e)
          }
        }
        Ok(MirExpr::ConstructStruct(name, mir_args))
      }
    FieldAccess(inner, field, ty, _) =>
      match lower_hir_expr(inner) {
        Err(e) => Err(e)
        Ok(m) =>
          Ok(MirExpr::FieldAccess(m, field, ty_to_mir_type_or_default(ty)))
      }
    Let(_, body, _, _) => lower_hir_expr(body)
    If(_, _, _, _, _) =>
      Err("If-expression in pure expression context; use lower_hir_expr_to_stmts")
    Case(_, _, _, _) =>
      Err("Case-expression in pure expression context; use lower_hir_expr_to_stmts")
  }
}

///|
fn lower_app(expr : @hir.HirExpr, ty : @typechecker.Ty) -> Result[MirExpr, String] {
  let (func_name, args_rev) = flatten_app(expr)
  let args : Array[@hir.HirExpr] = Array::new()
  let mut i = args_rev.length() - 1
  while i >= 0 {
    args.push(args_rev[i])
    i = i - 1
  }
  let mir_args : Array[MirExpr] = Array::new()
  for a in args {
    match lower_hir_expr(a) {
      Ok(m) => mir_args.push(m)
      Err(e) => return Err(e)
    }
  }
  let mir_ty = ty_to_mir_type_or_default(ty)
  match (func_name, mir_args.length()) {
    ("negate", 1) => Ok(MirExpr::UnaryOpE(Neg, mir_args[0], mir_ty))
    ("$mod", 2) => {
      let lhs = mir_args[0]
      let rhs = mir_args[1]
      let div = MirExpr::BinOpE(Div, lhs, rhs, mir_ty)
      let floored = MirExpr::Call("$floor", [div], mir_ty)
      let product = MirExpr::BinOpE(Mul, rhs, floored, mir_ty)
      Ok(MirExpr::BinOpE(Sub, lhs, product, mir_ty))
    }
    ("$atan", 2) =>
      Ok(MirExpr::Call("atan2", [mir_args[0], mir_args[1]], mir_ty))
    ("$splat2", 1) =>
      Ok(MirExpr::Call("$vec2", [mir_args[0], mir_args[0]], mir_ty))
    ("$splat3", 1) =>
      Ok(
        MirExpr::Call(
          "$vec3",
          [mir_args[0], mir_args[0], mir_args[0]],
          mir_ty,
        ),
      )
    ("$splat4", 1) =>
      Ok(
        MirExpr::Call(
          "$vec4",
          [mir_args[0], mir_args[0], mir_args[0], mir_args[0]],
          mir_ty,
        ),
      )
    ("$vecX", 1) => Ok(MirExpr::FieldAccess(mir_args[0], "x", mir_ty))
    ("$vecY", 1) => Ok(MirExpr::FieldAccess(mir_args[0], "y", mir_ty))
    ("$vecZ", 1) => Ok(MirExpr::FieldAccess(mir_args[0], "z", mir_ty))
    ("$vecW", 1) => Ok(MirExpr::FieldAccess(mir_args[0], "w", mir_ty))
    _ => Ok(MirExpr::Call(func_name, mir_args, mir_ty))
  }
}

///|
fn flatten_app(expr : @hir.HirExpr) -> (String, Array[@hir.HirExpr]) {
  let args : Array[@hir.HirExpr] = Array::new()
  let mut current = expr
  while true {
    match current {
      App(func, arg, _, _) => {
        args.push(arg)
        current = func
      }
      Var(name, _, _) => return (name, args)
      ConstructorCall(name, _, _, _, _) => return (name, args)
      _ => return ("_unknown", args)
    }
  }
  ("_unknown", args)
}

///|
fn lower_case_arms(
  scrut_name : String,
  scrut_ty : MirType,
  arms : Array[@hir.HirCaseArm],
  result_name : String
) -> Result[MirStmt?, String] {
  if arms.length() == 0 {
    return Ok(None)
  }
  let mut result : MirStmt? = None
  let mut i = arms.length() - 1
  while i >= 0 {
    let arm = arms[i]
    match lower_hir_expr_to_stmts(arm.body) {
      Err(e) => return Err(e)
      Ok(pair) => {
        let (body_stmts, body_expr) = pair
        body_stmts.push(MirStmt::Assign(result_name, body_expr))
        match arm.pattern {
          Wild | Var(_, _) => result = Some(MirStmt::Block(body_stmts))
          Constructor(_, tag, _) => {
            let cond = MirExpr::BinOpE(
              Eq,
              MirExpr::VarExpr(scrut_name, scrut_ty),
              MirExpr::Lit(MirLit::U32Lit(tag)),
              Bool,
            )
            let else_stmts = case_else_stmts(result)
            result = Some(MirStmt::IfStmt(cond, body_stmts, else_stmts))
          }
          PatLit(lit) => {
            let mir_lit = lit_to_mir(lit)
            let cond = MirExpr::BinOpE(
              Eq,
              MirExpr::VarExpr(scrut_name, scrut_ty),
              MirExpr::Lit(mir_lit),
              Bool,
            )
            let else_stmts = case_else_stmts(result)
            result = Some(MirStmt::IfStmt(cond, body_stmts, else_stmts))
          }
        }
      }
    }
    i = i - 1
  }
  Ok(result)
}

///|
fn case_else_stmts(prev : MirStmt?) -> Array[MirStmt] {
  match prev {
    Some(IfStmt(c, t, e)) => [MirStmt::IfStmt(c, t, e)]
    Some(Block(stmts)) => stmts
    Some(other) => [other]
    None => Array::new()
  }
}

///|
fn lower_binop(op : @hir.BinOp) -> MirBinOp {
  match op {
    Add => Add
    Sub => Sub
    Mul => Mul
    Div => Div
    Mod => Mod
    Eq => Eq
    Ne => Neq
    Lt => Lt
    Gt => Gt
    Le => Le
    Ge => Ge
    And => And
    Or => Or
  }
}

///|
fn lit_to_mir(lit : @hir.HirLit) -> MirLit {
  match lit {
    Int(v) => MirLit::I32Lit(v.to_int())
    Float(v) => MirLit::F32Lit(v)
    Bool(v) => MirLit::BoolLit(v)
  }
}

///|
fn lower_hir_lit(lit : @hir.HirLit, ty : @typechecker.Ty) -> MirLit {
  match lit {
    Int(v) =>
      match ty {
        Con(name) =>
          if name == "U32" {
            MirLit::U32Lit(v.to_int().reinterpret_as_uint())
          } else if name == "F32" {
            MirLit::F32Lit(v.to_double())
          } else {
            MirLit::I32Lit(v.to_int())
          }
        _ => MirLit::I32Lit(v.to_int())
      }
    Float(v) => MirLit::F32Lit(v)
    Bool(v) => MirLit::BoolLit(v)
  }
}

///|
fn default_lit_for_type(ty : MirType) -> MirLit {
  match ty {
    I32 => MirLit::I32Lit(0)
    U32 => MirLit::U32Lit(0)
    F32 => MirLit::F32Lit(0.0)
    Bool => MirLit::BoolLit(false)
    _ => MirLit::I32Lit(0)
  }
}

///|
fn default_expr_for_type(ty : MirType) -> MirExpr {
  match ty {
    I32 | U32 | F32 | Bool => MirExpr::Lit(default_lit_for_type(ty))
    MirVec(n, inner) => {
      let args : Array[MirExpr] = Array::new()
      let mut i : UInt = 0
      while i < n {
        args.push(default_expr_for_type(inner))
        i = i + 1
      }
      MirExpr::Call("vec" + n.to_string(), args, ty)
    }
    MirMat(cols, rows, inner) => {
      let args : Array[MirExpr] = Array::new()
      let total = cols * rows
      let mut i : UInt = 0
      while i < total {
        args.push(default_expr_for_type(inner))
        i = i + 1
      }
      MirExpr::Call(
        "mat" + cols.to_string() + "x" + rows.to_string(),
        args,
        ty,
      )
    }
    _ => MirExpr::Lit(MirLit::I32Lit(0))
  }
}
