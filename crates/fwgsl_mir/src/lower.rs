//! HIR → MIR lowering.
//!
//! Converts the type-annotated HIR into MIR, which maps closely to WGSL.
//! Key transformations:
//! - Curried applications `App(App(f, a), b)` are flattened to `Call(f, [a, b])`
//! - If-expressions become `var tmp; if { tmp = a } else { tmp = b }` statements
//! - Let-expressions become `let` statements followed by the body
//! - Case-expressions become chains of if-else on constructor tags
//! - Simple enum ADTs map to u32 tag values
//! - Record ADTs produce MirStruct definitions

use fwgsl_hir::*;
use fwgsl_typechecker::Ty;

use crate::*;

/// Lower a complete HIR program to MIR.
pub fn lower_hir_to_mir(hir: &HirProgram) -> Result<MirProgram, Vec<String>> {
    let mut errors = Vec::new();
    let mut structs = Vec::new();
    let mut functions = Vec::new();
    let mut entry_points = Vec::new();

    // Lower data types to structs (record ADTs)
    for dt in &hir.data_types {
        for con in &dt.constructors {
            if !con.fields.is_empty() {
                let fields = con
                    .fields
                    .iter()
                    .map(|(name, ty)| {
                        let mir_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
                        MirField {
                            name: name.clone(),
                            ty: mir_ty,
                        }
                    })
                    .collect();
                structs.push(MirStruct {
                    name: con.name.clone(),
                    fields,
                });
            }
        }
    }

    // Lower functions
    for f in &hir.functions {
        match lower_hir_function(f) {
            Ok(mir_f) => functions.push(mir_f),
            Err(e) => errors.push(e),
        }
    }

    // Lower entry points
    for ep in &hir.entry_points {
        match lower_hir_entry_point(ep) {
            Ok(mir_ep) => entry_points.push(mir_ep),
            Err(e) => errors.push(e),
        }
    }

    if errors.is_empty() {
        Ok(MirProgram {
            structs,
            functions,
            entry_points,
        })
    } else {
        Err(errors)
    }
}

/// Convert a Ty to MirType.
pub fn ty_to_mir_type(ty: &Ty) -> Result<MirType, String> {
    match ty {
        Ty::Con(name) => match name.as_str() {
            "I32" => Ok(MirType::I32),
            "U32" => Ok(MirType::U32),
            "F32" => Ok(MirType::F32),
            "Bool" => Ok(MirType::Bool),
            "()" => Ok(MirType::Unit),
            other => Ok(MirType::Struct(other.to_string())),
        },
        Ty::App(f, arg) => match f.as_ref() {
            Ty::App(ff, n) => match (ff.as_ref(), n.as_ref()) {
                (Ty::App(fff, nn), Ty::Nat(m)) => {
                    if let (Ty::Con(name), Ty::Nat(n)) = (fff.as_ref(), nn.as_ref()) {
                        if name == "Mat" {
                            let scalar = ty_to_mir_type(arg)?;
                            return Ok(MirType::Mat(*n as u8, *m as u8, Box::new(scalar)));
                        }
                    }
                    Err(format!("Cannot convert to MIR type: {}", ty))
                }
                (Ty::Con(name), Ty::Nat(n)) if name == "Vec" => {
                    let scalar = ty_to_mir_type(arg)?;
                    Ok(MirType::Vec(*n as u8, Box::new(scalar)))
                }
                _ => Err(format!("Cannot convert to MIR type: {}", ty)),
            },
            _ => Err(format!("Cannot convert to MIR type: {}", ty)),
        },
        Ty::Arrow(_, _) => {
            // Function types can't be represented in WGSL. This is only
            // reached for higher-order values that haven't been applied yet.
            Err("Function types cannot be represented in WGSL".into())
        }
        Ty::Var(_) => {
            // Unresolved type variable — default to i32 to avoid hard failure
            Ok(MirType::I32)
        }
        Ty::Error => Ok(MirType::I32),
        _ => Err(format!("Cannot convert to MIR type: {}", ty)),
    }
}

fn lower_hir_function(f: &HirFunction) -> Result<MirFunction, String> {
    let params: Vec<MirParam> = f
        .params
        .iter()
        .map(|(name, ty)| {
            let mir_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
            MirParam {
                name: name.clone(),
                ty: mir_ty,
            }
        })
        .collect();

    let return_ty = ty_to_mir_type(&f.return_ty).unwrap_or(MirType::I32);

    let (stmts, return_expr) = lower_hir_expr_to_body(&f.body)?;

    Ok(MirFunction {
        name: f.name.clone(),
        params,
        return_ty,
        body: stmts,
        return_expr: Some(return_expr),
    })
}

fn lower_hir_entry_point(ep: &HirEntryPoint) -> Result<MirEntryPoint, String> {
    // Parse stage and workgroup_size from attributes
    let mut stage = ShaderStage::Compute;
    let mut workgroup_size = None;

    for attr in &ep.attributes {
        match attr.name.as_str() {
            "compute" => stage = ShaderStage::Compute,
            "vertex" => stage = ShaderStage::Vertex,
            "fragment" => stage = ShaderStage::Fragment,
            "workgroup_size" => {
                let mut ws = [1u32; 3];
                for (i, arg) in attr.args.iter().take(3).enumerate() {
                    ws[i] = arg.parse().unwrap_or(1);
                }
                workgroup_size = Some(ws);
            }
            _ => {}
        }
    }

    let (mut stmts, return_expr) = lower_hir_expr_to_body(&ep.body)?;

    // For compute shaders, remap user parameters to builtin bindings.
    // The first parameter becomes `i32(_gid.x)` via global_invocation_id.
    let mut builtins = Vec::new();
    let mut params = Vec::new();

    if stage == ShaderStage::Compute {
        // Add @builtin(global_invocation_id) _gid: vec3<u32>
        let gid_ty = MirType::Vec(3, Box::new(MirType::U32));
        builtins.push((
            "_gid".to_string(),
            BuiltinBinding::GlobalInvocationId,
            gid_ty,
        ));

        // If the user declared a parameter (e.g. `main idx = ...`),
        // insert `let idx: i32 = i32(_gid.x);` at the start of the body.
        if let Some((name, _ty)) = ep.params.first() {
            let gid_x = MirExpr::FieldAccess(
                Box::new(MirExpr::Var(
                    "_gid".to_string(),
                    MirType::Vec(3, Box::new(MirType::U32)),
                )),
                "x".to_string(),
                MirType::U32,
            );
            let cast_to_i32 = MirExpr::Cast(Box::new(gid_x), MirType::I32);
            stmts.insert(0, MirStmt::Let(name.clone(), MirType::I32, cast_to_i32));
        }
    } else {
        params = ep
            .params
            .iter()
            .map(|(name, ty)| {
                let mir_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
                MirParam {
                    name: name.clone(),
                    ty: mir_ty,
                }
            })
            .collect();
    }

    // Compute shaders always have void return type in WGSL
    let return_ty = if stage == ShaderStage::Compute {
        MirType::Unit
    } else {
        ty_to_mir_type(&ep.return_ty).unwrap_or(MirType::Unit)
    };

    let return_expr = if return_ty == MirType::Unit {
        None
    } else {
        Some(return_expr)
    };

    Ok(MirEntryPoint {
        name: ep.name.clone(),
        stage,
        workgroup_size,
        params,
        builtins,
        return_ty,
        body: stmts,
        return_expr,
    })
}

/// Lower a HIR expression that forms a function body.
/// Returns (statements, final_return_expression).
fn lower_hir_expr_to_body(expr: &HirExpr) -> Result<(Vec<MirStmt>, MirExpr), String> {
    let (stmts, result_expr) = lower_hir_expr_to_stmts(expr)?;
    Ok((stmts, result_expr))
}

/// Lower a HIR expression, potentially producing statements (for Let, If, Case).
/// Returns (prefix_statements, result_expression).
fn lower_hir_expr_to_stmts(expr: &HirExpr) -> Result<(Vec<MirStmt>, MirExpr), String> {
    match expr {
        HirExpr::Let(binds, body, _ty, _span) => {
            let mut stmts = Vec::new();
            for (name, bind_expr) in binds {
                let (mut bind_stmts, bind_val) = lower_hir_expr_to_stmts(bind_expr)?;
                stmts.append(&mut bind_stmts);
                let bind_ty = ty_to_mir_type(bind_expr.ty()).unwrap_or(MirType::I32);
                stmts.push(MirStmt::Let(name.clone(), bind_ty, bind_val));
            }
            let (mut body_stmts, body_expr) = lower_hir_expr_to_stmts(body)?;
            stmts.append(&mut body_stmts);
            Ok((stmts, body_expr))
        }

        HirExpr::If(cond, then_expr, else_expr, ty, _span) => {
            let cond_mir = lower_hir_expr(cond)?;
            let result_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);

            // Flatten to: var tmp; if (cond) { tmp = then } else { tmp = else }; tmp
            let tmp_name = format!("_if_tmp_{}", _span.start);

            let (mut then_stmts, then_val) = lower_hir_expr_to_stmts(then_expr)?;
            then_stmts.push(MirStmt::Assign(tmp_name.clone(), then_val));

            let (mut else_stmts, else_val) = lower_hir_expr_to_stmts(else_expr)?;
            else_stmts.push(MirStmt::Assign(tmp_name.clone(), else_val));

            let stmts = vec![
                MirStmt::Var(
                    tmp_name.clone(),
                    result_ty.clone(),
                    MirExpr::Lit(default_lit_for_type(&result_ty)),
                ),
                MirStmt::If(cond_mir, then_stmts, else_stmts),
            ];

            Ok((stmts, MirExpr::Var(tmp_name, result_ty)))
        }

        HirExpr::Case(scrutinee, arms, ty, _span) => {
            let result_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
            let tmp_name = format!("_case_tmp_{}", _span.start);
            let scrut_mir = lower_hir_expr(scrutinee)?;
            let scrut_ty = ty_to_mir_type(scrutinee.ty()).unwrap_or(MirType::I32);
            let scrut_name = format!("_scrut_{}", _span.start);

            let mut stmts = vec![
                MirStmt::Let(scrut_name.clone(), scrut_ty.clone(), scrut_mir),
                MirStmt::Var(
                    tmp_name.clone(),
                    result_ty.clone(),
                    MirExpr::Lit(default_lit_for_type(&result_ty)),
                ),
            ];

            // Build if-else chain from arms
            let if_chain = lower_case_arms(&scrut_name, &scrut_ty, arms, &tmp_name)?;

            if let Some(stmt) = if_chain {
                stmts.push(stmt);
            }

            Ok((stmts, MirExpr::Var(tmp_name, result_ty)))
        }

        // Simple expressions produce no statements
        _ => {
            let mir_expr = lower_hir_expr(expr)?;
            Ok((vec![], mir_expr))
        }
    }
}

/// Lower a pure HIR expression to a MIR expression (no statements needed).
fn lower_hir_expr(expr: &HirExpr) -> Result<MirExpr, String> {
    match expr {
        HirExpr::Lit(lit, ty, _span) => {
            let mir_lit = lower_hir_lit(lit, ty);
            Ok(MirExpr::Lit(mir_lit))
        }

        HirExpr::Var(name, ty, _span) => {
            let mir_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
            Ok(MirExpr::Var(name.clone(), mir_ty))
        }

        HirExpr::BinOp(op, lhs, rhs, ty, _span) => {
            let mir_lhs = lower_hir_expr(lhs)?;
            let mir_rhs = lower_hir_expr(rhs)?;
            let mir_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
            let mir_op = lower_binop(op);
            Ok(MirExpr::BinOp(
                mir_op,
                Box::new(mir_lhs),
                Box::new(mir_rhs),
                mir_ty,
            ))
        }

        HirExpr::App(_, _, ty, _span) => {
            // Flatten curried applications: App(App(f, a), b) -> Call(f, [a, b])
            let (func_name, args) = flatten_app(expr);
            let mir_args: Result<Vec<MirExpr>, String> =
                args.iter().map(|a| lower_hir_expr(a)).collect();
            let mir_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
            Ok(MirExpr::Call(func_name, mir_args?, mir_ty))
        }

        HirExpr::ConstructorCall(name, _tag, args, _ty, _span) => {
            if args.is_empty() {
                // Nullary constructor: emit as u32 tag literal
                Ok(MirExpr::Lit(MirLit::U32(*_tag)))
            } else {
                // Record constructor: emit as struct construction
                let mir_args: Result<Vec<MirExpr>, String> =
                    args.iter().map(lower_hir_expr).collect();
                Ok(MirExpr::ConstructStruct(name.clone(), mir_args?))
            }
        }

        HirExpr::FieldAccess(expr, field, ty, _span) => {
            let mir_expr = lower_hir_expr(expr)?;
            let mir_ty = ty_to_mir_type(ty).unwrap_or(MirType::I32);
            Ok(MirExpr::FieldAccess(
                Box::new(mir_expr),
                field.clone(),
                mir_ty,
            ))
        }

        HirExpr::Let(binds, body, _ty, _span) => {
            // For Let in expression position, we need to lower it.
            // This shouldn't normally happen since lower_hir_expr_to_stmts
            // handles Let, but handle it gracefully.
            let _ = binds;
            lower_hir_expr(body)
        }

        HirExpr::If(cond, then_expr, else_expr, _ty, _span) => {
            // If in pure expression position: try to lower directly
            // This may lose some semantics, but is a fallback
            let _ = (cond, then_expr, else_expr);
            Err("If-expression in pure expression context; use lower_hir_expr_to_stmts".into())
        }

        HirExpr::Case(_, _, _ty, _span) => {
            Err("Case-expression in pure expression context; use lower_hir_expr_to_stmts".into())
        }
    }
}

/// Flatten nested App nodes into (function_name, [args]).
fn flatten_app(expr: &HirExpr) -> (String, Vec<&HirExpr>) {
    let mut args = Vec::new();
    let mut current = expr;

    loop {
        match current {
            HirExpr::App(func, arg, _, _) => {
                args.push(arg.as_ref());
                current = func.as_ref();
            }
            HirExpr::Var(name, _, _) => {
                args.reverse();
                return (name.clone(), args);
            }
            HirExpr::ConstructorCall(name, _, _, _, _) => {
                args.reverse();
                return (name.clone(), args);
            }
            _ => {
                args.reverse();
                return ("_unknown".to_string(), args);
            }
        }
    }
}

/// Lower case arms into a chain of if-else statements.
fn lower_case_arms(
    scrut_name: &str,
    scrut_ty: &MirType,
    arms: &[HirCaseArm],
    result_name: &str,
) -> Result<Option<MirStmt>, String> {
    if arms.is_empty() {
        return Ok(None);
    }

    // Build from last to first (fold right)
    let mut result: Option<MirStmt> = None;

    for arm in arms.iter().rev() {
        let (mut body_stmts, body_expr) = lower_hir_expr_to_stmts(&arm.body)?;
        body_stmts.push(MirStmt::Assign(result_name.to_string(), body_expr));

        match &arm.pattern {
            HirPattern::Wild | HirPattern::Var(_, _) => {
                // Wildcard/var: always matches — becomes the fallback else branch.
                // In our right-fold, the wildcard arm seen first (from the end)
                // is the outermost else.
                result = Some(MirStmt::Block(body_stmts));
            }

            HirPattern::Constructor(_name, tag, _sub_pats) => {
                // Match on tag: if (scrut == tag) { ... }
                let cond = MirExpr::BinOp(
                    MirBinOp::Eq,
                    Box::new(MirExpr::Var(scrut_name.to_string(), scrut_ty.clone())),
                    Box::new(MirExpr::Lit(MirLit::U32(*tag))),
                    MirType::Bool,
                );

                let else_stmts = match result {
                    Some(MirStmt::If(c, t, e)) => vec![MirStmt::If(c, t, e)],
                    Some(MirStmt::Block(stmts)) => stmts,
                    Some(other) => vec![other],
                    None => vec![],
                };

                result = Some(MirStmt::If(cond, body_stmts, else_stmts));
            }

            HirPattern::Lit(hir_lit) => {
                let mir_lit = match hir_lit {
                    HirLit::Int(v) => MirLit::I32(*v as i32),
                    HirLit::Float(v) => MirLit::F32(*v),
                    HirLit::Bool(v) => MirLit::Bool(*v),
                };
                let cond = MirExpr::BinOp(
                    MirBinOp::Eq,
                    Box::new(MirExpr::Var(scrut_name.to_string(), scrut_ty.clone())),
                    Box::new(MirExpr::Lit(mir_lit)),
                    MirType::Bool,
                );

                let else_stmts = match result {
                    Some(MirStmt::If(c, t, e)) => vec![MirStmt::If(c, t, e)],
                    Some(MirStmt::Block(stmts)) => stmts,
                    Some(other) => vec![other],
                    None => vec![],
                };

                result = Some(MirStmt::If(cond, body_stmts, else_stmts));
            }
        }
    }

    Ok(result)
}

fn lower_binop(op: &BinOp) -> MirBinOp {
    match op {
        BinOp::Add => MirBinOp::Add,
        BinOp::Sub => MirBinOp::Sub,
        BinOp::Mul => MirBinOp::Mul,
        BinOp::Div => MirBinOp::Div,
        BinOp::Mod => MirBinOp::Mod,
        BinOp::Eq => MirBinOp::Eq,
        BinOp::Ne => MirBinOp::Neq,
        BinOp::Lt => MirBinOp::Lt,
        BinOp::Gt => MirBinOp::Gt,
        BinOp::Le => MirBinOp::Le,
        BinOp::Ge => MirBinOp::Ge,
        BinOp::And => MirBinOp::And,
        BinOp::Or => MirBinOp::Or,
    }
}

fn lower_hir_lit(lit: &HirLit, ty: &Ty) -> MirLit {
    match lit {
        HirLit::Int(v) => match ty {
            Ty::Con(name) if name == "U32" => MirLit::U32(*v as u32),
            Ty::Con(name) if name == "F32" => MirLit::F32(*v as f64),
            _ => MirLit::I32(*v as i32),
        },
        HirLit::Float(v) => MirLit::F32(*v),
        HirLit::Bool(v) => MirLit::Bool(*v),
    }
}

fn default_lit_for_type(ty: &MirType) -> MirLit {
    match ty {
        MirType::I32 => MirLit::I32(0),
        MirType::U32 => MirLit::U32(0),
        MirType::F32 => MirLit::F32(0.0),
        MirType::Bool => MirLit::Bool(false),
        _ => MirLit::I32(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fwgsl_span::Span;

    fn span() -> Span {
        Span::new(0, 0)
    }

    #[test]
    fn test_lower_simple_function() {
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "add".into(),
                params: vec![("x".into(), Ty::i32()), ("y".into(), Ty::i32())],
                return_ty: Ty::i32(),
                body: HirExpr::BinOp(
                    BinOp::Add,
                    Box::new(HirExpr::Var("x".into(), Ty::i32(), span())),
                    Box::new(HirExpr::Var("y".into(), Ty::i32(), span())),
                    Ty::i32(),
                    span(),
                ),
                span: span(),
            }],
            data_types: vec![],
            entry_points: vec![],
        };

        let mir = lower_hir_to_mir(&hir).expect("lowering should succeed");
        assert_eq!(mir.functions.len(), 1);
        let f = &mir.functions[0];
        assert_eq!(f.name, "add");
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.return_ty, MirType::I32);
        assert!(f.return_expr.is_some());
    }

    #[test]
    fn test_lower_if_expression() {
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "f".into(),
                params: vec![("x".into(), Ty::i32())],
                return_ty: Ty::i32(),
                body: HirExpr::If(
                    Box::new(HirExpr::BinOp(
                        BinOp::Eq,
                        Box::new(HirExpr::Var("x".into(), Ty::i32(), span())),
                        Box::new(HirExpr::Lit(HirLit::Int(0), Ty::i32(), span())),
                        Ty::bool(),
                        span(),
                    )),
                    Box::new(HirExpr::Lit(HirLit::Int(1), Ty::i32(), span())),
                    Box::new(HirExpr::Lit(HirLit::Int(2), Ty::i32(), span())),
                    Ty::i32(),
                    span(),
                ),
                span: span(),
            }],
            data_types: vec![],
            entry_points: vec![],
        };

        let mir = lower_hir_to_mir(&hir).expect("lowering should succeed");
        assert_eq!(mir.functions.len(), 1);
        let f = &mir.functions[0];
        // Should have a var declaration and an if statement
        assert!(f.body.len() >= 2);
    }

    #[test]
    fn test_ty_to_mir_type_scalars() {
        assert_eq!(ty_to_mir_type(&Ty::i32()), Ok(MirType::I32));
        assert_eq!(ty_to_mir_type(&Ty::f32()), Ok(MirType::F32));
        assert_eq!(ty_to_mir_type(&Ty::u32()), Ok(MirType::U32));
        assert_eq!(ty_to_mir_type(&Ty::bool()), Ok(MirType::Bool));
        assert_eq!(ty_to_mir_type(&Ty::unit()), Ok(MirType::Unit));
    }

    #[test]
    fn test_lower_let_expression() {
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "f".into(),
                params: vec![],
                return_ty: Ty::i32(),
                body: HirExpr::Let(
                    vec![("x".into(), HirExpr::Lit(HirLit::Int(42), Ty::i32(), span()))],
                    Box::new(HirExpr::BinOp(
                        BinOp::Add,
                        Box::new(HirExpr::Var("x".into(), Ty::i32(), span())),
                        Box::new(HirExpr::Lit(HirLit::Int(1), Ty::i32(), span())),
                        Ty::i32(),
                        span(),
                    )),
                    Ty::i32(),
                    span(),
                ),
                span: span(),
            }],
            data_types: vec![],
            entry_points: vec![],
        };

        let mir = lower_hir_to_mir(&hir).expect("lowering should succeed");
        assert_eq!(mir.functions.len(), 1);
        let f = &mir.functions[0];
        // Should have a let statement for x
        assert!(!f.body.is_empty());
    }
}
