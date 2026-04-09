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

use std::collections::HashMap;

use fwgsl_hir::*;
use fwgsl_typechecker::Ty;

use crate::*;

/// Bitfield field metadata used during lowering.
#[derive(Clone, Debug)]
struct BitfieldFieldInfo {
    offset: u32,
    width: u32,
}

/// Context for HIR → MIR lowering, carrying data type information.
struct LowerCtx {
    /// Map from data type name to its constructors
    data_types: HashMap<String, Vec<HirConstructor>>,
    /// Map from constructor name to (data_type_name, tag, fields)
    constructors: HashMap<String, (String, u32, Vec<HirFieldDef>)>,
    /// Map from bitfield type name to its field metadata
    bitfields: HashMap<String, Vec<(String, BitfieldFieldInfo)>>,
}

impl LowerCtx {
    fn new(hir: &HirProgram) -> Self {
        let mut data_types = HashMap::new();
        let mut constructors = HashMap::new();
        for dt in &hir.data_types {
            data_types.insert(dt.name.clone(), dt.constructors.clone());
            for con in &dt.constructors {
                constructors.insert(
                    con.name.clone(),
                    (dt.name.clone(), con.tag, con.fields.clone()),
                );
            }
        }
        let mut bitfields = HashMap::new();
        for bf in &hir.bitfields {
            let fields: Vec<(String, BitfieldFieldInfo)> = bf
                .fields
                .iter()
                .map(|f| {
                    (
                        f.name.clone(),
                        BitfieldFieldInfo {
                            offset: f.offset,
                            width: f.width,
                        },
                    )
                })
                .collect();
            bitfields.insert(bf.name.clone(), fields);
        }
        LowerCtx {
            data_types,
            constructors,
            bitfields,
        }
    }

    /// Look up a bitfield field by type name and field name.
    fn lookup_bitfield_field(&self, type_name: &str, field_name: &str) -> Option<&BitfieldFieldInfo> {
        self.bitfields.get(type_name).and_then(|fields| {
            fields.iter().find(|(n, _)| n == field_name).map(|(_, info)| info)
        })
    }

    fn is_sum_type(&self, name: &str) -> bool {
        self.data_types
            .get(name)
            .is_some_and(|cons| cons.len() > 1)
    }

    /// Check if a data type has any constructor with fields.
    fn has_fields(&self, name: &str) -> bool {
        self.data_types
            .get(name)
            .is_some_and(|cons| cons.iter().any(|c| !c.fields.is_empty()))
    }

    /// Check if a name is a pure enum (sum type with no fields on any constructor).
    fn is_pure_enum(&self, name: &str) -> bool {
        self.data_types
            .get(name)
            .is_some_and(|cons| cons.len() > 1 && cons.iter().all(|c| c.fields.is_empty()))
    }

    /// Resolve a type constructor name to MirType, taking ADTs and bitfields into account.
    fn resolve_type_con(&self, name: &str) -> Option<MirType> {
        // Check if this is a bitfield type — resolve to its base MIR type
        if self.bitfields.contains_key(name) {
            // Bitfields are backed by u32 (could be extended to support u16/u8)
            return Some(MirType::U32);
        }
        if self.is_pure_enum(name) {
            Some(MirType::U32)
        } else if self.is_sum_type(name) && self.has_fields(name) {
            Some(MirType::Struct(name.to_string()))
        } else if self.data_types.contains_key(name) {
            // Single-constructor type — use struct if it has fields, otherwise u32
            if self.has_fields(name) {
                Some(MirType::Struct(name.to_string()))
            } else {
                Some(MirType::U32)
            }
        } else {
            None // Not a known data type
        }
    }
}

/// Lower a complete HIR program to MIR.
pub fn lower_hir_to_mir(hir: &HirProgram) -> Result<MirProgram, Vec<String>> {
    let ctx = LowerCtx::new(hir);
    let mut errors = Vec::new();
    let mut structs = Vec::new();
    let mut globals = Vec::new();
    let mut functions = Vec::new();
    let mut entry_points = Vec::new();

    // Lower data types to structs
    for dt in &hir.data_types {
        if dt.constructors.len() > 1 && dt.constructors.iter().any(|c| !c.fields.is_empty()) {
            // Sum type with fields: emit one struct with tag + union of fields
            let mut fields = vec![MirField {
                name: "tag".to_string(),
                ty: MirType::U32,
                attributes: vec![],
            }];
            // Collect all field names/types from the largest constructor
            let max_con = dt
                .constructors
                .iter()
                .max_by_key(|c| c.fields.len())
                .unwrap();
            for f in &max_con.fields {
                let mir_ty = ty_to_mir_type_with_ctx(&f.ty, Some(&ctx)).unwrap_or(MirType::I32);
                fields.push(MirField {
                    name: f.name.clone(),
                    ty: mir_ty,
                    attributes: f.attributes.iter().map(|a| MirAttribute {
                        name: a.name.clone(),
                        args: a.args.clone(),
                    }).collect(),
                });
            }
            structs.push(MirStruct {
                name: dt.name.clone(),
                fields,
            });
        } else if dt.constructors.len() == 1 {
            // Single constructor (record type): emit struct with just its fields
            let con = &dt.constructors[0];
            if !con.fields.is_empty() {
                let fields = con
                    .fields
                    .iter()
                    .map(|f| {
                        let mir_ty = ty_to_mir_type_with_ctx(&f.ty, Some(&ctx)).unwrap_or(MirType::I32);
                        MirField {
                            name: f.name.clone(),
                            ty: mir_ty,
                            attributes: f.attributes.iter().map(|a| MirAttribute {
                                name: a.name.clone(),
                                args: a.args.clone(),
                            }).collect(),
                        }
                    })
                    .collect();
                structs.push(MirStruct {
                    name: dt.name.clone(),
                    fields,
                });
            }
        }
        // Pure enums (all nullary constructors) → no struct needed, use u32
    }

    // Lower resource declarations to global bindings
    for res in &hir.resources {
        if let Some(global) = lower_hir_resource(res, &ctx) {
            globals.push(global);
        }
    }

    // Lower functions
    for f in &hir.functions {
        match lower_hir_function(f, &ctx) {
            Ok(mir_f) => functions.push(mir_f),
            Err(e) => errors.push(e),
        }
    }

    // Lower entry points
    for ep in &hir.entry_points {
        match lower_hir_entry_point(ep, &ctx) {
            Ok(mir_ep) => entry_points.push(mir_ep),
            Err(e) => errors.push(e),
        }
    }

    // Lower constants
    let mut constants = Vec::new();
    for c in &hir.constants {
        let mir_ty = ty_to_mir_type_with_ctx(&c.ty, Some(&ctx)).unwrap_or(MirType::I32);
        match lower_hir_expr(&c.value, &ctx) {
            Ok(mir_expr) => constants.push(MirConst {
                name: c.name.clone(),
                ty: mir_ty,
                value: mir_expr,
            }),
            Err(e) => errors.push(e),
        }
    }

    if errors.is_empty() {
        Ok(MirProgram {
            structs,
            globals,
            functions,
            entry_points,
            constants,
        })
    } else {
        Err(errors)
    }
}

/// Convert a Ty to MirType (without ADT context).
pub fn ty_to_mir_type(ty: &Ty) -> Result<MirType, String> {
    ty_to_mir_type_with_ctx(ty, None)
}

/// Convert a Ty to MirType, optionally using ADT context.
fn ty_to_mir_type_with_ctx(ty: &Ty, ctx: Option<&LowerCtx>) -> Result<MirType, String> {
    let ty = fwgsl_typechecker::normalize_type_aliases(ty);

    match &ty {
        Ty::Con(name) => match name.as_str() {
            "I32" => Ok(MirType::I32),
            "U32" => Ok(MirType::U32),
            "F32" => Ok(MirType::F32),
            "Bool" => Ok(MirType::Bool),
            "()" => Ok(MirType::Unit),
            other => {
                if let Some(ctx) = ctx {
                    if let Some(mir_ty) = ctx.resolve_type_con(other) {
                        return Ok(mir_ty);
                    }
                }
                Ok(MirType::Struct(other.to_string()))
            }
        },
        Ty::App(f, arg) => match f.as_ref() {
            // Unsized array: Tensor<T> (single application, no Nat dimension)
            Ty::Con(name) if name == "Tensor" => {
                let elem = ty_to_mir_type_with_ctx(arg, ctx)?;
                Ok(MirType::RuntimeArray(Box::new(elem)))
            }
            Ty::App(ff, n) => match (ff.as_ref(), n.as_ref()) {
                (Ty::App(fff, nn), Ty::Nat(m)) => {
                    if let (Ty::Con(name), Ty::Nat(n)) = (fff.as_ref(), nn.as_ref()) {
                        if name == "Mat" {
                            let scalar = ty_to_mir_type_with_ctx(arg, ctx)?;
                            return Ok(MirType::Mat(*n as u8, *m as u8, Box::new(scalar)));
                        }
                    }
                    Err(format!("Cannot convert to MIR type: {}", ty))
                }
                (Ty::Con(name), Ty::Nat(n)) if name == "Vec" => {
                    let scalar = ty_to_mir_type_with_ctx(arg, ctx)?;
                    Ok(MirType::Vec(*n as u8, Box::new(scalar)))
                }
                (Ty::Con(name), Ty::Nat(n)) if name == "Tensor" => {
                    let elem = ty_to_mir_type_with_ctx(arg, ctx)?;
                    let len = u32::try_from(*n)
                        .map_err(|_| format!("Tensor length out of range for MIR: {}", n))?;
                    Ok(MirType::Array(Box::new(elem), len))
                }
                // Handle surface syntax order: Tensor<T, N> (Array<T, N>)
                // where n is the elem type and arg is Nat
                (Ty::Con(name), _elem_ty) if name == "Tensor" => {
                    if let Ty::Nat(len) = arg.as_ref() {
                        let elem = ty_to_mir_type_with_ctx(n, ctx)?;
                        let len = u32::try_from(*len)
                            .map_err(|_| format!("Tensor length out of range for MIR: {}", len))?;
                        Ok(MirType::Array(Box::new(elem), len))
                    } else {
                        Err(format!("Cannot convert to MIR type: {}", ty))
                    }
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

fn lower_hir_function(f: &HirFunction, ctx: &LowerCtx) -> Result<MirFunction, String> {
    let resolve = |ty: &Ty| ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
    let params: Vec<MirParam> = f
        .params
        .iter()
        .map(|(name, ty)| {
            MirParam {
                name: name.clone(),
                ty: resolve(ty),
            }
        })
        .collect();

    let return_ty = resolve(&f.return_ty);

    let (stmts, return_expr) = lower_hir_expr_to_body(&f.body, ctx)?;

    Ok(MirFunction {
        name: f.name.clone(),
        params,
        return_ty,
        body: stmts,
        return_expr: Some(return_expr),
        comments: f.comments.clone(),
    })
}

fn lower_hir_entry_point(ep: &HirEntryPoint, ctx: &LowerCtx) -> Result<MirEntryPoint, String> {
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

    let (stmts, return_expr) = lower_hir_expr_to_body(&ep.body, ctx)?;

    // Entry point params pass through directly. Bindings like @builtin and
    // @location are carried by the struct-typed parameter's field attributes
    // (declared via `data` with attributed fields), not injected here.
    let params: Vec<MirParam> = ep
        .params
        .iter()
        .map(|(name, ty)| {
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
            MirParam {
                name: name.clone(),
                ty: mir_ty,
            }
        })
        .collect();

    // Compute shaders always have void return type in WGSL
    let return_ty = if stage == ShaderStage::Compute {
        MirType::Unit
    } else {
        ty_to_mir_type_with_ctx(&ep.return_ty, Some(ctx)).unwrap_or(MirType::Unit)
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
        return_ty,
        body: stmts,
        return_expr,
        comments: ep.comments.clone(),
    })
}

/// Lower a HIR expression that forms a function body.
/// Returns (statements, final_return_expression).
fn lower_hir_expr_to_body(expr: &HirExpr, ctx: &LowerCtx) -> Result<(Vec<MirStmt>, MirExpr), String> {
    let (stmts, result_expr) = lower_hir_expr_to_stmts(expr, ctx)?;
    Ok((stmts, result_expr))
}

/// Lower a HIR expression, potentially producing statements (for Let, If, Case).
/// Returns (prefix_statements, result_expression).
fn lower_hir_expr_to_stmts(expr: &HirExpr, ctx: &LowerCtx) -> Result<(Vec<MirStmt>, MirExpr), String> {
    match expr {
        HirExpr::Let(binds, body, _ty, _span) => {
            let mut stmts = Vec::new();
            for (name, bind_expr) in binds {
                let (mut bind_stmts, bind_val) = lower_hir_expr_to_stmts(bind_expr, ctx)?;
                stmts.append(&mut bind_stmts);
                // Prefer the MIR expression's result type (more accurate after
                // bitfield desugaring) over the HIR type (may be an unresolved var).
                let bind_ty = bind_val
                    .result_type()
                    .unwrap_or_else(|| ty_to_mir_type_with_ctx(bind_expr.ty(), Some(ctx)).unwrap_or(MirType::I32));
                stmts.push(MirStmt::Let(name.clone(), bind_ty, bind_val));
            }
            let (mut body_stmts, body_expr) = lower_hir_expr_to_stmts(body, ctx)?;
            stmts.append(&mut body_stmts);
            Ok((stmts, body_expr))
        }

        HirExpr::If(cond, then_expr, else_expr, ty, _span) => {
            let cond_mir = lower_hir_expr(cond, ctx)?;
            let result_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);

            // Flatten to: var tmp; if (cond) { tmp = then } else { tmp = else }; tmp
            let tmp_name = format!("_if_tmp_{}", _span.start);

            let (mut then_stmts, then_val) = lower_hir_expr_to_stmts(then_expr, ctx)?;
            then_stmts.push(MirStmt::Assign(tmp_name.clone(), then_val));

            let (mut else_stmts, else_val) = lower_hir_expr_to_stmts(else_expr, ctx)?;
            else_stmts.push(MirStmt::Assign(tmp_name.clone(), else_val));

            let stmts = vec![
                MirStmt::Var(
                    tmp_name.clone(),
                    result_ty.clone(),
                    default_expr_for_type(&result_ty),
                ),
                MirStmt::If(cond_mir, then_stmts, else_stmts),
            ];

            Ok((stmts, MirExpr::Var(tmp_name, result_ty)))
        }

        HirExpr::Case(scrutinee, arms, ty, _span) => {
            let result_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
            let tmp_name = format!("_case_tmp_{}", _span.start);
            let scrut_mir = lower_hir_expr(scrutinee, ctx)?;
            let scrut_ty = ty_to_mir_type_with_ctx(scrutinee.ty(), Some(ctx)).unwrap_or(MirType::I32);
            let scrut_name = format!("_scrut_{}", _span.start);

            let mut stmts = vec![
                MirStmt::Let(scrut_name.clone(), scrut_ty.clone(), scrut_mir),
                MirStmt::Var(
                    tmp_name.clone(),
                    result_ty.clone(),
                    default_expr_for_type(&result_ty),
                ),
            ];

            // Build if-else chain from arms
            let if_chain = lower_case_arms(&scrut_name, &scrut_ty, arms, &tmp_name, ctx)?;

            if let Some(stmt) = if_chain {
                stmts.push(stmt);
            }

            Ok((stmts, MirExpr::Var(tmp_name, result_ty)))
        }

        // Check for writeAt calls which become IndexAssign statements
        HirExpr::App(_, _, ty, _) => {
            let (func_name, args) = flatten_app(expr);
            // Lower each argument via lower_hir_expr_to_stmts so that
            // Let-bindings inside arguments (from beta-reduced lambdas)
            // are properly emitted as preceding statements.
            let mut pre_stmts = Vec::new();
            let mut mir_args = Vec::new();
            for a in &args {
                let (mut arg_stmts, arg_expr) = lower_hir_expr_to_stmts(a, ctx)?;
                pre_stmts.append(&mut arg_stmts);
                mir_args.push(arg_expr);
            }
            if func_name == "writeAt" && mir_args.len() == 3 {
                let stmt = MirStmt::IndexAssign(
                    mir_args[0].clone(),
                    mir_args[1].clone(),
                    mir_args[2].clone(),
                );
                pre_stmts.push(stmt);
                Ok((pre_stmts, MirExpr::Lit(MirLit::I32(0))))
            } else {
                let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
                let result = lower_app_with_args(&func_name, mir_args, mir_ty, ctx)?;
                Ok((pre_stmts, result))
            }
        }

        HirExpr::Loop(loop_name, bindings, body, ty, _span) => {
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
            let result_var = format!("{}_result", loop_name);

            let mut pre_stmts = Vec::new();

            // Emit `var` for each binding
            let mut binding_names = Vec::new();
            let mut binding_types = Vec::new();
            for (bind_name, init_expr) in bindings {
                let (mut init_stmts, init_mir) = lower_hir_expr_to_stmts(init_expr, ctx)?;
                let bind_ty = init_mir.result_type().unwrap_or(MirType::I32);
                pre_stmts.append(&mut init_stmts);
                pre_stmts.push(MirStmt::Var(bind_name.clone(), bind_ty.clone(), init_mir));
                binding_names.push(bind_name.clone());
                binding_types.push(bind_ty);
            }

            // Emit result var
            pre_stmts.push(MirStmt::Var(
                result_var.clone(),
                mir_ty.clone(),
                MirExpr::default_value(&mir_ty),
            ));

            // Lower the body, converting calls to loop_name into assignments + continue
            let loop_body = lower_loop_body(
                body,
                loop_name,
                &binding_names,
                &binding_types,
                &result_var,
                &mir_ty,
                ctx,
            )?;

            pre_stmts.push(MirStmt::Loop(loop_body));

            Ok((pre_stmts, MirExpr::Var(result_var, mir_ty)))
        }

        // Simple expressions produce no statements
        _ => {
            let mir_expr = lower_hir_expr(expr, ctx)?;
            Ok((vec![], mir_expr))
        }
    }
}

/// Lower a pure HIR expression to a MIR expression (no statements needed).
fn lower_hir_expr(expr: &HirExpr, ctx: &LowerCtx) -> Result<MirExpr, String> {
    match expr {
        HirExpr::Lit(lit, ty, _span) => {
            let mir_lit = lower_hir_lit(lit, ty);
            Ok(MirExpr::Lit(mir_lit))
        }

        HirExpr::Var(name, ty, _span) => {
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
            Ok(MirExpr::Var(name.clone(), mir_ty))
        }

        HirExpr::BinOp(op, lhs, rhs, ty, _span) => {
            let mir_lhs = lower_hir_expr(lhs, ctx)?;
            let mir_rhs = lower_hir_expr(rhs, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
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
                args.iter().map(|a| lower_hir_expr(a, ctx)).collect();
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
            let mir_args = mir_args?;
            lower_app_with_args(&func_name, mir_args, mir_ty, ctx)
        }

        HirExpr::ConstructorCall(name, _tag, args, _ty, _span) => {
            // Check if this constructor belongs to a sum type with fields
            if let Some((dt_name, tag, _fields)) = ctx.constructors.get(name.as_str()) {
                if ctx.is_sum_type(dt_name) && ctx.has_fields(dt_name) {
                    // Emit as DataType struct: DataType(tag, field0, ...)
                    let mut all_args = vec![MirExpr::Lit(MirLit::U32(*tag))];
                    for arg in args {
                        all_args.push(lower_hir_expr(arg, ctx)?);
                    }
                    return Ok(MirExpr::ConstructStruct(dt_name.clone(), all_args));
                }
            }
            if args.is_empty() {
                // Nullary constructor: emit as u32 tag literal
                Ok(MirExpr::Lit(MirLit::U32(*_tag)))
            } else {
                // Record constructor: emit as struct construction
                let mir_args: Result<Vec<MirExpr>, String> =
                    args.iter().map(|a| lower_hir_expr(a, ctx)).collect();
                Ok(MirExpr::ConstructStruct(name.clone(), mir_args?))
            }
        }

        HirExpr::FieldAccess(inner_expr, field, ty, _span) => {
            // Check if this is a bitfield field access
            let inner_ty = inner_expr.ty();
            if let Ty::Con(type_name) = inner_ty {
                if let Some(bf_info) = ctx.lookup_bitfield_field(type_name, field) {
                    let mir_expr = lower_hir_expr(inner_expr, ctx)?;
                    let mask = (1u32 << bf_info.width) - 1;
                    // (val >> offset) & mask
                    let shifted = if bf_info.offset > 0 {
                        MirExpr::BinOp(
                            MirBinOp::Shr,
                            Box::new(mir_expr),
                            Box::new(MirExpr::Lit(MirLit::U32(bf_info.offset))),
                            MirType::U32,
                        )
                    } else {
                        mir_expr
                    };
                    let masked = MirExpr::BinOp(
                        MirBinOp::BitAnd,
                        Box::new(shifted),
                        Box::new(MirExpr::Lit(MirLit::U32(mask))),
                        MirType::U32,
                    );
                    // For 1-bit fields, compare != 0 to produce bool
                    if bf_info.width == 1 {
                        return Ok(MirExpr::BinOp(
                            MirBinOp::Neq,
                            Box::new(masked),
                            Box::new(MirExpr::Lit(MirLit::U32(0))),
                            MirType::Bool,
                        ));
                    }
                    return Ok(masked);
                }
            }
            let mir_expr = lower_hir_expr(inner_expr, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
            Ok(MirExpr::FieldAccess(
                Box::new(mir_expr),
                field.clone(),
                mir_ty,
            ))
        }
        HirExpr::Index(base, index, ty, _span) => {
            let mir_base = lower_hir_expr(base, ctx)?;
            let mir_index = lower_hir_expr(index, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::I32);
            Ok(MirExpr::Index(
                Box::new(mir_base),
                Box::new(mir_index),
                mir_ty,
            ))
        }

        HirExpr::UnaryNeg(inner, ty, _span) => {
            let mir_inner = lower_hir_expr(inner, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx)).unwrap_or(MirType::F32);
            Ok(MirExpr::UnaryOp(
                MirUnaryOp::Neg,
                Box::new(mir_inner),
                mir_ty,
            ))
        }

        HirExpr::UnaryNot(inner, _ty, _span) => {
            let mir_inner = lower_hir_expr(inner, ctx)?;
            Ok(MirExpr::UnaryOp(
                MirUnaryOp::Not,
                Box::new(mir_inner),
                MirType::Bool,
            ))
        }

        HirExpr::Let(binds, body, _ty, _span) => {
            let _ = binds;
            lower_hir_expr(body, ctx)
        }

        HirExpr::If(cond, then_expr, else_expr, _ty, _span) => {
            let _ = (cond, then_expr, else_expr);
            Err("If-expression in pure expression context; use lower_hir_expr_to_stmts".into())
        }

        HirExpr::Case(_, _, _ty, _span) => {
            Err("Case-expression in pure expression context; use lower_hir_expr_to_stmts".into())
        }

        HirExpr::Loop(_, _, _, _, _) => {
            Err("Loop-expression in pure expression context; use lower_hir_expr_to_stmts".into())
        }

        HirExpr::BitfieldConstruct(type_name, fields, _ty, _span) => {
            lower_bitfield_construct(type_name, fields, ctx)
        }

        HirExpr::BitfieldUpdate(type_name, base, fields, _ty, _span) => {
            lower_bitfield_update(type_name, base, fields, ctx)
        }
    }
}

/// Lower a bitfield construction to `((v1 & mask1) << off1) | ((v2 & mask2) << off2) | ...`
fn lower_bitfield_construct(
    type_name: &str,
    fields: &[(String, HirExpr)],
    ctx: &LowerCtx,
) -> Result<MirExpr, String> {
    let mut result = MirExpr::Lit(MirLit::U32(0));

    for (field_name, field_expr) in fields {
        let bf_info = ctx.lookup_bitfield_field(type_name, field_name)
            .ok_or_else(|| format!("unknown bitfield field '{}' in '{}'", field_name, type_name))?;

        let mir_val = lower_hir_expr(field_expr, ctx)?;
        let mask = (1u32 << bf_info.width) - 1;

        // For 1-bit bool fields: select(0u, 1u, val)  →  if val then 1u else 0u
        // At MIR level we just do: u32(val) which maps to a cast, but simpler:
        // just mask the value which already works for u32, and for bool we
        // wrap in a conditional
        let coerced = if bf_info.width == 1 {
            // Bool → u32: select(0u, 1u, val)
            MirExpr::Call(
                "select".to_string(),
                vec![
                    MirExpr::Lit(MirLit::U32(0)),
                    MirExpr::Lit(MirLit::U32(1)),
                    mir_val,
                ],
                MirType::U32,
            )
        } else {
            // Ensure value is u32 (integer literals default to i32)
            let val_ty = mir_val.result_type();
            if val_ty.as_ref() == Some(&MirType::U32) {
                mir_val
            } else {
                MirExpr::Cast(Box::new(mir_val), MirType::U32)
            }
        };

        // (val & mask)
        let masked = MirExpr::BinOp(
            MirBinOp::BitAnd,
            Box::new(coerced),
            Box::new(MirExpr::Lit(MirLit::U32(mask))),
            MirType::U32,
        );

        // (masked << offset)
        let shifted = if bf_info.offset > 0 {
            MirExpr::BinOp(
                MirBinOp::Shl,
                Box::new(masked),
                Box::new(MirExpr::Lit(MirLit::U32(bf_info.offset))),
                MirType::U32,
            )
        } else {
            masked
        };

        // result = result | shifted
        result = MirExpr::BinOp(
            MirBinOp::BitOr,
            Box::new(result),
            Box::new(shifted),
            MirType::U32,
        );
    }

    Ok(result)
}

/// Lower a bitfield functional update to:
/// `(base & ~mask_combined) | ((val1 & mask1) << off1) | ((val2 & mask2) << off2) | ...`
/// where `mask_combined` is the OR of all field masks shifted to their positions.
fn lower_bitfield_update(
    type_name: &str,
    base: &HirExpr,
    fields: &[(String, HirExpr)],
    ctx: &LowerCtx,
) -> Result<MirExpr, String> {
    let mir_base = lower_hir_expr(base, ctx)?;

    // Build the clear-mask: AND-out all fields being updated
    let mut clear_mask: u32 = !0u32; // start with all bits set
    for (field_name, _) in fields {
        let bf_info = ctx.lookup_bitfield_field(type_name, field_name)
            .ok_or_else(|| format!("unknown bitfield field '{}' in '{}'", field_name, type_name))?;
        let field_mask = ((1u32 << bf_info.width) - 1) << bf_info.offset;
        clear_mask &= !field_mask;
    }

    // base & clear_mask
    let mut result = MirExpr::BinOp(
        MirBinOp::BitAnd,
        Box::new(mir_base),
        Box::new(MirExpr::Lit(MirLit::U32(clear_mask))),
        MirType::U32,
    );

    // OR in each updated field
    for (field_name, field_expr) in fields {
        let bf_info = ctx.lookup_bitfield_field(type_name, field_name)
            .ok_or_else(|| format!("unknown bitfield field '{}' in '{}'", field_name, type_name))?;

        let mir_val = lower_hir_expr(field_expr, ctx)?;
        let mask = (1u32 << bf_info.width) - 1;

        let coerced = if bf_info.width == 1 {
            MirExpr::Call(
                "select".to_string(),
                vec![
                    MirExpr::Lit(MirLit::U32(0)),
                    MirExpr::Lit(MirLit::U32(1)),
                    mir_val,
                ],
                MirType::U32,
            )
        } else {
            let val_ty = mir_val.result_type();
            if val_ty.as_ref() == Some(&MirType::U32) {
                mir_val
            } else {
                MirExpr::Cast(Box::new(mir_val), MirType::U32)
            }
        };

        let masked = MirExpr::BinOp(
            MirBinOp::BitAnd,
            Box::new(coerced),
            Box::new(MirExpr::Lit(MirLit::U32(mask))),
            MirType::U32,
        );

        let shifted = if bf_info.offset > 0 {
            MirExpr::BinOp(
                MirBinOp::Shl,
                Box::new(masked),
                Box::new(MirExpr::Lit(MirLit::U32(bf_info.offset))),
                MirType::U32,
            )
        } else {
            masked
        };

        result = MirExpr::BinOp(
            MirBinOp::BitOr,
            Box::new(result),
            Box::new(shifted),
            MirType::U32,
        );
    }

    Ok(result)
}

/// Collect arguments from a chain of curried applications.
/// `App(App(App(f, a), b), c)` → `(f, [a, b, c])`
fn collect_app_args(expr: &HirExpr) -> (&HirExpr, Vec<&HirExpr>) {
    let mut args = Vec::new();
    let mut current = expr;
    while let HirExpr::App(func, arg, _, _) = current {
        args.push(arg.as_ref());
        current = func.as_ref();
    }
    args.reverse();
    (current, args)
}

/// Lower the body of a named loop into MIR statements.
///
/// Calls to the loop name become assignments to the loop vars + `continue`.
/// Non-recursive branches become `result_var = expr; break;`.
fn lower_loop_body(
    body: &HirExpr,
    loop_name: &str,
    binding_names: &[String],
    binding_types: &[MirType],
    result_var: &str,
    result_ty: &MirType,
    ctx: &LowerCtx,
) -> Result<Vec<MirStmt>, String> {
    match body {
        // If-expression: recursively handle both branches
        HirExpr::If(cond, then_branch, else_branch, _ty, _span) => {
            let (cond_stmts, cond_expr) = lower_hir_expr_to_stmts(cond, ctx)?;
            let then_stmts = lower_loop_body(
                then_branch, loop_name, binding_names, binding_types,
                result_var, result_ty, ctx,
            )?;
            let else_stmts = lower_loop_body(
                else_branch, loop_name, binding_names, binding_types,
                result_var, result_ty, ctx,
            )?;
            let mut stmts = cond_stmts;
            stmts.push(MirStmt::If(cond_expr, then_stmts, else_stmts));
            Ok(stmts)
        }

        // Let-expression: emit let bindings then recurse into body
        HirExpr::Let(bindings, inner_body, _ty, _span) => {
            let mut stmts = Vec::new();
            for (name, init_expr) in bindings {
                let (mut init_stmts, init_mir) = lower_hir_expr_to_stmts(init_expr, ctx)?;
                let bind_ty = init_mir.result_type().unwrap_or(MirType::I32);
                stmts.append(&mut init_stmts);
                stmts.push(MirStmt::Let(name.clone(), bind_ty, init_mir));
            }
            let mut body_stmts = lower_loop_body(
                inner_body, loop_name, binding_names, binding_types,
                result_var, result_ty, ctx,
            )?;
            stmts.append(&mut body_stmts);
            Ok(stmts)
        }

        // Check if this is a call to the loop name (tail recursion)
        other => {
            let (func, args) = collect_app_args(other);
            if let HirExpr::Var(name, _, _) = func {
                if name == loop_name && args.len() == binding_names.len() {
                    // Tail call: assign new values to loop vars and continue
                    let mut stmts = Vec::new();
                    // Lower all args first (to temp lets) before assigning,
                    // to avoid ordering issues when bindings reference each other
                    let mut arg_temps = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        let (mut arg_stmts, arg_mir) = lower_hir_expr_to_stmts(arg, ctx)?;
                        stmts.append(&mut arg_stmts);
                        let tmp = format!("_loop_tmp_{}", i);
                        let tmp_ty = binding_types[i].clone();
                        stmts.push(MirStmt::Let(tmp.clone(), tmp_ty.clone(), arg_mir));
                        arg_temps.push((tmp, tmp_ty));
                    }
                    for (i, (tmp, tmp_ty)) in arg_temps.into_iter().enumerate() {
                        stmts.push(MirStmt::Assign(
                            binding_names[i].clone(),
                            MirExpr::Var(tmp, tmp_ty),
                        ));
                    }
                    stmts.push(MirStmt::Continue);
                    return Ok(stmts);
                }
            }

            // Non-recursive: this is a result expression
            let (mut stmts, expr) = lower_hir_expr_to_stmts(other, ctx)?;
            stmts.push(MirStmt::Assign(result_var.to_string(), expr));
            stmts.push(MirStmt::Break);
            Ok(stmts)
        }
    }
}

/// Lower an application with already-lowered arguments into a MIR expression.
fn lower_app_with_args(
    func_name: &str,
    mir_args: Vec<MirExpr>,
    mir_ty: MirType,
    ctx: &LowerCtx,
) -> Result<MirExpr, String> {
    match (func_name, mir_args.as_slice()) {
        ("negate", [arg]) => Ok(MirExpr::UnaryOp(
            MirUnaryOp::Neg,
            Box::new(arg.clone()),
            mir_ty,
        )),
        ("mod", [lhs, rhs]) => {
            let div = MirExpr::BinOp(
                MirBinOp::Div,
                Box::new(lhs.clone()),
                Box::new(rhs.clone()),
                mir_ty.clone(),
            );
            let floored = MirExpr::Call("floor".to_string(), vec![div], mir_ty.clone());
            let product = MirExpr::BinOp(
                MirBinOp::Mul,
                Box::new(rhs.clone()),
                Box::new(floored),
                mir_ty.clone(),
            );
            Ok(MirExpr::BinOp(
                MirBinOp::Sub,
                Box::new(lhs.clone()),
                Box::new(product),
                mir_ty,
            ))
        }
        ("atan", [y, x]) | ("atan2", [y, x]) => Ok(MirExpr::Call(
            "atan2".to_string(),
            vec![y.clone(), x.clone()],
            mir_ty,
        )),
        ("load", [arg]) => {
            // load is identity — just pass through the argument
            Ok(arg.clone())
        }
        ("toF32", [arg]) => Ok(MirExpr::Cast(Box::new(arg.clone()), MirType::F32)),
        ("toI32", [arg]) => Ok(MirExpr::Cast(Box::new(arg.clone()), MirType::I32)),
        ("toU32", [arg]) => Ok(MirExpr::Cast(Box::new(arg.clone()), MirType::U32)),
        ("toBool", [arg]) => Ok(MirExpr::Cast(Box::new(arg.clone()), MirType::Bool)),
        ("splat2", [arg]) => Ok(MirExpr::Call(
            "vec2".to_string(),
            vec![arg.clone(), arg.clone()],
            mir_ty,
        )),
        ("splat3", [arg]) => Ok(MirExpr::Call(
            "vec3".to_string(),
            vec![arg.clone(), arg.clone(), arg.clone()],
            mir_ty,
        )),
        ("splat4", [arg]) => Ok(MirExpr::Call(
            "vec4".to_string(),
            vec![arg.clone(), arg.clone(), arg.clone(), arg.clone()],
            mir_ty,
        )),
        ("vecX", [arg]) => Ok(MirExpr::FieldAccess(
            Box::new(arg.clone()),
            "x".to_string(),
            mir_ty,
        )),
        ("vecY", [arg]) => Ok(MirExpr::FieldAccess(
            Box::new(arg.clone()),
            "y".to_string(),
            mir_ty,
        )),
        ("vecZ", [arg]) => Ok(MirExpr::FieldAccess(
            Box::new(arg.clone()),
            "z".to_string(),
            mir_ty,
        )),
        ("vecW", [arg]) => Ok(MirExpr::FieldAccess(
            Box::new(arg.clone()),
            "w".to_string(),
            mir_ty,
        )),
        _ => {
            // Check if this is a constructor call for a sum type with fields
            if let Some((dt_name, tag, _fields)) = ctx.constructors.get(func_name) {
                if ctx.is_sum_type(dt_name) && ctx.has_fields(dt_name) {
                    let mut all_args = vec![MirExpr::Lit(MirLit::U32(*tag))];
                    all_args.extend(mir_args);
                    return Ok(MirExpr::ConstructStruct(dt_name.clone(), all_args));
                }
            }
            Ok(MirExpr::Call(func_name.to_string(), mir_args, mir_ty))
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

/// Check if a pattern is compatible with WGSL switch emission (int lits, wilds, or-patterns of int lits).
fn is_switch_compatible_pattern(pat: &HirPattern) -> bool {
    match pat {
        HirPattern::Lit(HirLit::Int(_)) | HirPattern::Lit(HirLit::UInt(_)) => true,
        HirPattern::Wild | HirPattern::Var(_, _) => true,
        HirPattern::Or(alts) => alts.iter().all(|p| is_switch_compatible_pattern(p)),
        _ => false,
    }
}

/// Lower case arms into a chain of if-else statements (or a switch statement
/// when all arms are integer literal patterns on an I32/U32 scrutinee).
fn lower_case_arms(
    scrut_name: &str,
    scrut_ty: &MirType,
    arms: &[HirCaseArm],
    result_name: &str,
    ctx: &LowerCtx,
) -> Result<Option<MirStmt>, String> {
    if arms.is_empty() {
        return Ok(None);
    }

    // --- Try to emit a native WGSL switch for integer literal patterns ---
    if matches!(scrut_ty, MirType::I32 | MirType::U32) {
        let all_int_or_wild = arms.iter().all(|arm| is_switch_compatible_pattern(&arm.pattern));

        if all_int_or_wild {
            let mut cases: Vec<MirSwitchCase> = Vec::new();
            let mut default_body: Vec<MirStmt> = Vec::new();

            for arm in arms {
                let (mut body_stmts, body_expr) = lower_hir_expr_to_stmts(&arm.body, ctx)?;
                body_stmts.push(MirStmt::Assign(result_name.to_string(), body_expr));

                match &arm.pattern {
                    HirPattern::Lit(HirLit::Int(v)) => {
                        let mir_lit = match scrut_ty {
                            MirType::U32 => MirLit::U32(*v as u32),
                            _ => MirLit::I32(*v as i32),
                        };
                        cases.push(MirSwitchCase {
                            values: vec![mir_lit],
                            body: body_stmts,
                        });
                    }
                    HirPattern::Lit(HirLit::UInt(v)) => {
                        cases.push(MirSwitchCase {
                            values: vec![MirLit::U32(*v as u32)],
                            body: body_stmts,
                        });
                    }
                    HirPattern::Or(alts) => {
                        let values: Vec<MirLit> = alts
                            .iter()
                            .filter_map(|p| match p {
                                HirPattern::Lit(HirLit::Int(v)) => Some(match scrut_ty {
                                    MirType::U32 => MirLit::U32(*v as u32),
                                    _ => MirLit::I32(*v as i32),
                                }),
                                HirPattern::Lit(HirLit::UInt(v)) => Some(MirLit::U32(*v as u32)),
                                _ => None,
                            })
                            .collect();
                        if values.is_empty() {
                            default_body = body_stmts;
                        } else {
                            cases.push(MirSwitchCase {
                                values,
                                body: body_stmts,
                            });
                        }
                    }
                    HirPattern::Wild | HirPattern::Var(_, _) => {
                        default_body = body_stmts;
                    }
                    _ => unreachable!(),
                }
            }

            let scrut_expr = MirExpr::Var(scrut_name.to_string(), scrut_ty.clone());
            return Ok(Some(MirStmt::Switch(scrut_expr, cases, default_body)));
        }
    }

    // --- Fallback: build from last to first (fold right) as if-else chain ---
    let mut result: Option<MirStmt> = None;

    for arm in arms.iter().rev() {
        let (mut body_stmts, body_expr) = lower_hir_expr_to_stmts(&arm.body, ctx)?;
        body_stmts.push(MirStmt::Assign(result_name.to_string(), body_expr));

        match &arm.pattern {
            HirPattern::Wild | HirPattern::Var(_, _) => {
                result = Some(MirStmt::Block(body_stmts));
            }

            HirPattern::Constructor(con_name, tag, sub_pats) => {
                // Bind pattern variables from the scrutinee's fields
                let mut bindings = Vec::new();
                if let Some((dt_name, _, con_fields)) = ctx.constructors.get(con_name.as_str()) {
                    if ctx.is_sum_type(dt_name) && ctx.has_fields(dt_name) {
                        // Sum type with fields: extract from struct fields
                        for (i, pat) in sub_pats.iter().enumerate() {
                            if let HirPattern::Var(var_name, var_ty) = pat {
                                let field_name = if i < con_fields.len() {
                                    con_fields[i].name.clone()
                                } else {
                                    format!("field{}", i)
                                };
                                let mir_ty = ty_to_mir_type_with_ctx(var_ty, Some(ctx)).unwrap_or(MirType::F32);
                                bindings.push(MirStmt::Let(
                                    var_name.clone(),
                                    mir_ty.clone(),
                                    MirExpr::FieldAccess(
                                        Box::new(MirExpr::Var(scrut_name.to_string(), scrut_ty.clone())),
                                        field_name,
                                        mir_ty,
                                    ),
                                ));
                            }
                        }
                    }
                }

                // Prepend bindings to the body
                let mut full_body = bindings;
                full_body.append(&mut body_stmts);

                // Match on tag: if (scrut.tag == tag) { ... }
                let scrut_tag = if scrut_ty == &MirType::U32 {
                    MirExpr::Var(scrut_name.to_string(), scrut_ty.clone())
                } else {
                    MirExpr::FieldAccess(
                        Box::new(MirExpr::Var(scrut_name.to_string(), scrut_ty.clone())),
                        "tag".to_string(),
                        MirType::U32,
                    )
                };
                let cond = MirExpr::BinOp(
                    MirBinOp::Eq,
                    Box::new(scrut_tag),
                    Box::new(MirExpr::Lit(MirLit::U32(*tag))),
                    MirType::Bool,
                );

                let else_stmts = match result {
                    Some(MirStmt::If(c, t, e)) => vec![MirStmt::If(c, t, e)],
                    Some(MirStmt::Block(stmts)) => stmts,
                    Some(other) => vec![other],
                    None => vec![],
                };

                result = Some(MirStmt::If(cond, full_body, else_stmts));
            }

            HirPattern::Lit(hir_lit) => {
                let mir_lit = match hir_lit {
                    HirLit::Int(v) => match scrut_ty {
                        MirType::U32 => MirLit::U32(*v as u32),
                        MirType::F32 => MirLit::F32(*v as f64),
                        _ => MirLit::I32(*v as i32),
                    },
                    HirLit::UInt(v) => MirLit::U32(*v as u32),
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

            HirPattern::Or(alts) => {
                // Build OR chain: scrut == alt1 || scrut == alt2 || ...
                let mut cond: Option<MirExpr> = None;
                for alt in alts {
                    if let HirPattern::Lit(hir_lit) = alt {
                        let mir_lit = match hir_lit {
                            HirLit::Int(v) => match scrut_ty {
                                MirType::U32 => MirLit::U32(*v as u32),
                                MirType::F32 => MirLit::F32(*v as f64),
                                _ => MirLit::I32(*v as i32),
                            },
                            HirLit::UInt(v) => MirLit::U32(*v as u32),
                            HirLit::Float(v) => MirLit::F32(*v),
                            HirLit::Bool(v) => MirLit::Bool(*v),
                        };
                        let eq = MirExpr::BinOp(
                            MirBinOp::Eq,
                            Box::new(MirExpr::Var(scrut_name.to_string(), scrut_ty.clone())),
                            Box::new(MirExpr::Lit(mir_lit)),
                            MirType::Bool,
                        );
                        cond = Some(match cond {
                            Some(prev) => MirExpr::BinOp(
                                MirBinOp::Or,
                                Box::new(prev),
                                Box::new(eq),
                                MirType::Bool,
                            ),
                            None => eq,
                        });
                    }
                }
                if let Some(cond) = cond {
                    let else_stmts = match result {
                        Some(MirStmt::If(c, t, e)) => vec![MirStmt::If(c, t, e)],
                        Some(MirStmt::Block(stmts)) => stmts,
                        Some(other) => vec![other],
                        None => vec![],
                    };
                    result = Some(MirStmt::If(cond, body_stmts, else_stmts));
                } else {
                    // Or-pattern with only wilds
                    result = Some(MirStmt::Block(body_stmts));
                }
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
        HirLit::UInt(v) => MirLit::U32(*v as u32),
        HirLit::Float(v) => MirLit::F32(*v),
        HirLit::Bool(v) => MirLit::Bool(*v),
    }
}

/// Lower a HIR resource declaration to a MIR global binding.
fn lower_hir_resource(res: &HirResource, ctx: &LowerCtx) -> Option<MirGlobal> {
    // Parse the resource type: Uniform<T> or Storage<ReadWrite, T>
    // The `ty` from the semantic analyzer is the full applied type.
    // We need to extract the inner type and determine the address space.
    let (address_space, inner_ty) = parse_resource_type(&res.ty, &res.address_space);
    let mir_ty = ty_to_mir_type_with_ctx(&inner_ty, Some(ctx)).ok()?;
    Some(MirGlobal {
        name: res.name.clone(),
        address_space,
        ty: mir_ty,
        group: res.group,
        binding: res.binding,
    })
}

/// Parse a resource type to extract address space and inner type.
fn parse_resource_type(ty: &Ty, hint: &str) -> (AddressSpace, Ty) {
    // Try to extract inner type from App(Con("Uniform"), inner)
    // or App(App(Con("Storage"), mode), inner)
    if let Ty::App(f, inner) = ty {
        if let Ty::Con(name) = f.as_ref() {
            if name == "Uniform" {
                return (AddressSpace::Uniform, *inner.clone());
            }
        }
        if let Ty::App(ff, _mode) = f.as_ref() {
            if let Ty::Con(name) = ff.as_ref() {
                if name == "Storage" {
                    return (AddressSpace::StorageReadWrite, *inner.clone());
                }
            }
        }
    }
    // Fallback based on hint string
    let space = if hint.contains("Uniform") {
        AddressSpace::Uniform
    } else {
        AddressSpace::StorageReadWrite
    };
    (space, ty.clone())
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

fn default_expr_for_type(ty: &MirType) -> MirExpr {
    match ty {
        MirType::I32 | MirType::U32 | MirType::F32 | MirType::Bool => {
            MirExpr::Lit(default_lit_for_type(ty))
        }
        MirType::Vec(n, inner) => MirExpr::Call(
            format!("vec{}", n),
            (0..*n).map(|_| default_expr_for_type(inner)).collect(),
            ty.clone(),
        ),
        MirType::Mat(cols, rows, inner) => MirExpr::Call(
            format!("mat{}x{}", cols, rows),
            (0..(u32::from(*cols) * u32::from(*rows)))
                .map(|_| default_expr_for_type(inner))
                .collect(),
            ty.clone(),
        ),
        _ => MirExpr::Lit(MirLit::I32(0)),
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
                comments: vec![],
            }],
            data_types: vec![],
            entry_points: vec![],
            resources: vec![],
            bitfields: vec![],
            constants: vec![],
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
                comments: vec![],
            }],
            data_types: vec![],
            entry_points: vec![],
            resources: vec![],
            bitfields: vec![],
            constants: vec![],
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
    fn test_ty_to_mir_type_tensor() {
        let ty = fwgsl_typechecker::tensor_ty(Ty::Nat(4), Ty::f32());
        assert_eq!(
            ty_to_mir_type(&ty),
            Ok(MirType::Array(Box::new(MirType::F32), 4))
        );
    }

    #[test]
    fn test_ty_to_mir_type_nested_tensor() {
        let ty = fwgsl_typechecker::tensor_ty(
            Ty::Nat(2),
            fwgsl_typechecker::tensor_ty(Ty::Nat(4), Ty::f32()),
        );
        assert_eq!(
            ty_to_mir_type(&ty),
            Ok(MirType::Array(
                Box::new(MirType::Array(Box::new(MirType::F32), 4)),
                2,
            ))
        );
    }

    #[test]
    fn test_ty_aliases_normalize_before_mir_lowering() {
        let ty = Ty::app(Ty::app(Ty::Con("Array".into()), Ty::Nat(8)), Ty::f32());
        assert_eq!(
            ty_to_mir_type(&ty),
            Ok(MirType::Array(Box::new(MirType::F32), 8))
        );
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
                comments: vec![],
            }],
            data_types: vec![],
            entry_points: vec![],
            resources: vec![],
            bitfields: vec![],
            constants: vec![],
        };

        let mir = lower_hir_to_mir(&hir).expect("lowering should succeed");
        assert_eq!(mir.functions.len(), 1);
        let f = &mir.functions[0];
        // Should have a let statement for x
        assert!(!f.body.is_empty());
    }
}
