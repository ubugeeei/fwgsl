//! AST → HIR lowering for fwgsl.
//!
//! Converts the parser's AST into a type-annotated HIR by re-running
//! type inference (mirroring the semantic analyzer) while simultaneously
//! building HIR nodes.

use std::collections::HashMap;

use fwgsl_hir::*;
use fwgsl_parser::parser::*;
use fwgsl_span::Span;
use fwgsl_typechecker::*;

/// Lowers a parsed AST `Program` into an `HirProgram`.
///
/// Requires a `SemanticAnalyzer` that has already run `analyze()` so that
/// constructor and data-type information is available.
pub struct AstLowering {
    pub env: TypeEnv,
    pub engine: InferEngine,
    pub constructors: HashMap<String, ConstructorInfo>,
    pub data_types: HashMap<String, fwgsl_semantic::DataTypeInfo>,
    pub type_aliases: HashMap<String, Ty>,
}

impl AstLowering {
    /// Create a new lowering context from a completed semantic analyzer.
    pub fn new(sa: &fwgsl_semantic::SemanticAnalyzer) -> Self {
        let mut engine = InferEngine::new();
        if let Some(max_var_id) = sa.env.max_var_id() {
            engine.reserve_above(max_var_id + 1);
        }
        let mut lowering = Self {
            env: sa.env.clone(),
            engine,
            constructors: sa.constructors.clone(),
            data_types: sa.data_types.clone(),
            type_aliases: sa.type_aliases.clone(),
        };
        lowering.add_builtins();
        lowering
    }

    fn add_builtins(&mut self) {
        let scalar = fresh_var_id(&mut self.engine);
        let numeric_binop = Scheme::poly(
            vec![scalar],
            Ty::arrow(Ty::Var(scalar), Ty::arrow(Ty::Var(scalar), Ty::Var(scalar))),
        );
        for op in ["+", "-", "*", "/", "%"] {
            self.env.insert(op.to_string(), numeric_binop.clone());
        }
        let scalar = fresh_var_id(&mut self.engine);
        let numeric_cmp = Scheme::poly(
            vec![scalar],
            Ty::arrow(Ty::Var(scalar), Ty::arrow(Ty::Var(scalar), Ty::bool())),
        );
        for op in ["==", "/=", "<", ">", "<=", ">="] {
            self.env.insert(op.to_string(), numeric_cmp.clone());
        }
        let bool_binop = Scheme::mono(Ty::arrow(Ty::bool(), Ty::arrow(Ty::bool(), Ty::bool())));
        self.env.insert("&&".to_string(), bool_binop.clone());
        self.env.insert("||".to_string(), bool_binop);
    }

    /// Lower the entire program.
    pub fn lower_program(&mut self, program: &Program) -> HirProgram {
        // Pass 1: register data types (re-populate env with constructor types)
        for decl in &program.decls {
            if let Decl::DataDecl {
                name,
                type_params,
                constructors: cons,
                span: _,
            } = decl
            {
                let mut type_scope = self.new_type_var_scope(type_params);
                let scheme_vars = scope_vars(&type_scope);
                let result_ty = apply_type_params(name, type_params, &type_scope);
                for con in cons {
                    let con_ty = match &con.fields {
                        ConFields::Empty => result_ty.clone(),
                        ConFields::Positional(fields) => {
                            let mut ty = result_ty.clone();
                            for field in fields.iter().rev() {
                                let ft =
                                    self.convert_syntax_type_with_scope(field, &mut type_scope);
                                ty = Ty::arrow(ft, ty);
                            }
                            ty
                        }
                        ConFields::Record(fields) => {
                            let mut ty = result_ty.clone();
                            for f in fields.iter().rev() {
                                let ft =
                                    self.convert_syntax_type_with_scope(&f.ty, &mut type_scope);
                                ty = Ty::arrow(ft, ty);
                            }
                            ty
                        }
                    };
                    self.env
                        .insert(con.name.clone(), Scheme::poly(scheme_vars.clone(), con_ty));
                }
            }
        }

        // Pass 2: collect type signatures
        for decl in &program.decls {
            if let Decl::TypeSig { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type_scheme(ty);
                let flattened = Scheme {
                    vars: inferred_ty.vars.clone(),
                    ty: flatten_tuple_arrows(&inferred_ty.ty),
                };
                self.env.insert(name.clone(), flattened);
            }
            if let Decl::ConstDecl { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type_scheme(ty);
                self.env.insert(name.clone(), inferred_ty);
            }
        }

        let mut functions = Vec::new();
        let mut data_types = Vec::new();
        let mut entry_points = Vec::new();
        let mut resources = Vec::new();
        let mut bitfields = Vec::new();
        let mut constants = Vec::new();

        for decl in &program.decls {
            match decl {
                Decl::FunDecl {
                    name,
                    params,
                    body,
                    where_binds,
                    span,
                    ..
                } => {
                    if let Some(f) = self.lower_fun_decl(name, params, body, where_binds, *span) {
                        functions.push(f);
                    }
                }
                Decl::EntryPoint {
                    attributes,
                    name,
                    params,
                    body,
                    span,
                } => {
                    if let Some(ep) = self.lower_entry_point(attributes, name, params, body, *span)
                    {
                        entry_points.push(ep);
                    }
                }
                Decl::DataDecl {
                    name, constructors, ..
                } => {
                    data_types.push(self.lower_data_decl(name, constructors));
                }
                Decl::ResourceDecl {
                    name,
                    ty,
                    group,
                    binding,
                    ..
                } => {
                    let scheme = self.convert_syntax_type_scheme(ty);
                    resources.push(fwgsl_hir::HirResource {
                        name: name.clone(),
                        ty: scheme.ty,
                        address_space: self.extract_address_space(ty),
                        group: *group,
                        binding: *binding,
                    });
                }
                Decl::BitfieldDecl {
                    name,
                    base_ty,
                    fields,
                    ..
                } => {
                    let base_scheme = self.convert_syntax_type_scheme(base_ty);
                    let mut offset = 0u32;
                    let hir_fields: Vec<fwgsl_hir::HirBitfieldField> = fields
                        .iter()
                        .map(|f| {
                            let hf = fwgsl_hir::HirBitfieldField {
                                name: f.name.clone(),
                                offset,
                                width: f.width,
                            };
                            offset += f.width;
                            hf
                        })
                        .collect();
                    bitfields.push(fwgsl_hir::HirBitfield {
                        name: name.clone(),
                        base_ty: base_scheme.ty,
                        fields: hir_fields,
                    });
                }
                Decl::ConstDecl {
                    name,
                    ty,
                    value,
                    span,
                } => {
                    let scheme = self.convert_syntax_type_scheme(ty);
                    let mut local_env = self.env.clone();
                    let (hir_expr, _val_ty) = self.lower_expr(value, &mut local_env);
                    constants.push(fwgsl_hir::HirConst {
                        name: name.clone(),
                        ty: scheme.ty,
                        value: hir_expr,
                        span: *span,
                    });
                }
                Decl::TypeSig { .. } | Decl::TypeAlias { .. } => {}
            }
        }

        HirProgram {
            functions,
            data_types,
            entry_points,
            resources,
            bitfields,
            constants,
        }
    }

    fn lower_fun_decl(
        &mut self,
        name: &str,
        params: &[Pat],
        body: &Expr,
        where_binds: &[(String, Expr)],
        span: Span,
    ) -> Option<HirFunction> {
        let mut local_env = self.env.clone();

        let mut hir_params = Vec::new();
        let mut param_types = Vec::new();
        for pat in params {
            // Flatten tuple patterns into individual parameters
            if let Pat::Tuple(sub_pats, _) = pat {
                for sub_pat in sub_pats {
                    let ty = self.engine.fresh_var();
                    let pname = pat_name(sub_pat);
                    self.bind_pattern(sub_pat, &ty, &mut local_env);
                    param_types.push(ty.clone());
                    hir_params.push((pname, ty));
                }
            } else {
                let ty = self.engine.fresh_var();
                let pname = pat_name(pat);
                self.bind_pattern(pat, &ty, &mut local_env);
                param_types.push(ty.clone());
                hir_params.push((pname, ty));
            }
        }

        let body = desugar_where(body, where_binds, span);
        let (hir_body, body_ty) = self.lower_expr(&body, &mut local_env);

        // Build function type and unify with declared type
        let mut fun_ty = body_ty.clone();
        for pt in param_types.iter().rev() {
            fun_ty = Ty::arrow(pt.clone(), fun_ty);
        }

        if let Some(scheme) = self.env.lookup(name) {
            let declared = self.engine.instantiate(scheme);
            self.engine.unify(&fun_ty, &declared, span);
        }

        // Finalize types
        let final_params: Vec<(String, Ty)> = hir_params
            .into_iter()
            .map(|(n, ty)| (n, self.engine.finalize(&ty)))
            .collect();
        let return_ty = self.engine.finalize(&body_ty);
        let body = self.finalize_expr(hir_body);

        Some(HirFunction {
            name: name.to_string(),
            params: final_params,
            return_ty,
            body,
            span,
        })
    }

    fn lower_entry_point(
        &mut self,
        attributes: &[Attribute],
        name: &str,
        params: &[Pat],
        body: &Expr,
        span: Span,
    ) -> Option<HirEntryPoint> {
        let mut local_env = self.env.clone();

        let mut hir_params = Vec::new();
        let mut param_types = Vec::new();
        for pat in params {
            if let Pat::Tuple(sub_pats, _) = pat {
                for sub_pat in sub_pats {
                    let ty = self.engine.fresh_var();
                    let pname = pat_name(sub_pat);
                    self.bind_pattern(sub_pat, &ty, &mut local_env);
                    param_types.push(ty.clone());
                    hir_params.push((pname, ty));
                }
            } else {
                let ty = self.engine.fresh_var();
                let pname = pat_name(pat);
                self.bind_pattern(pat, &ty, &mut local_env);
                param_types.push(ty.clone());
                hir_params.push((pname, ty));
            }
        }

        let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);

        let mut fun_ty = body_ty.clone();
        for pt in param_types.iter().rev() {
            fun_ty = Ty::arrow(pt.clone(), fun_ty);
        }

        if let Some(scheme) = self.env.lookup(name) {
            let declared = self.engine.instantiate(scheme);
            self.engine.unify(&fun_ty, &declared, span);
        }

        let final_params: Vec<(String, Ty)> = hir_params
            .into_iter()
            .map(|(n, ty)| (n, self.engine.finalize(&ty)))
            .collect();
        let return_ty = self.engine.finalize(&body_ty);
        let body = self.finalize_expr(hir_body);

        let hir_attrs = attributes
            .iter()
            .map(|a| HirAttribute {
                name: a.name.clone(),
                args: a.args.clone(),
            })
            .collect();

        Some(HirEntryPoint {
            name: name.to_string(),
            attributes: hir_attrs,
            params: final_params,
            return_ty,
            body,
            span,
        })
    }

    fn lower_data_decl(&self, name: &str, cons: &[ConDecl]) -> HirDataType {
        let mut hir_cons = Vec::new();
        for (tag, con) in cons.iter().enumerate() {
            let fields = match &con.fields {
                ConFields::Empty => vec![],
                ConFields::Positional(tys) => tys
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        let ty = self.convert_syntax_type_pure(t);
                        HirFieldDef { name: format!("field{}", i), ty, attributes: vec![] }
                    })
                    .collect(),
                ConFields::Record(fields) => fields
                    .iter()
                    .map(|f| {
                        let ty = self.convert_syntax_type_pure(&f.ty);
                        let attrs = f.attributes.iter().map(|a| HirAttribute {
                            name: a.name.clone(),
                            args: a.args.clone(),
                        }).collect();
                        HirFieldDef { name: f.name.clone(), ty, attributes: attrs }
                    })
                    .collect(),
            };
            hir_cons.push(HirConstructor {
                name: con.name.clone(),
                tag: tag as u32,
                fields,
            });
        }
        HirDataType {
            name: name.to_string(),
            constructors: hir_cons,
        }
    }

    /// Lower an expression, returning the HIR expression and its inferred type.
    fn lower_expr(&mut self, expr: &Expr, env: &mut TypeEnv) -> (HirExpr, Ty) {
        match expr {
            Expr::Lit(lit, span) => {
                let (hir_lit, ty) = self.lower_lit(lit);
                (HirExpr::Lit(hir_lit, ty.clone(), *span), ty)
            }

            Expr::Var(name, span) => {
                let ty = if let Some(scheme) = env.lookup(name) {
                    self.engine.instantiate(scheme)
                } else {
                    Ty::Error
                };
                (HirExpr::Var(name.clone(), ty.clone(), *span), ty)
            }

            Expr::Con(name, span) => {
                // A constructor used as a value. If it takes no arguments, it's
                // a nullary constructor call; otherwise it's just a variable
                // reference (will be applied later).
                let ty = if let Some(scheme) = env.lookup(name) {
                    self.engine.instantiate(scheme)
                } else {
                    Ty::Error
                };

                if let Some(con_info) = self.constructors.get(name).cloned() {
                    match &con_info.fields {
                        ConstructorFields::Empty => (
                            HirExpr::ConstructorCall(
                                name.clone(),
                                con_info.tag,
                                vec![],
                                ty.clone(),
                                *span,
                            ),
                            ty,
                        ),
                        _ => (HirExpr::Var(name.clone(), ty.clone(), *span), ty),
                    }
                } else {
                    (HirExpr::Var(name.clone(), ty.clone(), *span), ty)
                }
            }

            Expr::App(func, arg, span) => {
                // Beta-reduce: App(Lambda([p], body), arg) → Let([(p, arg)], body)
                // Unwrap Paren wrappers to find the underlying Lambda
                let unwrapped_func = {
                    let mut f = func.as_ref();
                    while let Expr::Paren(inner, _) = f {
                        f = inner.as_ref();
                    }
                    f
                };
                if let Expr::Lambda(pats, body, lam_span) = unwrapped_func {
                    if let Some((first_pat, rest_pats)) = pats.split_first() {
                        let mut local_env = env.clone();
                        let param_ty = self.engine.fresh_var();
                        let (hir_arg, arg_ty) = self.lower_expr(arg, &mut local_env);
                        self.engine.unify(&param_ty, &arg_ty, *span);

                        // Extract variable name from pattern for Let binding
                        let param_name = match first_pat {
                            Pat::Var(name, _) => name.clone(),
                            Pat::Wild(_) => "_lambda_param".to_string(),
                            _ => "_lambda_param".to_string(),
                        };
                        self.bind_pattern(first_pat, &param_ty, &mut local_env);

                        if rest_pats.is_empty() {
                            // Single-param lambda: Let([(name, arg)], body)
                            let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);
                            (
                                HirExpr::Let(
                                    vec![(param_name, hir_arg)],
                                    Box::new(hir_body),
                                    body_ty.clone(),
                                    *span,
                                ),
                                body_ty,
                            )
                        } else {
                            // Multi-param lambda: reduce first param, recurse on remaining
                            let inner_lambda = Expr::Lambda(
                                rest_pats.to_vec(),
                                body.clone(),
                                *lam_span,
                            );
                            let (hir_body, body_ty) =
                                self.lower_expr(&inner_lambda, &mut local_env);
                            (
                                HirExpr::Let(
                                    vec![(param_name, hir_arg)],
                                    Box::new(hir_body),
                                    body_ty.clone(),
                                    *span,
                                ),
                                body_ty,
                            )
                        }
                    } else {
                        // Empty lambda params — shouldn't happen, fall through to normal App
                        let (hir_func, func_ty) = self.lower_expr(func, env);
                        let (hir_arg, arg_ty) = self.lower_expr(arg, env);
                        let ret_ty = self.engine.fresh_var();
                        let expected = Ty::arrow(arg_ty, ret_ty.clone());
                        self.engine.unify(&func_ty, &expected, *span);
                        (
                            HirExpr::App(
                                Box::new(hir_func),
                                Box::new(hir_arg),
                                ret_ty.clone(),
                                *span,
                            ),
                            ret_ty,
                        )
                    }
                } else {
                    // Flatten tuple arguments: f (a, b, c) => f a b c
                    if let Expr::Tuple(elems, _) = arg.as_ref() {
                        if !elems.is_empty() {
                            let (mut expr, mut cur_ty) = self.lower_expr(func, env);
                            for elem in elems {
                                let (hir_arg, arg_ty) = self.lower_expr(elem, env);
                                let ret_ty = self.engine.fresh_var();
                                let expected = Ty::arrow(arg_ty, ret_ty.clone());
                                self.engine.unify(&cur_ty, &expected, *span);
                                expr = HirExpr::App(
                                    Box::new(expr),
                                    Box::new(hir_arg),
                                    ret_ty.clone(),
                                    *span,
                                );
                                cur_ty = ret_ty;
                            }
                            return (expr, cur_ty);
                        }
                    }
                    let (hir_func, func_ty) = self.lower_expr(func, env);
                    let (hir_arg, arg_ty) = self.lower_expr(arg, env);
                    let ret_ty = self.engine.fresh_var();
                    let expected = Ty::arrow(arg_ty, ret_ty.clone());
                    self.engine.unify(&func_ty, &expected, *span);
                    (
                        HirExpr::App(
                            Box::new(hir_func),
                            Box::new(hir_arg),
                            ret_ty.clone(),
                            *span,
                        ),
                        ret_ty,
                    )
                }
            }

            Expr::Infix(lhs, op, rhs, span) => {
                // Desugar infix to BinOp if it's a known operator
                if let Some(binop) = BinOp::parse(op) {
                    let (hir_lhs, lhs_ty) = self.lower_expr(lhs, env);
                    let (hir_rhs, rhs_ty) = self.lower_expr(rhs, env);

                    // Get operator type and unify
                    let op_ty = if let Some(scheme) = env.lookup(op) {
                        self.engine.instantiate(scheme)
                    } else {
                        Ty::Error
                    };
                    let ret_ty = self.engine.fresh_var();
                    self.engine.unify(
                        &op_ty,
                        &Ty::arrow(lhs_ty, Ty::arrow(rhs_ty, ret_ty.clone())),
                        *span,
                    );
                    (
                        HirExpr::BinOp(
                            binop,
                            Box::new(hir_lhs),
                            Box::new(hir_rhs),
                            ret_ty.clone(),
                            *span,
                        ),
                        ret_ty,
                    )
                } else {
                    // Desugar as function application: (op lhs) rhs
                    let op_ty = if let Some(scheme) = env.lookup(op) {
                        self.engine.instantiate(scheme)
                    } else {
                        Ty::Error
                    };
                    let (hir_lhs, lhs_ty) = self.lower_expr(lhs, env);
                    let (hir_rhs, rhs_ty) = self.lower_expr(rhs, env);
                    let ret_ty = self.engine.fresh_var();
                    self.engine.unify(
                        &op_ty,
                        &Ty::arrow(lhs_ty, Ty::arrow(rhs_ty, ret_ty.clone())),
                        *span,
                    );
                    let op_expr = HirExpr::Var(op.clone(), op_ty, *span);
                    let app1_ty = self.engine.fresh_var();
                    let app1 = HirExpr::App(Box::new(op_expr), Box::new(hir_lhs), app1_ty, *span);
                    (
                        HirExpr::App(Box::new(app1), Box::new(hir_rhs), ret_ty.clone(), *span),
                        ret_ty,
                    )
                }
            }

            Expr::Lambda(pats, body, _span) => {
                let mut local_env = env.clone();
                let mut param_types = Vec::new();
                for pat in pats {
                    let ty = self.engine.fresh_var();
                    self.bind_pattern(pat, &ty, &mut local_env);
                    param_types.push(ty);
                }
                let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);
                let mut result_ty = body_ty;
                for pt in param_types.into_iter().rev() {
                    result_ty = Ty::arrow(pt, result_ty);
                }
                // Lambda is desugared: for now, emit as the body directly
                // (lambdas in fwgsl source get applied away or become function params)
                (hir_body, result_ty)
            }

            Expr::Let(binds, body, span) => {
                let mut local_env = env.clone();
                let mut hir_binds = Vec::new();
                for (name, expr) in binds {
                    let (hir_expr, ty) = self.lower_expr(expr, &mut local_env);
                    local_env.insert(name.clone(), Scheme::mono(ty.clone()));
                    hir_binds.push((name.clone(), hir_expr));
                }
                let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);
                (
                    HirExpr::Let(hir_binds, Box::new(hir_body), body_ty.clone(), *span),
                    body_ty,
                )
            }

            Expr::Case(scrutinee, arms, span) => {
                let (hir_scrut, scrut_ty) = self.lower_expr(scrutinee, env);
                let result_ty = self.engine.fresh_var();

                let mut hir_arms = Vec::new();
                for (pat, body) in arms {
                    let mut arm_env = env.clone();
                    self.bind_pattern(pat, &scrut_ty, &mut arm_env);
                    let hir_pattern = self.lower_pattern(pat, &scrut_ty);
                    let (hir_body, body_ty) = self.lower_expr(body, &mut arm_env);
                    self.engine.unify(&result_ty, &body_ty, *span);
                    hir_arms.push(HirCaseArm {
                        pattern: hir_pattern,
                        body: hir_body,
                    });
                }

                (
                    HirExpr::Case(Box::new(hir_scrut), hir_arms, result_ty.clone(), *span),
                    result_ty,
                )
            }

            Expr::If(cond, then_expr, else_expr, span) => {
                let (hir_cond, cond_ty) = self.lower_expr(cond, env);
                self.engine.unify(&cond_ty, &Ty::bool(), *span);
                let (hir_then, then_ty) = self.lower_expr(then_expr, env);
                let (hir_else, else_ty) = self.lower_expr(else_expr, env);
                self.engine.unify(&then_ty, &else_ty, *span);
                (
                    HirExpr::If(
                        Box::new(hir_cond),
                        Box::new(hir_then),
                        Box::new(hir_else),
                        then_ty.clone(),
                        *span,
                    ),
                    then_ty,
                )
            }

            Expr::Paren(inner, _) => self.lower_expr(inner, env),

            Expr::Tuple(elems, span) => {
                if elems.is_empty() {
                    (HirExpr::Lit(HirLit::Int(0), Ty::unit(), *span), Ty::unit())
                } else {
                    // Lower all elements and return the last one as a fallback.
                    // Tuple expressions used as function arguments are handled
                    // by the App flattening above; this path is for standalone tuples.
                    let mut last = None;
                    for e in elems {
                        last = Some(self.lower_expr(e, env));
                    }
                    last.unwrap()
                }
            }

            Expr::Record(fields, span) => {
                // Lower record fields; for now just emit first field value
                if let Some((_, expr)) = fields.first() {
                    self.lower_expr(expr, env)
                } else {
                    (HirExpr::Lit(HirLit::Int(0), Ty::unit(), *span), Ty::unit())
                }
            }

            Expr::FieldAccess(expr, field, span) => {
                let (hir_expr, expr_ty) = self.lower_expr(expr, env);
                let expr_ty_final = self.engine.finalize(&expr_ty);

                // Check for Vec swizzle patterns
                let result_ty = if fwgsl_semantic::is_swizzle(field) {
                    if let Some((n, scalar)) = fwgsl_semantic::extract_vec_type(&expr_ty_final) {
                        if fwgsl_semantic::validate_swizzle(field, n) {
                            if field.len() == 1 {
                                scalar
                            } else {
                                Ty::app(
                                    Ty::app(Ty::Con("Vec".into()), Ty::Nat(field.len() as u64)),
                                    scalar,
                                )
                            }
                        } else {
                            self.engine.fresh_var()
                        }
                    } else {
                        self.engine.fresh_var()
                    }
                } else {
                    self.engine.fresh_var()
                };

                (
                    HirExpr::FieldAccess(
                        Box::new(hir_expr),
                        field.clone(),
                        result_ty.clone(),
                        *span,
                    ),
                    result_ty,
                )
            }

            Expr::Index(base, index, span) => {
                let (hir_base, _base_ty) = self.lower_expr(base, env);
                let (hir_index, _idx_ty) = self.lower_expr(index, env);
                let result_ty = self.engine.fresh_var();
                (
                    HirExpr::Index(
                        Box::new(hir_base),
                        Box::new(hir_index),
                        result_ty.clone(),
                        *span,
                    ),
                    result_ty,
                )
            }

            Expr::VecLit(elems, span) => {
                // Lower each element, compute total components, emit as vecN call.
                let scalar_ty = self.engine.fresh_var();
                let mut total_components: u64 = 0;
                let mut hir_args = Vec::new();

                for elem in elems {
                    let (hir_elem, elem_ty) = self.lower_expr(elem, env);
                    let elem_ty_final = self.engine.finalize(&elem_ty);

                    if let Some((n, inner_scalar)) =
                        fwgsl_semantic::extract_vec_type(&elem_ty_final)
                    {
                        total_components += n as u64;
                        self.engine.unify(&scalar_ty, &inner_scalar, *span);
                    } else {
                        total_components += 1;
                        self.engine.unify(&scalar_ty, &elem_ty, *span);
                    }

                    hir_args.push(hir_elem);
                }

                // Construct vecN call: App(App(...(Var("vecN"), arg1), arg2), argN)
                let n = total_components.clamp(2, 4);
                let vec_name = format!("$vec{}", n);
                let result_ty = Ty::app(
                    Ty::app(Ty::Con("Vec".into()), Ty::Nat(n)),
                    scalar_ty.clone(),
                );

                // Build curried application chain
                let mut expr = HirExpr::Var(vec_name, result_ty.clone(), *span);
                for arg in hir_args {
                    let app_ty = result_ty.clone();
                    expr = HirExpr::App(Box::new(expr), Box::new(arg), app_ty, *span);
                }

                (expr, result_ty)
            }

            Expr::OpSection(op, span) => {
                let ty = if let Some(scheme) = env.lookup(op) {
                    self.engine.instantiate(scheme)
                } else {
                    Ty::Error
                };
                (HirExpr::Var(op.clone(), ty.clone(), *span), ty)
            }

            Expr::Neg(inner, span) => {
                let (hir_inner, inner_ty) = self.lower_expr(inner, env);
                (
                    HirExpr::UnaryNeg(Box::new(hir_inner), inner_ty.clone(), *span),
                    inner_ty,
                )
            }

            Expr::Do(stmts, span) => {
                // Lower do-notation as sequential let bindings
                let mut local_env = env.clone();
                let mut hir_binds = Vec::new();
                let mut last_expr = None;

                for stmt in stmts {
                    match stmt {
                        DoStmt::Expr(expr, _) => {
                            let (hir_expr, _ty) = self.lower_expr(expr, &mut local_env);
                            last_expr = Some(hir_expr);
                        }
                        DoStmt::Bind(name, expr, _) => {
                            let (hir_expr, ty) = self.lower_expr(expr, &mut local_env);
                            let inner_ty = self.engine.fresh_var();
                            local_env.insert(name.clone(), Scheme::mono(inner_ty));
                            hir_binds.push((name.clone(), hir_expr));
                            last_expr = None;
                            let _ = ty;
                        }
                        DoStmt::Let(name, expr, _) => {
                            let (hir_expr, ty) = self.lower_expr(expr, &mut local_env);
                            local_env.insert(name.clone(), Scheme::mono(ty));
                            hir_binds.push((name.clone(), hir_expr));
                            last_expr = None;
                        }
                    }
                }

                let body = last_expr.unwrap_or(HirExpr::Lit(HirLit::Int(0), Ty::unit(), *span));
                let body_ty = body.ty().clone();
                if hir_binds.is_empty() {
                    (body, body_ty)
                } else {
                    (
                        HirExpr::Let(hir_binds, Box::new(body), body_ty.clone(), *span),
                        body_ty,
                    )
                }
            }

            Expr::Loop(loop_name, bindings, body, span) => {
                // Lower each binding's initial value
                let mut hir_bindings = Vec::new();
                let mut loop_env = env.clone();
                // Build the type of the loop result from the first binding
                // (for single-binding loops), or a tuple of all binding types.
                let mut binding_tys = Vec::new();
                for (bind_name, init_expr) in bindings {
                    let (hir_init, init_ty) = self.lower_expr(init_expr, env);
                    loop_env.insert(bind_name.clone(), Scheme::mono(init_ty.clone()));
                    binding_tys.push(init_ty);
                    hir_bindings.push((bind_name.clone(), hir_init));
                }

                // The result type of the loop is inferred from the body's
                // non-recursive branches (not the tuple of bindings).
                let result_ty = self.engine.fresh_var();
                // The loop name is a function: binding_ty1 -> ... -> result_ty
                let loop_fn_ty = binding_tys.iter().rev().fold(result_ty.clone(), |acc, ty| {
                    Ty::Arrow(Box::new(ty.clone()), Box::new(acc))
                });
                loop_env.insert(loop_name.clone(), Scheme::mono(loop_fn_ty));

                let (hir_body, body_ty) = self.lower_expr(body, &mut loop_env);
                self.engine.unify(&result_ty, &body_ty, *span);
                (
                    HirExpr::Loop(
                        loop_name.clone(),
                        hir_bindings,
                        Box::new(hir_body),
                        result_ty.clone(),
                        *span,
                    ),
                    result_ty,
                )
            }
        }
    }

    fn lower_pattern(&mut self, pat: &Pat, _scrutinee_ty: &Ty) -> HirPattern {
        match pat {
            Pat::Wild(_) => HirPattern::Wild,
            Pat::Var(name, _) => HirPattern::Var(name.clone(), self.engine.finalize(_scrutinee_ty)),
            Pat::Con(name, sub_pats, _) => {
                if let Some(con_info) = self
                    .constructors
                    .get(name)
                    .cloned()
                    .map(|info| info.instantiate(&mut self.engine))
                {
                    let sub_hir: Vec<HirPattern> = match &con_info.fields {
                        ConstructorFields::Positional(field_tys) => sub_pats
                            .iter()
                            .zip(field_tys.iter())
                            .map(|(p, ty)| self.lower_pattern(p, ty))
                            .collect(),
                        ConstructorFields::Record(fields) => sub_pats
                            .iter()
                            .zip(fields.iter())
                            .map(|(p, (_, ty))| self.lower_pattern(p, ty))
                            .collect(),
                        ConstructorFields::Empty => vec![],
                    };
                    HirPattern::Constructor(name.clone(), con_info.tag, sub_hir)
                } else {
                    HirPattern::Wild
                }
            }
            Pat::Lit(lit, _) => {
                let (hir_lit, _) = self.lower_lit(lit);
                HirPattern::Lit(hir_lit)
            }
            Pat::Paren(inner, _) => self.lower_pattern(inner, _scrutinee_ty),
            Pat::Tuple(pats, _) => {
                if let Some(first) = pats.first() {
                    self.lower_pattern(first, _scrutinee_ty)
                } else {
                    HirPattern::Wild
                }
            }
            Pat::Record(con_name, fields, _) => {
                if let Some(con_info) = self
                    .constructors
                    .get(con_name)
                    .cloned()
                    .map(|info| info.instantiate(&mut self.engine))
                {
                    let sub_pats: Vec<HirPattern> = if let ConstructorFields::Record(con_fields) =
                        &con_info.fields
                    {
                        fields
                            .iter()
                            .map(|(fname, maybe_pat)| {
                                let field_ty = con_fields
                                    .iter()
                                    .find(|(n, _)| n == fname)
                                    .map(|(_, ty)| ty.clone())
                                    .unwrap_or(Ty::Error);
                                if let Some(p) = maybe_pat {
                                    self.lower_pattern(p, &field_ty)
                                } else {
                                    HirPattern::Var(fname.clone(), self.engine.finalize(&field_ty))
                                }
                            })
                            .collect()
                    } else {
                        vec![]
                    };
                    HirPattern::Constructor(con_name.clone(), con_info.tag, sub_pats)
                } else {
                    HirPattern::Wild
                }
            }
            Pat::As(name, inner, _) => {
                // As-pattern: for HIR, just use the inner pattern
                // (the name binding is already in the env)
                let _ = name;
                self.lower_pattern(inner, _scrutinee_ty)
            }
        }
    }

    fn lower_lit(&self, lit: &Lit) -> (HirLit, Ty) {
        match lit {
            Lit::Int(v) => (HirLit::Int(*v), Ty::i32()),
            Lit::Float(v) => (HirLit::Float(*v), Ty::f32()),
            Lit::String(_) => (HirLit::Int(0), Ty::Con("String".into())),
            Lit::Char(_) => (HirLit::Int(0), Ty::Con("Char".into())),
        }
    }

    fn bind_pattern(&mut self, pat: &Pat, ty: &Ty, env: &mut TypeEnv) {
        match pat {
            Pat::Var(name, _) => {
                env.insert(name.clone(), Scheme::mono(ty.clone()));
            }
            Pat::Wild(_) => {}
            Pat::Con(name, sub_pats, span) => {
                if let Some(con_info) = self
                    .constructors
                    .get(name)
                    .cloned()
                    .map(|info| info.instantiate(&mut self.engine))
                {
                    self.engine.unify(ty, &con_info.result_ty, *span);
                    match &con_info.fields {
                        ConstructorFields::Positional(field_tys) => {
                            for (pat, field_ty) in sub_pats.iter().zip(field_tys.iter()) {
                                self.bind_pattern(pat, field_ty, env);
                            }
                        }
                        ConstructorFields::Empty => {}
                        ConstructorFields::Record(fields) => {
                            for (pat, (_, field_ty)) in sub_pats.iter().zip(fields.iter()) {
                                self.bind_pattern(pat, field_ty, env);
                            }
                        }
                    }
                }
            }
            Pat::Lit(lit, span) => {
                let lit_ty = match lit {
                    Lit::Int(_) => {
                        // Use a fresh var so integer literals can unify with
                        // both I32 and U32 (determined by the scrutinee type).
                        let v = self.engine.fresh_var();
                        v
                    }
                    Lit::Float(_) => Ty::f32(),
                    Lit::String(_) => Ty::Con("String".into()),
                    Lit::Char(_) => Ty::Con("Char".into()),
                };
                self.engine.unify(ty, &lit_ty, *span);
            }
            Pat::Paren(inner, _) => self.bind_pattern(inner, ty, env),
            Pat::Tuple(pats, _) => {
                let elem_tys: Vec<Ty> = pats.iter().map(|_| self.engine.fresh_var()).collect();
                if !elem_tys.is_empty() {
                    let tuple_ty = Ty::Tuple(elem_tys.clone());
                    self.engine.unify(ty, &tuple_ty, Span::new(0, 0));
                }
                for (pat, elem_ty) in pats.iter().zip(elem_tys.iter()) {
                    self.bind_pattern(pat, elem_ty, env);
                }
            }
            Pat::Record(con_name, fields, span) => {
                if let Some(con_info) = self
                    .constructors
                    .get(con_name)
                    .cloned()
                    .map(|info| info.instantiate(&mut self.engine))
                {
                    self.engine.unify(ty, &con_info.result_ty, *span);
                    if let ConstructorFields::Record(con_fields) = &con_info.fields {
                        for (field_name, maybe_pat) in fields {
                            if let Some((_, field_ty)) =
                                con_fields.iter().find(|(n, _)| n == field_name)
                            {
                                if let Some(pat) = maybe_pat {
                                    self.bind_pattern(pat, field_ty, env);
                                } else {
                                    env.insert(field_name.clone(), Scheme::mono(field_ty.clone()));
                                }
                            }
                        }
                    }
                }
            }
            Pat::As(name, inner, _) => {
                env.insert(name.clone(), Scheme::mono(ty.clone()));
                self.bind_pattern(inner, ty, env);
            }
        }
    }

    fn convert_syntax_type_scheme(&mut self, ty: &Type) -> Scheme {
        let mut scope = HashMap::new();
        let ty = self.convert_syntax_type_with_scope(ty, &mut scope);
        Scheme::poly(scope_vars(&scope), ty)
    }

    fn convert_syntax_type_with_scope(
        &mut self,
        ty: &Type,
        scope: &mut HashMap<String, TyVarId>,
    ) -> Ty {
        let ty = match ty {
            Type::Con(name, _) => {
                if let Some(expanded) = self.type_aliases.get(name).cloned() {
                    return expanded;
                }
                Ty::Con(name.clone())
            }
            Type::Var(name, _) => Ty::Var(
                *scope
                    .entry(name.clone())
                    .or_insert_with(|| fresh_var_id(&mut self.engine)),
            ),
            Type::Nat(n, _) => Ty::Nat(*n),
            Type::Arrow(a, b, _) => {
                let a = self.convert_syntax_type_with_scope(a, scope);
                let b = self.convert_syntax_type_with_scope(b, scope);
                Ty::arrow(a, b)
            }
            Type::App(f, a, _) => {
                let f = self.convert_syntax_type_with_scope(f, scope);
                let a = self.convert_syntax_type_with_scope(a, scope);
                Ty::app(f, a)
            }
            Type::Paren(inner, _) => self.convert_syntax_type_with_scope(inner, scope),
            Type::Tuple(elems, _) => {
                if elems.is_empty() {
                    Ty::unit()
                } else {
                    Ty::Tuple(
                        elems
                            .iter()
                            .map(|e| self.convert_syntax_type_with_scope(e, scope))
                            .collect(),
                    )
                }
            }
            Type::Unit(_) => Ty::unit(),
        };
        normalize_type_aliases(&ty)
    }

    fn new_type_var_scope(&mut self, names: &[String]) -> HashMap<String, TyVarId> {
        names
            .iter()
            .map(|name| (name.clone(), fresh_var_id(&mut self.engine)))
            .collect()
    }

    /// Pure version that doesn't need &mut self (no fresh vars for type vars).
    fn convert_syntax_type_pure(&self, ty: &Type) -> Ty {
        let ty = match ty {
            Type::Con(name, _) => {
                if let Some(expanded) = self.type_aliases.get(name) {
                    return expanded.clone();
                }
                Ty::Con(name.clone())
            }
            Type::Var(name, _) => Ty::Con(name.clone()),
            Type::Nat(n, _) => Ty::Nat(*n),
            Type::Arrow(a, b, _) => {
                let a = self.convert_syntax_type_pure(a);
                let b = self.convert_syntax_type_pure(b);
                Ty::arrow(a, b)
            }
            Type::App(f, a, _) => {
                let f = self.convert_syntax_type_pure(f);
                let a = self.convert_syntax_type_pure(a);
                Ty::app(f, a)
            }
            Type::Paren(inner, _) => self.convert_syntax_type_pure(inner),
            Type::Tuple(elems, _) => {
                if elems.is_empty() {
                    Ty::unit()
                } else {
                    Ty::Tuple(elems.iter().map(|e| self.convert_syntax_type_pure(e)).collect())
                }
            }
            Type::Unit(_) => Ty::unit(),
        };
        normalize_type_aliases(&ty)
    }

    /// Extract the address space hint from a resource type syntax node.
    fn extract_address_space(&self, ty: &Type) -> String {
        match ty {
            Type::Con(name, _) => name.clone(),
            Type::App(f, _, _) => self.extract_address_space(f),
            Type::Paren(inner, _) => self.extract_address_space(inner),
            _ => "Uniform".to_string(),
        }
    }

    pub fn has_errors(&self) -> bool {
        self.engine.diagnostics.has_errors()
    }

    pub fn diagnostics(&self) -> &fwgsl_diagnostics::DiagnosticSink {
        &self.engine.diagnostics
    }

    fn finalize_expr(&self, expr: HirExpr) -> HirExpr {
        match expr {
            HirExpr::Lit(lit, ty, span) => HirExpr::Lit(lit, self.engine.finalize(&ty), span),
            HirExpr::Var(name, ty, span) => HirExpr::Var(name, self.engine.finalize(&ty), span),
            HirExpr::App(func, arg, ty, span) => HirExpr::App(
                Box::new(self.finalize_expr(*func)),
                Box::new(self.finalize_expr(*arg)),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::Let(binds, body, ty, span) => HirExpr::Let(
                binds
                    .into_iter()
                    .map(|(name, expr)| (name, self.finalize_expr(expr)))
                    .collect(),
                Box::new(self.finalize_expr(*body)),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::Case(scrutinee, arms, ty, span) => HirExpr::Case(
                Box::new(self.finalize_expr(*scrutinee)),
                arms.into_iter()
                    .map(|arm| HirCaseArm {
                        pattern: self.finalize_pattern(arm.pattern),
                        body: self.finalize_expr(arm.body),
                    })
                    .collect(),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::If(cond, then_expr, else_expr, ty, span) => HirExpr::If(
                Box::new(self.finalize_expr(*cond)),
                Box::new(self.finalize_expr(*then_expr)),
                Box::new(self.finalize_expr(*else_expr)),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::BinOp(op, lhs, rhs, ty, span) => HirExpr::BinOp(
                op,
                Box::new(self.finalize_expr(*lhs)),
                Box::new(self.finalize_expr(*rhs)),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::ConstructorCall(name, tag, args, ty, span) => HirExpr::ConstructorCall(
                name,
                tag,
                args.into_iter()
                    .map(|arg| self.finalize_expr(arg))
                    .collect(),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::FieldAccess(expr, field, ty, span) => HirExpr::FieldAccess(
                Box::new(self.finalize_expr(*expr)),
                field,
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::Index(base, index, ty, span) => HirExpr::Index(
                Box::new(self.finalize_expr(*base)),
                Box::new(self.finalize_expr(*index)),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::UnaryNeg(inner, ty, span) => HirExpr::UnaryNeg(
                Box::new(self.finalize_expr(*inner)),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::Loop(loop_name, bindings, body, ty, span) => HirExpr::Loop(
                loop_name,
                bindings
                    .into_iter()
                    .map(|(n, e)| (n, self.finalize_expr(e)))
                    .collect(),
                Box::new(self.finalize_expr(*body)),
                self.engine.finalize(&ty),
                span,
            ),
        }
    }

    fn finalize_pattern(&self, pattern: HirPattern) -> HirPattern {
        match pattern {
            HirPattern::Wild => HirPattern::Wild,
            HirPattern::Var(name, ty) => HirPattern::Var(name, self.engine.finalize(&ty)),
            HirPattern::Constructor(name, tag, sub_patterns) => HirPattern::Constructor(
                name,
                tag,
                sub_patterns
                    .into_iter()
                    .map(|pattern| self.finalize_pattern(pattern))
                    .collect(),
            ),
            HirPattern::Lit(lit) => HirPattern::Lit(lit),
        }
    }
}

fn desugar_where(body: &Expr, where_binds: &[(String, Expr)], span: Span) -> Expr {
    if where_binds.is_empty() {
        body.clone()
    } else {
        Expr::Let(where_binds.to_vec(), Box::new(body.clone()), span)
    }
}

fn fresh_var_id(engine: &mut InferEngine) -> TyVarId {
    match engine.fresh_var() {
        Ty::Var(id) => id,
        _ => unreachable!(),
    }
}

fn scope_vars(scope: &HashMap<String, TyVarId>) -> Vec<TyVarId> {
    let mut vars: Vec<_> = scope.values().copied().collect();
    vars.sort_unstable();
    vars.dedup();
    vars
}

fn apply_type_params(name: &str, type_params: &[String], scope: &HashMap<String, TyVarId>) -> Ty {
    type_params
        .iter()
        .fold(Ty::Con(name.to_string()), |ty, param| {
            let var = scope
                .get(param)
                .copied()
                .expect("type parameter should exist in scope");
            Ty::app(ty, Ty::Var(var))
        })
}

/// Extract a name from a pattern (for parameter names).
fn pat_name(pat: &Pat) -> String {
    match pat {
        Pat::Var(name, _) => name.clone(),
        Pat::Wild(_) => "_".to_string(),
        Pat::Paren(inner, _) => pat_name(inner),
        Pat::As(name, _, _) => name.clone(),
        _ => "_".to_string(),
    }
}

/// Flatten tuple arrows: `(A, B) -> R` becomes `A -> B -> R`.
fn flatten_tuple_arrows(ty: &Ty) -> Ty {
    match ty {
        Ty::Arrow(from, to) => {
            let to_flat = flatten_tuple_arrows(to);
            if let Ty::Tuple(elems) = from.as_ref() {
                let mut result = to_flat;
                for elem in elems.iter().rev() {
                    result = Ty::arrow(flatten_tuple_arrows(elem), result);
                }
                result
            } else {
                Ty::arrow(flatten_tuple_arrows(from), to_flat)
            }
        }
        _ => ty.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fwgsl_semantic::SemanticAnalyzer;

    fn span() -> Span {
        Span::new(0, 0)
    }

    #[test]
    fn test_lower_add_function() {
        let program = Program {
            decls: vec![
                Decl::TypeSig {
                    name: "add".into(),
                    ty: Type::Arrow(
                        Box::new(Type::Con("I32".into(), span())),
                        Box::new(Type::Arrow(
                            Box::new(Type::Con("I32".into(), span())),
                            Box::new(Type::Con("I32".into(), span())),
                            span(),
                        )),
                        span(),
                    ),
                    span: span(),
                },
                Decl::FunDecl {
                    name: "add".into(),
                    params: vec![Pat::Var("x".into(), span()), Pat::Var("y".into(), span())],
                    body: Expr::Infix(
                        Box::new(Expr::Var("x".into(), span())),
                        "+".into(),
                        Box::new(Expr::Var("y".into(), span())),
                        span(),
                    ),
                    where_binds: vec![],
                    span: span(),
                },
            ],
        };

        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        assert!(!sa.has_errors());

        let mut lowering = AstLowering::new(&sa);
        let hir = lowering.lower_program(&program);

        assert_eq!(hir.functions.len(), 1);
        let f = &hir.functions[0];
        assert_eq!(f.name, "add");
        assert_eq!(f.params.len(), 2);
        assert!(matches!(f.body, HirExpr::BinOp(BinOp::Add, _, _, _, _)));
    }

    #[test]
    fn test_lower_empty_program() {
        let program = Program { decls: vec![] };
        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        let mut lowering = AstLowering::new(&sa);
        let hir = lowering.lower_program(&program);
        assert!(hir.functions.is_empty());
        assert!(hir.data_types.is_empty());
        assert!(hir.entry_points.is_empty());
    }

    #[test]
    fn test_lower_where_clause() {
        let program = Program {
            decls: vec![Decl::FunDecl {
                name: "f".into(),
                params: vec![Pat::Var("x".into(), span())],
                body: Expr::Infix(
                    Box::new(Expr::Var("y".into(), span())),
                    "+".into(),
                    Box::new(Expr::Lit(Lit::Int(1), span())),
                    span(),
                ),
                where_binds: vec![("y".into(), Expr::Var("x".into(), span()))],
                span: span(),
            }],
        };

        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        assert!(!sa.has_errors());

        let mut lowering = AstLowering::new(&sa);
        let hir = lowering.lower_program(&program);

        assert_eq!(hir.functions.len(), 1);
        let f = &hir.functions[0];
        match &f.body {
            HirExpr::Let(binds, body, _, _) => {
                assert_eq!(binds.len(), 1);
                assert_eq!(binds[0].0, "y");
                assert!(matches!(
                    body.as_ref(),
                    HirExpr::BinOp(BinOp::Add, _, _, _, _)
                ));
            }
            other => panic!("expected HirExpr::Let, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_data_type() {
        let program = Program {
            decls: vec![Decl::DataDecl {
                name: "Color".into(),
                type_params: vec![],
                constructors: vec![
                    ConDecl {
                        name: "Red".into(),
                        fields: ConFields::Empty,
                        span: span(),
                    },
                    ConDecl {
                        name: "Green".into(),
                        fields: ConFields::Empty,
                        span: span(),
                    },
                    ConDecl {
                        name: "Blue".into(),
                        fields: ConFields::Empty,
                        span: span(),
                    },
                ],
                span: span(),
            }],
        };

        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        let mut lowering = AstLowering::new(&sa);
        let hir = lowering.lower_program(&program);

        assert_eq!(hir.data_types.len(), 1);
        let dt = &hir.data_types[0];
        assert_eq!(dt.name, "Color");
        assert_eq!(dt.constructors.len(), 3);
        assert_eq!(dt.constructors[0].name, "Red");
        assert_eq!(dt.constructors[0].tag, 0);
        assert_eq!(dt.constructors[1].name, "Green");
        assert_eq!(dt.constructors[1].tag, 1);
        assert_eq!(dt.constructors[2].name, "Blue");
        assert_eq!(dt.constructors[2].tag, 2);
    }
}
