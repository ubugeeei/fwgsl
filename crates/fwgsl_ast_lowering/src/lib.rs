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
}

impl AstLowering {
    /// Create a new lowering context from a completed semantic analyzer.
    pub fn new(sa: &fwgsl_semantic::SemanticAnalyzer) -> Self {
        let mut lowering = Self {
            env: sa.env.clone(),
            engine: InferEngine::new(),
            constructors: sa.constructors.clone(),
            data_types: sa.data_types.clone(),
        };
        lowering.add_builtins();
        lowering
    }

    fn add_builtins(&mut self) {
        let i32_binop = Scheme::mono(Ty::arrow(Ty::i32(), Ty::arrow(Ty::i32(), Ty::i32())));
        for op in ["+", "-", "*", "/", "%"] {
            self.env.insert(op.to_string(), i32_binop.clone());
        }
        let i32_cmp = Scheme::mono(Ty::arrow(Ty::i32(), Ty::arrow(Ty::i32(), Ty::bool())));
        for op in ["==", "/=", "<", ">", "<=", ">="] {
            self.env.insert(op.to_string(), i32_cmp.clone());
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
                type_params: _,
                constructors: cons,
                span: _,
            } = decl
            {
                let result_ty = Ty::Con(name.to_string());
                for con in cons {
                    let con_ty = match &con.fields {
                        ConFields::Empty => result_ty.clone(),
                        ConFields::Positional(fields) => {
                            let mut ty = result_ty.clone();
                            for field in fields.iter().rev() {
                                let ft = self.convert_syntax_type(field);
                                ty = Ty::arrow(ft, ty);
                            }
                            ty
                        }
                        ConFields::Record(fields) => {
                            let mut ty = result_ty.clone();
                            for (_, field_ty) in fields.iter().rev() {
                                let ft = self.convert_syntax_type(field_ty);
                                ty = Ty::arrow(ft, ty);
                            }
                            ty
                        }
                    };
                    self.env.insert(con.name.clone(), Scheme::mono(con_ty));
                }
            }
        }

        // Pass 2: collect type signatures
        for decl in &program.decls {
            if let Decl::TypeSig { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type(ty);
                self.env.insert(name.clone(), Scheme::mono(inferred_ty));
            }
        }

        let mut functions = Vec::new();
        let mut data_types = Vec::new();
        let mut entry_points = Vec::new();

        for decl in &program.decls {
            match decl {
                Decl::FunDecl {
                    name,
                    params,
                    body,
                    span,
                    ..
                } => {
                    if let Some(f) = self.lower_fun_decl(name, params, body, *span) {
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
                Decl::TypeSig { .. } | Decl::TypeAlias { .. } => {}
            }
        }

        HirProgram {
            functions,
            data_types,
            entry_points,
        }
    }

    fn lower_fun_decl(
        &mut self,
        name: &str,
        params: &[Pat],
        body: &Expr,
        span: Span,
    ) -> Option<HirFunction> {
        let mut local_env = self.env.clone();

        let mut hir_params = Vec::new();
        let mut param_types = Vec::new();
        for pat in params {
            let ty = self.engine.fresh_var();
            let pname = pat_name(pat);
            self.bind_pattern(pat, &ty, &mut local_env);
            param_types.push(ty.clone());
            hir_params.push((pname, ty));
        }

        let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);

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

        Some(HirFunction {
            name: name.to_string(),
            params: final_params,
            return_ty,
            body: hir_body,
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
            let ty = self.engine.fresh_var();
            let pname = pat_name(pat);
            self.bind_pattern(pat, &ty, &mut local_env);
            param_types.push(ty.clone());
            hir_params.push((pname, ty));
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
            body: hir_body,
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
                        (format!("field{}", i), ty)
                    })
                    .collect(),
                ConFields::Record(fields) => fields
                    .iter()
                    .map(|(n, t)| {
                        let ty = self.convert_syntax_type_pure(t);
                        (n.clone(), ty)
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
                let final_ty = self.engine.finalize(&ty);
                (
                    HirExpr::Var(name.clone(), final_ty.clone(), *span),
                    final_ty,
                )
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
                        ConstructorFields::Empty => {
                            let final_ty = self.engine.finalize(&ty);
                            (
                                HirExpr::ConstructorCall(
                                    name.clone(),
                                    con_info.tag,
                                    vec![],
                                    final_ty.clone(),
                                    *span,
                                ),
                                final_ty,
                            )
                        }
                        _ => {
                            let final_ty = self.engine.finalize(&ty);
                            (
                                HirExpr::Var(name.clone(), final_ty.clone(), *span),
                                final_ty,
                            )
                        }
                    }
                } else {
                    let final_ty = self.engine.finalize(&ty);
                    (
                        HirExpr::Var(name.clone(), final_ty.clone(), *span),
                        final_ty,
                    )
                }
            }

            Expr::App(func, arg, span) => {
                let (hir_func, func_ty) = self.lower_expr(func, env);
                let (hir_arg, arg_ty) = self.lower_expr(arg, env);
                let ret_ty = self.engine.fresh_var();
                let expected = Ty::arrow(arg_ty, ret_ty.clone());
                self.engine.unify(&func_ty, &expected, *span);
                let final_ty = self.engine.finalize(&ret_ty);
                (
                    HirExpr::App(
                        Box::new(hir_func),
                        Box::new(hir_arg),
                        final_ty.clone(),
                        *span,
                    ),
                    final_ty,
                )
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
                    let final_ty = self.engine.finalize(&ret_ty);
                    (
                        HirExpr::BinOp(
                            binop,
                            Box::new(hir_lhs),
                            Box::new(hir_rhs),
                            final_ty.clone(),
                            *span,
                        ),
                        final_ty,
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
                    let final_ty = self.engine.finalize(&ret_ty);
                    let op_expr = HirExpr::Var(op.clone(), op_ty, *span);
                    let app1_ty = self.engine.fresh_var();
                    let app1 = HirExpr::App(Box::new(op_expr), Box::new(hir_lhs), app1_ty, *span);
                    (
                        HirExpr::App(Box::new(app1), Box::new(hir_rhs), final_ty.clone(), *span),
                        final_ty,
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
                let final_ty = self.engine.finalize(&result_ty);
                // Lambda is desugared: for now, emit as the body directly
                // (lambdas in fwgsl source get applied away or become function params)
                (hir_body, final_ty)
            }

            Expr::Let(binds, body, span) => {
                let mut local_env = env.clone();
                let mut hir_binds = Vec::new();
                for (name, expr) in binds {
                    let (hir_expr, ty) = self.lower_expr(expr, &mut local_env);
                    let scheme = self.engine.generalize(&local_env, &ty);
                    local_env.insert(name.clone(), scheme);
                    hir_binds.push((name.clone(), hir_expr));
                }
                let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);
                let final_ty = self.engine.finalize(&body_ty);
                (
                    HirExpr::Let(hir_binds, Box::new(hir_body), final_ty.clone(), *span),
                    final_ty,
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

                let final_ty = self.engine.finalize(&result_ty);
                (
                    HirExpr::Case(Box::new(hir_scrut), hir_arms, final_ty.clone(), *span),
                    final_ty,
                )
            }

            Expr::If(cond, then_expr, else_expr, span) => {
                let (hir_cond, cond_ty) = self.lower_expr(cond, env);
                self.engine.unify(&cond_ty, &Ty::bool(), *span);
                let (hir_then, then_ty) = self.lower_expr(then_expr, env);
                let (hir_else, else_ty) = self.lower_expr(else_expr, env);
                self.engine.unify(&then_ty, &else_ty, *span);
                let final_ty = self.engine.finalize(&then_ty);
                (
                    HirExpr::If(
                        Box::new(hir_cond),
                        Box::new(hir_then),
                        Box::new(hir_else),
                        final_ty.clone(),
                        *span,
                    ),
                    final_ty,
                )
            }

            Expr::Paren(inner, _) => self.lower_expr(inner, env),

            Expr::Tuple(elems, span) => {
                if elems.is_empty() {
                    (HirExpr::Lit(HirLit::Int(0), Ty::unit(), *span), Ty::unit())
                } else {
                    // Lower first element as a simplification
                    self.lower_expr(&elems[0], env)
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

                let final_ty = self.engine.finalize(&result_ty);
                (
                    HirExpr::FieldAccess(
                        Box::new(hir_expr),
                        field.clone(),
                        final_ty.clone(),
                        *span,
                    ),
                    final_ty,
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
                let n = total_components.min(4).max(2);
                let vec_name = format!("$vec{}", n);
                let result_ty = Ty::app(
                    Ty::app(Ty::Con("Vec".into()), Ty::Nat(n)),
                    self.engine.finalize(&scalar_ty),
                );
                let final_ty = self.engine.finalize(&result_ty);

                // Build curried application chain
                let mut expr = HirExpr::Var(vec_name, final_ty.clone(), *span);
                for arg in hir_args {
                    let app_ty = final_ty.clone();
                    expr = HirExpr::App(Box::new(expr), Box::new(arg), app_ty, *span);
                }

                (expr, final_ty)
            }

            Expr::OpSection(op, span) => {
                let ty = if let Some(scheme) = env.lookup(op) {
                    self.engine.instantiate(scheme)
                } else {
                    Ty::Error
                };
                let final_ty = self.engine.finalize(&ty);
                (HirExpr::Var(op.clone(), final_ty.clone(), *span), final_ty)
            }

            Expr::Neg(inner, span) => {
                let (hir_inner, inner_ty) = self.lower_expr(inner, env);
                // Negation: emit as (0 - x)
                let zero = HirExpr::Lit(HirLit::Int(0), inner_ty.clone(), *span);
                let final_ty = self.engine.finalize(&inner_ty);
                (
                    HirExpr::BinOp(
                        BinOp::Sub,
                        Box::new(zero),
                        Box::new(hir_inner),
                        final_ty.clone(),
                        *span,
                    ),
                    final_ty,
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
                    let final_ty = self.engine.finalize(&body_ty);
                    (
                        HirExpr::Let(hir_binds, Box::new(body), final_ty.clone(), *span),
                        final_ty,
                    )
                }
            }
        }
    }

    fn lower_pattern(&self, pat: &Pat, _scrutinee_ty: &Ty) -> HirPattern {
        match pat {
            Pat::Wild(_) => HirPattern::Wild,
            Pat::Var(name, _) => HirPattern::Var(name.clone(), self.engine.finalize(_scrutinee_ty)),
            Pat::Con(name, sub_pats, _) => {
                if let Some(con_info) = self.constructors.get(name) {
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
                if let Some(con_info) = self.constructors.get(con_name) {
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
                if let Some(con_info) = self.constructors.get(name).cloned() {
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
                    Lit::Int(_) => Ty::i32(),
                    Lit::Float(_) => Ty::f32(),
                    Lit::String(_) => Ty::Con("String".into()),
                    Lit::Char(_) => Ty::Con("Char".into()),
                };
                self.engine.unify(ty, &lit_ty, *span);
            }
            Pat::Paren(inner, _) => self.bind_pattern(inner, ty, env),
            Pat::Tuple(pats, _) => {
                if let Some(first) = pats.first() {
                    self.bind_pattern(first, ty, env);
                }
            }
            Pat::Record(con_name, fields, span) => {
                if let Some(con_info) = self.constructors.get(con_name).cloned() {
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

    fn convert_syntax_type(&mut self, ty: &Type) -> Ty {
        match ty {
            Type::Con(name, _) => Ty::Con(name.clone()),
            Type::Var(_, _) => self.engine.fresh_var(),
            Type::Arrow(a, b, _) => {
                let a = self.convert_syntax_type(a);
                let b = self.convert_syntax_type(b);
                Ty::arrow(a, b)
            }
            Type::App(f, a, _) => {
                let f = self.convert_syntax_type(f);
                let a = self.convert_syntax_type(a);
                Ty::app(f, a)
            }
            Type::Paren(inner, _) => self.convert_syntax_type(inner),
            Type::Tuple(elems, _) => {
                if elems.is_empty() {
                    Ty::unit()
                } else {
                    self.convert_syntax_type(&elems[0])
                }
            }
            Type::Unit(_) => Ty::unit(),
        }
    }

    /// Pure version that doesn't need &mut self (no fresh vars for type vars).
    fn convert_syntax_type_pure(&self, ty: &Type) -> Ty {
        match ty {
            Type::Con(name, _) => Ty::Con(name.clone()),
            Type::Var(name, _) => Ty::Con(name.clone()),
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
                    self.convert_syntax_type_pure(&elems[0])
                }
            }
            Type::Unit(_) => Ty::unit(),
        }
    }

    pub fn has_errors(&self) -> bool {
        self.engine.diagnostics.has_errors()
    }
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
