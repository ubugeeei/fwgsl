//! Semantic analysis for fwgsl.
//!
//! Performs name resolution and type inference over the parser's AST.
//! Collects data type definitions, constructor info, type signatures,
//! and infers types for function bodies using Algorithm W (HM inference).

use std::collections::HashMap;

use fwgsl_diagnostics::{Diagnostic, DiagnosticSink, Label};
use fwgsl_parser::parser::*;
use fwgsl_span::Span;
use fwgsl_typechecker::*;

/// The semantic analyzer: collects definitions and performs type inference.
pub struct SemanticAnalyzer {
    pub env: TypeEnv,
    pub engine: InferEngine,
    pub constructors: HashMap<String, ConstructorInfo>,
    pub data_types: HashMap<String, DataTypeInfo>,
}

/// Information about a data type collected during semantic analysis.
#[derive(Debug, Clone)]
pub struct DataTypeInfo {
    pub name: String,
    pub type_params: Vec<String>,
    pub constructors: Vec<String>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        let mut sa = Self {
            env: TypeEnv::new(),
            engine: InferEngine::new(),
            constructors: HashMap::new(),
            data_types: HashMap::new(),
        };
        sa.add_builtins();
        sa
    }

    fn add_builtins(&mut self) {
        // Arithmetic: I32 -> I32 -> I32
        let i32_binop = Scheme::mono(Ty::arrow(Ty::i32(), Ty::arrow(Ty::i32(), Ty::i32())));
        let _f32_binop = Scheme::mono(Ty::arrow(Ty::f32(), Ty::arrow(Ty::f32(), Ty::f32())));

        for op in ["+", "-", "*", "/", "%"] {
            // For now, make these work on I32. We'll add overloading later via type classes.
            self.env.insert(op.to_string(), i32_binop.clone());
        }

        // Comparison: I32 -> I32 -> Bool
        let i32_cmp = Scheme::mono(Ty::arrow(Ty::i32(), Ty::arrow(Ty::i32(), Ty::bool())));
        for op in ["==", "/=", "<", ">", "<=", ">="] {
            self.env.insert(op.to_string(), i32_cmp.clone());
        }

        // Boolean ops
        let bool_binop = Scheme::mono(Ty::arrow(Ty::bool(), Ty::arrow(Ty::bool(), Ty::bool())));
        self.env.insert("&&".to_string(), bool_binop.clone());
        self.env.insert("||".to_string(), bool_binop);
    }

    /// Analyze a full program.
    pub fn analyze(&mut self, program: &Program) {
        // Pass 1: collect data types and constructors
        for decl in &program.decls {
            if let Decl::DataDecl {
                name,
                type_params,
                constructors,
                span,
            } = decl
            {
                self.register_data_type(name, type_params, constructors, *span);
            }
        }

        // Pass 1b: collect type aliases (treated as synonyms for semantic purposes)
        for decl in &program.decls {
            if let Decl::TypeAlias { name, ty, .. } = decl {
                let alias_ty = self.convert_syntax_type(ty);
                // Register the alias name as a type constructor
                self.env.insert(name.clone(), Scheme::mono(alias_ty));
            }
        }

        // Pass 2: collect type signatures
        for decl in &program.decls {
            if let Decl::TypeSig { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type(ty);
                self.env.insert(name.clone(), Scheme::mono(inferred_ty));
            }
        }

        // Pass 3: type check function bodies
        for decl in &program.decls {
            match decl {
                Decl::FunDecl {
                    name,
                    params,
                    body,
                    span,
                    ..
                } => {
                    self.check_function(name, params, body, *span);
                }
                Decl::EntryPoint {
                    name,
                    params,
                    body,
                    span,
                    ..
                } => {
                    self.check_function(name, params, body, *span);
                }
                _ => {}
            }
        }
    }

    fn register_data_type(
        &mut self,
        name: &str,
        type_params: &[String],
        cons: &[ConDecl],
        _span: Span,
    ) {
        let result_ty = Ty::Con(name.to_string());
        let mut con_names = Vec::new();

        for (tag, con) in cons.iter().enumerate() {
            let con_ty = match &con.fields {
                ConFields::Empty => result_ty.clone(),
                ConFields::Positional(fields) => {
                    let mut ty = result_ty.clone();
                    for field in fields.iter().rev() {
                        let field_ty = self.convert_syntax_type(field);
                        ty = Ty::arrow(field_ty, ty);
                    }
                    ty
                }
                ConFields::Record(fields) => {
                    // Record constructor: takes all fields positionally
                    let mut ty = result_ty.clone();
                    for (_, field_ty) in fields.iter().rev() {
                        let ft = self.convert_syntax_type(field_ty);
                        ty = Ty::arrow(ft, ty);
                    }
                    ty
                }
            };

            let field_info = match &con.fields {
                ConFields::Empty => ConstructorFields::Empty,
                ConFields::Positional(fields) => ConstructorFields::Positional(
                    fields.iter().map(|f| self.convert_syntax_type(f)).collect(),
                ),
                ConFields::Record(fields) => ConstructorFields::Record(
                    fields
                        .iter()
                        .map(|(n, t)| (n.clone(), self.convert_syntax_type(t)))
                        .collect(),
                ),
            };

            self.constructors.insert(
                con.name.clone(),
                ConstructorInfo {
                    type_name: name.to_string(),
                    tag: tag as u32,
                    fields: field_info,
                    result_ty: result_ty.clone(),
                },
            );

            self.env.insert(con.name.clone(), Scheme::mono(con_ty));
            con_names.push(con.name.clone());
        }

        self.data_types.insert(
            name.to_string(),
            DataTypeInfo {
                name: name.to_string(),
                type_params: type_params.to_vec(),
                constructors: con_names,
            },
        );
    }

    /// Convert syntax-level Type to internal Ty.
    fn convert_syntax_type(&mut self, ty: &Type) -> Ty {
        match ty {
            Type::Con(name, _) => Ty::Con(name.clone()),
            Type::Var(_, _) => {
                // For now treat type vars as fresh inference vars.
                // A proper implementation would track scoped type variables.
                self.engine.fresh_var()
            }
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
                // Tuples as type application of a tuple constructor
                if elems.is_empty() {
                    Ty::unit()
                } else {
                    // Just use first element for now
                    self.convert_syntax_type(&elems[0])
                }
            }
            Type::Unit(_) => Ty::unit(),
        }
    }

    fn check_function(&mut self, name: &str, params: &[Pat], body: &Expr, span: Span) {
        let mut local_env = self.env.clone();

        // Create types for parameters
        let mut param_types = Vec::new();
        for pat in params {
            let ty = self.engine.fresh_var();
            self.bind_pattern(pat, &ty, &mut local_env);
            param_types.push(ty);
        }

        // Infer body type
        let body_ty = self.infer_expr(body, &mut local_env);

        // Build function type: p1 -> p2 -> ... -> body_ty
        let mut fun_ty = body_ty;
        for param_ty in param_types.into_iter().rev() {
            fun_ty = Ty::arrow(param_ty, fun_ty);
        }

        // If there's a declared type, unify with it
        if let Some(scheme) = self.env.lookup(name) {
            let declared_ty = self.engine.instantiate(scheme);
            self.engine.unify(&fun_ty, &declared_ty, span);
        } else {
            // Add inferred type
            let scheme = self.engine.generalize(&self.env, &fun_ty);
            self.env.insert(name.to_string(), scheme);
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
                    // Unify result type
                    self.engine.unify(ty, &con_info.result_ty, *span);
                    // Bind sub-patterns
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
                } else {
                    self.engine.diagnostics.push(
                        Diagnostic::error(format!("Unknown constructor: {}", name))
                            .with_label(Label::primary(*span, "not found")),
                    );
                }
            }
            Pat::Lit(lit, span) => {
                let lit_ty = self.lit_type(lit);
                self.engine.unify(ty, &lit_ty, *span);
            }
            Pat::Paren(inner, _) => self.bind_pattern(inner, ty, env),
            Pat::Tuple(pats, _span) => {
                // For simplicity, treat as first element
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
                                    // Punned field: bind field name as variable
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

    fn infer_expr(&mut self, expr: &Expr, env: &mut TypeEnv) -> Ty {
        match expr {
            Expr::Lit(lit, _) => self.lit_type(lit),

            Expr::Var(name, span) => {
                if let Some(scheme) = env.lookup(name) {
                    self.engine.instantiate(scheme)
                } else {
                    self.engine.diagnostics.push(
                        Diagnostic::error(format!("Unbound variable: {}", name))
                            .with_label(Label::primary(*span, "not in scope")),
                    );
                    Ty::Error
                }
            }

            Expr::Con(name, span) => {
                if let Some(scheme) = env.lookup(name) {
                    self.engine.instantiate(scheme)
                } else {
                    self.engine.diagnostics.push(
                        Diagnostic::error(format!("Unknown constructor: {}", name))
                            .with_label(Label::primary(*span, "not in scope")),
                    );
                    Ty::Error
                }
            }

            Expr::App(func, arg, span) => {
                let func_ty = self.infer_expr(func, env);
                let arg_ty = self.infer_expr(arg, env);
                let ret_ty = self.engine.fresh_var();
                let expected = Ty::arrow(arg_ty, ret_ty.clone());
                self.engine.unify(&func_ty, &expected, *span);
                ret_ty
            }

            Expr::Infix(lhs, op, rhs, span) => {
                let op_ty = if let Some(scheme) = env.lookup(op) {
                    self.engine.instantiate(scheme)
                } else {
                    self.engine.diagnostics.push(
                        Diagnostic::error(format!("Unknown operator: {}", op))
                            .with_label(Label::primary(*span, "not in scope")),
                    );
                    return Ty::Error;
                };
                let lhs_ty = self.infer_expr(lhs, env);
                let rhs_ty = self.infer_expr(rhs, env);
                let ret_ty = self.engine.fresh_var();
                self.engine.unify(
                    &op_ty,
                    &Ty::arrow(lhs_ty, Ty::arrow(rhs_ty, ret_ty.clone())),
                    *span,
                );
                ret_ty
            }

            Expr::Lambda(pats, body, _span) => {
                let mut local_env = env.clone();
                let mut param_types = Vec::new();
                for pat in pats {
                    let ty = self.engine.fresh_var();
                    self.bind_pattern(pat, &ty, &mut local_env);
                    param_types.push(ty);
                }
                let body_ty = self.infer_expr(body, &mut local_env);
                let mut result = body_ty;
                for pt in param_types.into_iter().rev() {
                    result = Ty::arrow(pt, result);
                }
                result
            }

            Expr::Let(binds, body, _span) => {
                let mut local_env = env.clone();
                for (name, expr) in binds {
                    let ty = self.infer_expr(expr, &mut local_env);
                    let scheme = self.engine.generalize(&local_env, &ty);
                    local_env.insert(name.clone(), scheme);
                }
                self.infer_expr(body, &mut local_env)
            }

            Expr::Case(scrutinee, arms, span) => {
                let scrut_ty = self.infer_expr(scrutinee, env);
                let result_ty = self.engine.fresh_var();
                for (pat, body) in arms {
                    let mut arm_env = env.clone();
                    self.bind_pattern(pat, &scrut_ty, &mut arm_env);
                    let body_ty = self.infer_expr(body, &mut arm_env);
                    self.engine.unify(&result_ty, &body_ty, *span);
                }
                result_ty
            }

            Expr::If(cond, then_expr, else_expr, span) => {
                let cond_ty = self.infer_expr(cond, env);
                self.engine.unify(&cond_ty, &Ty::bool(), *span);
                let then_ty = self.infer_expr(then_expr, env);
                let else_ty = self.infer_expr(else_expr, env);
                self.engine.unify(&then_ty, &else_ty, *span);
                then_ty
            }

            Expr::Paren(inner, _) => self.infer_expr(inner, env),

            Expr::Tuple(elems, _span) => {
                // Simplified: just infer all elements
                let mut tys = Vec::new();
                for e in elems {
                    tys.push(self.infer_expr(e, env));
                }
                if tys.is_empty() {
                    Ty::unit()
                } else {
                    tys.remove(0)
                }
            }

            Expr::Record(fields, _span) => {
                for (_, expr) in fields {
                    self.infer_expr(expr, env);
                }
                self.engine.fresh_var()
            }

            Expr::FieldAccess(expr, field, _span) => {
                let base_ty = self.infer_expr(expr, env);
                let base_ty = self.engine.finalize(&base_ty);

                // Check for Vec swizzle patterns
                if is_swizzle(field) {
                    if let Some((n, scalar)) = extract_vec_type(&base_ty) {
                        let swizzle_len = field.len();
                        // Validate: each component index must be < n
                        if validate_swizzle(field, n) {
                            if swizzle_len == 1 {
                                return scalar;
                            } else {
                                return Ty::app(
                                    Ty::app(Ty::Con("Vec".into()), Ty::Nat(swizzle_len as u64)),
                                    scalar,
                                );
                            }
                        }
                    }
                }

                // Fallback: fresh var (record field access)
                self.engine.fresh_var()
            }

            Expr::VecLit(elems, span) => {
                if elems.is_empty() {
                    self.engine.diagnostics.push(
                        Diagnostic::error("Empty vec literal")
                            .with_label(Label::primary(*span, "needs at least 2 elements")),
                    );
                    return Ty::Error;
                }

                // Infer types of all elements and unify their scalar types
                let scalar_ty = self.engine.fresh_var();
                let mut total_components: u64 = 0;

                for elem in elems {
                    let elem_ty = self.infer_expr(elem, env);
                    let elem_ty = self.engine.finalize(&elem_ty);

                    if let Some((n, inner_scalar)) = extract_vec_type(&elem_ty) {
                        // Vec element: contributes n components
                        total_components += n as u64;
                        self.engine.unify(&scalar_ty, &inner_scalar, *span);
                    } else {
                        // Scalar element: contributes 1 component
                        total_components += 1;
                        self.engine.unify(&scalar_ty, &elem_ty, *span);
                    }
                }

                if total_components < 2 || total_components > 4 {
                    self.engine.diagnostics.push(
                        Diagnostic::error(format!(
                            "Vec literal has {} components, expected 2, 3, or 4",
                            total_components
                        ))
                        .with_label(Label::primary(*span, "invalid component count")),
                    );
                    return Ty::Error;
                }

                Ty::app(
                    Ty::app(Ty::Con("Vec".into()), Ty::Nat(total_components)),
                    scalar_ty,
                )
            }

            Expr::OpSection(op, span) => {
                if let Some(scheme) = env.lookup(op) {
                    self.engine.instantiate(scheme)
                } else {
                    self.engine.diagnostics.push(
                        Diagnostic::error(format!("Unknown operator: {}", op))
                            .with_label(Label::primary(*span, "not in scope")),
                    );
                    Ty::Error
                }
            }

            Expr::Neg(inner, _span) => {
                // Negation works on numeric types
                self.infer_expr(inner, env)
            }

            Expr::Do(stmts, _span) => {
                let mut local_env = env.clone();
                let mut last_ty = Ty::unit();
                for stmt in stmts {
                    match stmt {
                        DoStmt::Expr(expr, _) => {
                            last_ty = self.infer_expr(expr, &mut local_env);
                        }
                        DoStmt::Bind(name, expr, _) => {
                            let ty = self.infer_expr(expr, &mut local_env);
                            // bind extracts the inner type from m a
                            let inner_ty = self.engine.fresh_var();
                            local_env.insert(name.clone(), Scheme::mono(inner_ty));
                            last_ty = ty;
                        }
                        DoStmt::Let(name, expr, _) => {
                            let ty = self.infer_expr(expr, &mut local_env);
                            local_env.insert(name.clone(), Scheme::mono(ty));
                        }
                    }
                }
                last_ty
            }
        }
    }

    fn lit_type(&self, lit: &Lit) -> Ty {
        match lit {
            Lit::Int(_) => Ty::i32(),
            Lit::Float(_) => Ty::f32(),
            Lit::String(_) => Ty::Con("String".into()),
            Lit::Char(_) => Ty::Con("Char".into()),
        }
    }

    pub fn has_errors(&self) -> bool {
        self.engine.diagnostics.has_errors()
    }

    pub fn diagnostics(&self) -> &DiagnosticSink {
        &self.engine.diagnostics
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Swizzle / Vec helpers ─────────────────────────────────────────────

/// Check if a field name is a valid swizzle pattern (xyzw or rgba, 1-4 chars).
pub fn is_swizzle(field: &str) -> bool {
    if field.is_empty() || field.len() > 4 {
        return false;
    }
    let all_xyzw = field.chars().all(|c| matches!(c, 'x' | 'y' | 'z' | 'w'));
    let all_rgba = field.chars().all(|c| matches!(c, 'r' | 'g' | 'b' | 'a'));
    all_xyzw || all_rgba
}

/// Get the component index for a swizzle character.
fn swizzle_index(c: char) -> usize {
    match c {
        'x' | 'r' => 0,
        'y' | 'g' => 1,
        'z' | 'b' => 2,
        'w' | 'a' => 3,
        _ => 0,
    }
}

/// Validate that all swizzle components are within bounds for a Vec of size n.
pub fn validate_swizzle(field: &str, n: u8) -> bool {
    field.chars().all(|c| swizzle_index(c) < n as usize)
}

/// Extract Vec type info: Vec n T → Some((n, T))
pub fn extract_vec_type(ty: &Ty) -> Option<(u8, Ty)> {
    // Vec n T = App(App(Con("Vec"), Nat(n)), T)
    if let Ty::App(f, scalar) = ty {
        if let Ty::App(con, nat) = f.as_ref() {
            if let (Ty::Con(name), Ty::Nat(n)) = (con.as_ref(), nat.as_ref()) {
                if name == "Vec" {
                    return Some((*n as u8, scalar.as_ref().clone()));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use fwgsl_span::Span;

    fn span() -> Span {
        Span::new(0, 0)
    }

    #[test]
    fn test_empty_program() {
        let mut sa = SemanticAnalyzer::new();
        let program = Program { decls: vec![] };
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_simple_function_inference() {
        let mut sa = SemanticAnalyzer::new();
        // f x = x + 1
        let program = Program {
            decls: vec![Decl::FunDecl {
                name: "f".into(),
                params: vec![Pat::Var("x".into(), span())],
                body: Expr::Infix(
                    Box::new(Expr::Var("x".into(), span())),
                    "+".into(),
                    Box::new(Expr::Lit(Lit::Int(1), span())),
                    span(),
                ),
                where_binds: vec![],
                span: span(),
            }],
        };
        sa.analyze(&program);
        assert!(!sa.has_errors());
        // f should have type I32 -> I32
        let scheme = sa.env.lookup("f").expect("f should be in env");
        let ty = sa.engine.finalize(&scheme.ty);
        assert_eq!(format!("{}", ty), "(I32 -> I32)");
    }

    #[test]
    fn test_data_type_registration() {
        let mut sa = SemanticAnalyzer::new();
        // data Color = Red | Green | Blue
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
        sa.analyze(&program);
        assert!(!sa.has_errors());
        assert!(sa.constructors.contains_key("Red"));
        assert!(sa.constructors.contains_key("Green"));
        assert!(sa.constructors.contains_key("Blue"));
        assert_eq!(sa.constructors["Red"].tag, 0);
        assert_eq!(sa.constructors["Green"].tag, 1);
        assert_eq!(sa.constructors["Blue"].tag, 2);
    }

    #[test]
    fn test_unbound_variable_error() {
        let mut sa = SemanticAnalyzer::new();
        // f x = y  (y is unbound)
        let program = Program {
            decls: vec![Decl::FunDecl {
                name: "f".into(),
                params: vec![Pat::Var("x".into(), span())],
                body: Expr::Var("y".into(), span()),
                where_binds: vec![],
                span: span(),
            }],
        };
        sa.analyze(&program);
        assert!(sa.has_errors());
    }

    #[test]
    fn test_type_signature_check() {
        let mut sa = SemanticAnalyzer::new();
        // add :: I32 -> I32 -> I32
        // add x y = x + y
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
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_if_expr_type_check() {
        let mut sa = SemanticAnalyzer::new();
        // f x = if x == 0 then 1 else 2
        let program = Program {
            decls: vec![Decl::FunDecl {
                name: "f".into(),
                params: vec![Pat::Var("x".into(), span())],
                body: Expr::If(
                    Box::new(Expr::Infix(
                        Box::new(Expr::Var("x".into(), span())),
                        "==".into(),
                        Box::new(Expr::Lit(Lit::Int(0), span())),
                        span(),
                    )),
                    Box::new(Expr::Lit(Lit::Int(1), span())),
                    Box::new(Expr::Lit(Lit::Int(2), span())),
                    span(),
                ),
                where_binds: vec![],
                span: span(),
            }],
        };
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_let_expr() {
        let mut sa = SemanticAnalyzer::new();
        // f = let x = 42 in x + 1
        let program = Program {
            decls: vec![Decl::FunDecl {
                name: "f".into(),
                params: vec![],
                body: Expr::Let(
                    vec![("x".into(), Expr::Lit(Lit::Int(42), span()))],
                    Box::new(Expr::Infix(
                        Box::new(Expr::Var("x".into(), span())),
                        "+".into(),
                        Box::new(Expr::Lit(Lit::Int(1), span())),
                        span(),
                    )),
                    span(),
                ),
                where_binds: vec![],
                span: span(),
            }],
        };
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_lambda_inference() {
        let mut sa = SemanticAnalyzer::new();
        // f = \x -> x + 1
        let program = Program {
            decls: vec![Decl::FunDecl {
                name: "f".into(),
                params: vec![],
                body: Expr::Lambda(
                    vec![Pat::Var("x".into(), span())],
                    Box::new(Expr::Infix(
                        Box::new(Expr::Var("x".into(), span())),
                        "+".into(),
                        Box::new(Expr::Lit(Lit::Int(1), span())),
                        span(),
                    )),
                    span(),
                ),
                where_binds: vec![],
                span: span(),
            }],
        };
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_case_expr_with_data_type() {
        let mut sa = SemanticAnalyzer::new();
        // data Bool2 = True2 | False2
        // f x = match x | True2 -> 1 | False2 -> 0
        let program = Program {
            decls: vec![
                Decl::DataDecl {
                    name: "Bool2".into(),
                    type_params: vec![],
                    constructors: vec![
                        ConDecl {
                            name: "True2".into(),
                            fields: ConFields::Empty,
                            span: span(),
                        },
                        ConDecl {
                            name: "False2".into(),
                            fields: ConFields::Empty,
                            span: span(),
                        },
                    ],
                    span: span(),
                },
                Decl::FunDecl {
                    name: "f".into(),
                    params: vec![Pat::Var("x".into(), span())],
                    body: Expr::Case(
                        Box::new(Expr::Var("x".into(), span())),
                        vec![
                            (
                                Pat::Con("True2".into(), vec![], span()),
                                Expr::Lit(Lit::Int(1), span()),
                            ),
                            (
                                Pat::Con("False2".into(), vec![], span()),
                                Expr::Lit(Lit::Int(0), span()),
                            ),
                        ],
                        span(),
                    ),
                    where_binds: vec![],
                    span: span(),
                },
            ],
        };
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }
}
