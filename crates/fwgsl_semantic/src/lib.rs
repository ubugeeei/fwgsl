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

/// Information about a trait declaration.
#[derive(Debug, Clone)]
pub struct TraitInfo {
    pub name: String,
    /// The type variable the trait is parameterised over.
    pub var: String,
    /// Method signatures: method_name → type (with `var` as a free type variable).
    pub methods: Vec<(String, Ty)>,
}

/// Information about a trait impl.
#[derive(Debug, Clone)]
pub struct ImplInfo {
    /// None for standalone impls.
    pub trait_name: Option<String>,
    /// The concrete type this impl is for.
    pub ty: Ty,
    /// Method implementations: method_name → mangled function name.
    pub methods: HashMap<String, String>,
}

/// The semantic analyzer: collects definitions and performs type inference.
pub struct SemanticAnalyzer {
    pub env: TypeEnv,
    pub engine: InferEngine,
    pub constructors: HashMap<String, ConstructorInfo>,
    pub data_types: HashMap<String, DataTypeInfo>,
    /// User-defined type aliases (e.g. `alias Float2 = Vec<2, F32>`).
    /// Maps alias name → expanded Ty so they can be resolved during type conversion.
    pub type_aliases: HashMap<String, Ty>,
    /// Trait declarations: trait_name → TraitInfo.
    pub traits: HashMap<String, TraitInfo>,
    /// Trait impls.
    pub impls: Vec<ImplInfo>,
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
        Self {
            env: TypeEnv::new(),
            engine: InferEngine::new(),
            constructors: HashMap::new(),
            data_types: HashMap::new(),
            type_aliases: HashMap::new(),
            traits: HashMap::new(),
            impls: Vec::new(),
        }
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
                ..
            } = decl
            {
                self.register_data_type(name, type_params, constructors, *span);
            }
        }

        // Pass 1b: collect type aliases (treated as synonyms for semantic purposes)
        for decl in &program.decls {
            if let Decl::TypeAlias { name, ty, .. } = decl {
                let alias_ty = self.convert_syntax_type(ty);
                // Store the expanded type for alias resolution during type conversion
                self.type_aliases.insert(name.clone(), alias_ty.ty.clone());
                // Register the alias name as a type constructor
                self.env.insert(name.clone(), alias_ty);
            }
            if let Decl::BitfieldDecl { base_ty, .. } = decl {
                let _base = self.convert_syntax_type(base_ty);
            }
        }

        // Pass 2: collect type signatures
        for decl in &program.decls {
            if let Decl::TypeSig { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type(ty);
                // Flatten tuple args: (A, B) -> R becomes A -> B -> R
                let flattened = Scheme {
                    vars: inferred_ty.vars.clone(),
                    ty: flatten_tuple_arrows(&inferred_ty.ty),
                };
                self.env.insert(name.clone(), flattened);
            }
            if let Decl::ConstDecl { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type(ty);
                self.env.insert(name.clone(), inferred_ty);
            }
            if let Decl::ResourceDecl { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type(ty);
                // Unwrap the resource wrapper (Uniform<T> → T, Storage<_, T> → T)
                // so that in the body, the variable has the inner type.
                let inner_ty = unwrap_resource_type(&inferred_ty.ty);
                self.env.insert(name.clone(), Scheme::mono(inner_ty));
            }
            if let Decl::ExternDecl { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type(ty);
                let flattened = Scheme {
                    vars: inferred_ty.vars.clone(),
                    ty: flatten_tuple_arrows(&inferred_ty.ty),
                };
                self.env.insert(name.clone(), flattened);
            }
        }

        // Pass 2b: collect trait declarations
        for decl in &program.decls {
            if let Decl::TraitDecl {
                name,
                var,
                methods,
                ..
            } = decl
            {
                // Build a type variable scope seeded with the trait variable
                let var_id = fresh_var_id(&mut self.engine);
                let mut trait_methods = Vec::new();
                for m in methods {
                    let mut scope = HashMap::new();
                    scope.insert(var.clone(), var_id);
                    let method_ty = self.convert_syntax_type_with_scope(&m.ty, &mut scope);
                    let scheme = Scheme::poly(scope_vars(&scope), method_ty.clone());
                    // Register the method as a polymorphic function in the env
                    // (overrides any existing builtin operator with the same name)
                    self.env.insert(m.name.clone(), scheme);
                    trait_methods.push((m.name.clone(), method_ty));
                }
                self.traits.insert(
                    name.clone(),
                    TraitInfo {
                        name: name.clone(),
                        var: var.clone(),
                        methods: trait_methods,
                    },
                );
            }
        }

        // Pass 2c: collect impl declarations — generate mangled function names
        for decl in &program.decls {
            if let Decl::ImplDecl {
                trait_name,
                ty,
                methods,
                span,
                ..
            } = decl
            {
                let impl_ty_scheme = self.convert_syntax_type(ty);
                let type_suffix = format_type_suffix(&impl_ty_scheme.ty);
                let mut impl_methods = HashMap::new();
                for m in methods {
                    let mangled = mangle_instance_method(&m.name, &type_suffix);
                    impl_methods.insert(m.name.clone(), mangled.clone());

                    if let Some(tname) = trait_name {
                        // Trait impl: look up trait method signature and substitute
                        if let Some(trait_info) = self.traits.get(tname) {
                            for (tmethod_name, tmethod_ty) in &trait_info.methods {
                                if tmethod_name == &m.name {
                                    let concrete_ty = replace_all_vars(tmethod_ty, &impl_ty_scheme.ty);
                                    self.env.insert(mangled.clone(), Scheme::mono(concrete_ty));
                                }
                            }
                        } else {
                            self.engine.diagnostics.push(
                                Diagnostic::error(format!("Unknown trait '{}' in impl declaration", tname))
                                    .with_label(Label::primary(*span, "unknown trait")),
                            );
                        }
                    } else {
                        // Standalone impl: build a partial function type from the impl type
                        // and the number of parameters. First param gets the impl type;
                        // the rest and the return type are fresh vars (quantified so they
                        // get instantiated fresh in each call site's inference engine).
                        let mut poly_vars = Vec::new();
                        let ret_var = self.engine.fresh_var();
                        if let Ty::Var(id) = ret_var { poly_vars.push(id); }
                        let mut result_ty = ret_var;
                        for i in (0..m.params.len()).rev() {
                            let param_ty = if i == 0 {
                                impl_ty_scheme.ty.clone()
                            } else {
                                let v = self.engine.fresh_var();
                                if let Ty::Var(id) = v { poly_vars.push(id); }
                                v
                            };
                            result_ty = Ty::arrow(param_ty, result_ty);
                        }
                        self.env.insert(m.name.clone(), Scheme::poly(poly_vars.clone(), result_ty.clone()));
                        self.env.insert(mangled.clone(), Scheme::poly(poly_vars, result_ty));
                    }
                }
                self.impls.push(ImplInfo {
                    trait_name: trait_name.clone(),
                    ty: impl_ty_scheme.ty,
                    methods: impl_methods,
                });
            }
        }

        // Pass 3: type check function bodies
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
                    self.check_function(name, params, body, where_binds, *span);
                }
                Decl::EntryPoint {
                    name,
                    params,
                    body,
                    span,
                    ..
                } => {
                    self.check_function(name, params, body, &[], *span);
                }
                Decl::ImplDecl { methods, .. } => {
                    // Type-check impl method bodies
                    for m in methods {
                        self.check_function(&m.name, &m.params, &m.body, &[], m.span);
                    }
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
        let mut type_scope = self.new_type_var_scope(type_params);
        let scheme_vars = scope_vars(&type_scope);
        let result_ty = apply_type_params(name, type_params, &type_scope);
        let mut con_names = Vec::new();

        for (tag, con) in cons.iter().enumerate() {
            let con_ty = match &con.fields {
                ConFields::Empty => result_ty.clone(),
                ConFields::Positional(fields) => {
                    let mut ty = result_ty.clone();
                    for field in fields.iter().rev() {
                        let field_ty = self.convert_syntax_type_with_scope(field, &mut type_scope);
                        ty = Ty::arrow(field_ty, ty);
                    }
                    ty
                }
                ConFields::Record(fields) => {
                    // Record constructor: takes all fields positionally
                    let mut ty = result_ty.clone();
                    for f in fields.iter().rev() {
                        let ft = self.convert_syntax_type_with_scope(&f.ty, &mut type_scope);
                        ty = Ty::arrow(ft, ty);
                    }
                    ty
                }
            };

            let field_info = match &con.fields {
                ConFields::Empty => ConstructorFields::Empty,
                ConFields::Positional(fields) => ConstructorFields::Positional(
                    fields
                        .iter()
                        .map(|f| self.convert_syntax_type_with_scope(f, &mut type_scope))
                        .collect(),
                ),
                ConFields::Record(fields) => ConstructorFields::Record(
                    fields
                        .iter()
                        .map(|f| {
                            (
                                f.name.clone(),
                                self.convert_syntax_type_with_scope(&f.ty, &mut type_scope),
                            )
                        })
                        .collect(),
                ),
            };

            let resolved_tag = con.discriminant.unwrap_or(tag as i64) as u32;
            self.constructors.insert(
                con.name.clone(),
                ConstructorInfo {
                    type_name: name.to_string(),
                    tag: resolved_tag,
                    scheme_vars: scheme_vars.clone(),
                    fields: field_info,
                    result_ty: result_ty.clone(),
                },
            );

            self.env
                .insert(con.name.clone(), Scheme::poly(scheme_vars.clone(), con_ty));
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
    fn convert_syntax_type(&mut self, ty: &Type) -> Scheme {
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

    fn check_function(
        &mut self,
        name: &str,
        params: &[Pat],
        body: &Expr,
        where_binds: &[(String, Expr)],
        span: Span,
    ) {
        let mut local_env = self.env.clone();

        // Create types for parameters, flattening tuple patterns
        let mut param_types = Vec::new();
        for pat in params {
            if let Pat::Tuple(sub_pats, _) = pat {
                for sub_pat in sub_pats {
                    let ty = self.engine.fresh_var();
                    self.bind_pattern(sub_pat, &ty, &mut local_env);
                    param_types.push(ty);
                }
            } else {
                let ty = self.engine.fresh_var();
                self.bind_pattern(pat, &ty, &mut local_env);
                param_types.push(ty);
            }
        }

        // If there's a declared type, pre-unify parameter types so that
        // concrete type information (e.g. Vec dimensions) is available
        // during body inference for swizzle resolution and field access.
        // The declared type has already been flattened (tuple arrows -> curried).
        if let Some(scheme) = self.env.lookup(name) {
            let declared_ty = self.engine.instantiate(scheme);
            let mut cursor = &declared_ty;
            for param_ty in &param_types {
                if let Ty::Arrow(from, to) = cursor {
                    self.engine.unify(param_ty, from, span);
                    cursor = to;
                } else {
                    break;
                }
            }
        }

        // Infer body type. `where` is desugared to a local `let`.
        let body = desugar_where(body, where_binds, span);
        let body_ty = self.infer_expr(&body, &mut local_env);

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
                if let Some(con_info) = self
                    .constructors
                    .get(name)
                    .cloned()
                    .map(|info| info.instantiate(&mut self.engine))
                {
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
                            .with_label(Label::primary(*span, "not found"))
                            .with_help(
                                "declare it in a data definition before using it in a pattern",
                            ),
                    );
                }
            }
            Pat::Lit(lit, span) => {
                let lit_ty = match lit {
                    // Integer literals in patterns should be polymorphic over
                    // numeric types (I32, U32) — use a fresh var so the
                    // scrutinee type drives the unification.
                    Lit::Int(_) => self.engine.fresh_var(),
                    _ => self.lit_type(lit),
                };
                self.engine.unify(ty, &lit_ty, *span);
            }
            Pat::Paren(inner, _) => self.bind_pattern(inner, ty, env),
            Pat::Tuple(pats, span) => {
                let elem_tys: Vec<Ty> = pats.iter().map(|_| self.engine.fresh_var()).collect();
                let tuple_ty = if elem_tys.is_empty() {
                    Ty::unit()
                } else {
                    Ty::Tuple(elem_tys.clone())
                };
                self.engine.unify(ty, &tuple_ty, *span);
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
            Pat::Or(alternatives, _span) => {
                for alt in alternatives {
                    self.bind_pattern(alt, ty, env);
                }
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
                            .with_label(Label::primary(*span, "not in scope"))
                            .with_help(
                                "bind the name in a parameter, let, where, or import declaration",
                            ),
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
                            .with_label(Label::primary(*span, "not in scope"))
                            .with_help(
                                "declare it in a data definition before constructing it here",
                            ),
                    );
                    Ty::Error
                }
            }

            Expr::App(func, arg, span) => {
                let func_ty = self.infer_expr(func, env);
                // Flatten tuple arguments: f (a, b, c) => f a b c
                if let Expr::Tuple(elems, _) = arg.as_ref() {
                    if !elems.is_empty() {
                        let mut cur_ty = func_ty;
                        for elem in elems {
                            let arg_ty = self.infer_expr(elem, env);
                            let ret_ty = self.engine.fresh_var();
                            let expected = Ty::arrow(arg_ty, ret_ty.clone());
                            self.engine.unify(&cur_ty, &expected, *span);
                            cur_ty = ret_ty;
                        }
                        return cur_ty;
                    }
                }
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
                            .with_label(Label::primary(*span, "not in scope"))
                            .with_help("operators are regular functions; define one or import it into scope"),
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
                let tys: Vec<Ty> = elems.iter().map(|e| self.infer_expr(e, env)).collect();
                if tys.is_empty() {
                    Ty::unit()
                } else {
                    Ty::Tuple(tys)
                }
            }

            Expr::Record(name, fields, _span) => {
                for (_, expr) in fields {
                    self.infer_expr(expr, env);
                }
                // If named, resolve to the type constructor; otherwise fresh var
                if let Some(type_name) = name {
                    Ty::Con(type_name.clone())
                } else {
                    self.engine.fresh_var()
                }
            }

            Expr::FieldAccess(expr, field, span) => {
                let base_ty = self.infer_expr(expr, env);
                let base_ty = self.engine.finalize(&base_ty);

                // Check for Vec swizzle patterns
                if is_swizzle(field) {
                    if let Some((n, scalar)) = extract_vec_type(&base_ty) {
                        let swizzle_len = field.len();
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

                // Method-call syntax sugar: `x.method` → `method x`
                if let Some(scheme) = env.lookup(field) {
                    let func_ty = self.engine.instantiate(scheme);
                    let ret_ty = self.engine.fresh_var();
                    let expected = Ty::arrow(base_ty, ret_ty.clone());
                    self.engine.unify(&func_ty, &expected, *span);
                    return ret_ty;
                }

                // Fallback: fresh var (record field access)
                self.engine.fresh_var()
            }

            Expr::Index(base, index, _span) => {
                let _ = self.infer_expr(base, env);
                let _ = self.infer_expr(index, env);
                self.engine.fresh_var()
            }

            Expr::VecLit(elems, span) => {
                if elems.is_empty() {
                    self.engine.diagnostics.push(
                        Diagnostic::error("Empty vec literal")
                            .with_label(Label::primary(*span, "needs at least 2 elements"))
                            .with_help(
                                "add vector elements so the scalar type and arity can be inferred",
                            ),
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

                if !(2..=4).contains(&total_components) {
                    self.engine.diagnostics.push(
                        Diagnostic::error(format!(
                            "Vec literal has {} components, expected 2, 3, or 4",
                            total_components
                        ))
                        .with_label(Label::primary(*span, "invalid component count"))
                        .with_help("WGSL vectors must have exactly 2, 3, or 4 scalar components"),
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
                            .with_label(Label::primary(*span, "not in scope"))
                            .with_help("operators are regular functions; define one or import it into scope"),
                    );
                    Ty::Error
                }
            }

            Expr::Neg(inner, _span) => {
                // Negation works on numeric types
                self.infer_expr(inner, env)
            }

            Expr::Not(inner, span) => {
                // Boolean not: operand and result are Bool
                let inner_ty = self.infer_expr(inner, env);
                let bool_ty = Ty::Con("Bool".into());
                self.engine.unify(&inner_ty, &bool_ty, *span);
                bool_ty
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

            Expr::Loop(loop_name, bindings, body, span) => {
                let mut loop_env = env.clone();
                let mut binding_tys = Vec::new();
                for (bind_name, init_expr) in bindings {
                    let init_ty = self.infer_expr(init_expr, env);
                    loop_env.insert(bind_name.clone(), Scheme::mono(init_ty.clone()));
                    binding_tys.push(init_ty);
                }
                let result_ty = self.engine.fresh_var();
                let loop_fn_ty = binding_tys.iter().rev().fold(result_ty.clone(), |acc, ty| {
                    Ty::Arrow(Box::new(ty.clone()), Box::new(acc))
                });
                loop_env.insert(loop_name.clone(), Scheme::mono(loop_fn_ty));
                let body_ty = self.infer_expr(body, &mut loop_env);
                self.engine.unify(&result_ty, &body_ty, *span);
                result_ty
            }

            Expr::RecordUpdate(base, fields, _span) => {
                let base_ty = self.infer_expr(base, env);
                for (_, expr) in fields {
                    self.infer_expr(expr, env);
                }
                base_ty
            }
        }
    }

    fn lit_type(&self, lit: &Lit) -> Ty {
        match lit {
            Lit::Int(_) => Ty::i32(),
            Lit::UInt(_) => Ty::u32(),
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

fn desugar_where(body: &Expr, where_binds: &[(String, Expr)], span: Span) -> Expr {
    if where_binds.is_empty() {
        body.clone()
    } else {
        Expr::Let(where_binds.to_vec(), Box::new(body.clone()), span)
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Produce a short suffix string from a Ty for name-mangling.
pub fn format_type_suffix(ty: &Ty) -> String {
    match ty {
        Ty::Con(name) => name.clone(),
        Ty::App(f, a) => format!("{}_{}", format_type_suffix(f), format_type_suffix(a)),
        Ty::Nat(n) => format!("{}", n),
        _ => "unknown".to_string(),
    }
}

/// Map operator symbols to readable names for mangling.
pub fn sanitise_operator_name(name: &str) -> String {
    match name {
        "+" => "add".to_string(),
        "-" => "sub".to_string(),
        "*" => "mul".to_string(),
        "/" => "div".to_string(),
        "%" => "mod".to_string(),
        "==" => "eq".to_string(),
        "/=" => "ne".to_string(),
        "<" => "lt".to_string(),
        ">" => "gt".to_string(),
        "<=" => "le".to_string(),
        ">=" => "ge".to_string(),
        "&&" => "and".to_string(),
        "||" => "or".to_string(),
        _ => name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"),
    }
}

/// Mangle an instance method name: `(+)` for F32 → `add_F32`, `scale` for F32 → `scale_F32`.
pub fn mangle_instance_method(method_name: &str, type_suffix: &str) -> String {
    let sanitised = sanitise_operator_name(method_name);
    format!("{}_{}", sanitised, type_suffix)
}

/// Replace all `Ty::Var(_)` occurrences in a type with a concrete type.
/// Used for building concrete instance method types from trait method types
/// that have a single polymorphic type variable (the class parameter).
pub fn replace_all_vars(ty: &Ty, replacement: &Ty) -> Ty {
    match ty {
        Ty::Var(_) => replacement.clone(),
        Ty::Con(_) | Ty::Nat(_) | Ty::Error => ty.clone(),
        Ty::App(f, a) => Ty::App(
            Box::new(replace_all_vars(f, replacement)),
            Box::new(replace_all_vars(a, replacement)),
        ),
        Ty::Arrow(a, b) => Ty::Arrow(
            Box::new(replace_all_vars(a, replacement)),
            Box::new(replace_all_vars(b, replacement)),
        ),
        Ty::Tuple(elems) => Ty::Tuple(
            elems.iter().map(|e| replace_all_vars(e, replacement)).collect(),
        ),
        Ty::Forall(vars, body) => Ty::Forall(
            vars.clone(),
            Box::new(replace_all_vars(body, replacement)),
        ),
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
    let ty = normalize_type_aliases(ty);

    // Vec n T = App(App(Con("Vec"), Nat(n)), T)
    if let Ty::App(f, scalar) = &ty {
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

/// Unwrap a resource wrapper type: Uniform<T> → T, Storage<_, T> → T.
/// If the type is not a recognized wrapper, return it unchanged.
fn unwrap_resource_type(ty: &Ty) -> Ty {
    if let Ty::App(f, inner) = ty {
        if let Ty::Con(name) = f.as_ref() {
            if name == "Uniform" {
                return *inner.clone();
            }
        }
        // Storage<mode, T> = App(App(Con("Storage"), mode), T)
        if let Ty::App(ff, _mode) = f.as_ref() {
            if let Ty::Con(name) = ff.as_ref() {
                if name == "Storage" {
                    return *inner.clone();
                }
            }
        }
    }
    ty.clone()
}

/// Flatten tuple arrows: `(A, B) -> R` becomes `A -> B -> R`.
/// This allows tuple-parameter function signatures to be stored
/// in curried form, matching the flattened function definitions.
fn flatten_tuple_arrows(ty: &Ty) -> Ty {
    match ty {
        Ty::Arrow(from, to) => {
            let to_flat = flatten_tuple_arrows(to);
            if let Ty::Tuple(elems) = from.as_ref() {
                // (A, B, C) -> R => A -> B -> C -> R
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
    use fwgsl_span::Span;

    fn span() -> Span {
        Span::new(0, 0)
    }

    fn with_prelude(program: &mut Program) {
        let prelude = fwgsl_parser::prelude_program();
        let mut combined = prelude.decls.clone();
        combined.append(&mut program.decls);
        program.decls = combined;
    }

    #[test]
    fn test_empty_program() {
        let mut sa = SemanticAnalyzer::new();
        let mut program = Program { decls: vec![] };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_simple_function_inference() {
        let mut sa = SemanticAnalyzer::new();
        // f x = x + 1
        let mut program = Program {
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
                comments: vec![],
            }],
        };
        with_prelude(&mut program);
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
        let mut program = Program {
            decls: vec![Decl::DataDecl {
                name: "Color".into(),
                type_params: vec![],
                constructors: vec![
                    ConDecl {
                        name: "Red".into(),
                        fields: ConFields::Empty,
                        discriminant: None,
                        span: span(),
                    },
                    ConDecl {
                        name: "Green".into(),
                        fields: ConFields::Empty,
                        discriminant: None,
                        span: span(),
                    },
                    ConDecl {
                        name: "Blue".into(),
                        fields: ConFields::Empty,
                        discriminant: None,
                        span: span(),
                    },
                ],
                span: span(),
                comments: vec![],
            }],
        };
        with_prelude(&mut program);
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
        let mut program = Program {
            decls: vec![Decl::FunDecl {
                name: "f".into(),
                params: vec![Pat::Var("x".into(), span())],
                body: Expr::Var("y".into(), span()),
                where_binds: vec![],
                span: span(),
                comments: vec![],
            }],
        };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(sa.has_errors());
    }

    #[test]
    fn test_type_signature_check() {
        let mut sa = SemanticAnalyzer::new();
        // add :: I32 -> I32 -> I32
        // add x y = x + y
        let mut program = Program {
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
                    comments: vec![],
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
                    comments: vec![],
                },
            ],
        };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_generic_type_signature_reuses_type_variable() {
        let mut sa = SemanticAnalyzer::new();
        let mut program = Program {
            decls: vec![
                Decl::TypeSig {
                    name: "id".into(),
                    ty: Type::Arrow(
                        Box::new(Type::Var("a".into(), span())),
                        Box::new(Type::Var("a".into(), span())),
                        span(),
                    ),
                    span: span(),
                    comments: vec![],
                },
                Decl::FunDecl {
                    name: "id".into(),
                    params: vec![Pat::Var("x".into(), span())],
                    body: Expr::Var("x".into(), span()),
                    where_binds: vec![],
                    span: span(),
                    comments: vec![],
                },
            ],
        };

        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());

        let scheme = sa.env.lookup("id").expect("id should be in env");
        assert_eq!(scheme.vars.len(), 1);
        assert!(matches!(
            &scheme.ty,
            Ty::Arrow(from, to) if from.as_ref() == to.as_ref()
        ));
    }

    #[test]
    fn test_generic_constructor_pattern_inference() {
        let mut sa = SemanticAnalyzer::new();
        let mut program = Program {
            decls: vec![
                Decl::DataDecl {
                    name: "Box".into(),
                    type_params: vec!["a".into()],
                    constructors: vec![ConDecl {
                        name: "Box".into(),
                        fields: ConFields::Positional(vec![Type::Var("a".into(), span())]),
                        discriminant: None,
                        span: span(),
                    }],
                    span: span(),
                    comments: vec![],
                },
                Decl::FunDecl {
                    name: "unbox".into(),
                    params: vec![Pat::Con(
                        "Box".into(),
                        vec![Pat::Var("x".into(), span())],
                        span(),
                    )],
                    body: Expr::Var("x".into(), span()),
                    where_binds: vec![],
                    span: span(),
                    comments: vec![],
                },
            ],
        };

        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());

        let constructor = sa
            .env
            .lookup("Box")
            .expect("Box constructor should be in env");
        assert_eq!(constructor.vars.len(), 1);

        let scheme = sa.env.lookup("unbox").expect("unbox should be in env");
        assert_eq!(scheme.vars.len(), 1);
        assert!(format!("{}", scheme.ty).contains("Box"));
    }

    #[test]
    fn test_phantom_constructor_is_polymorphic() {
        let mut sa = SemanticAnalyzer::new();
        let mut program = Program {
            decls: vec![Decl::DataDecl {
                name: "Phantom".into(),
                type_params: vec!["a".into()],
                constructors: vec![ConDecl {
                    name: "Phantom".into(),
                    fields: ConFields::Empty,
                    discriminant: None,
                    span: span(),
                }],
                span: span(),
                comments: vec![],
            }],
        };

        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());

        let scheme = sa
            .env
            .lookup("Phantom")
            .expect("Phantom constructor should be in env");
        assert_eq!(scheme.vars.len(), 1);
        assert_eq!(sa.constructors["Phantom"].scheme_vars.len(), 1);
    }

    #[test]
    fn test_if_expr_type_check() {
        let mut sa = SemanticAnalyzer::new();
        // f x = if x == 0 then 1 else 2
        let mut program = Program {
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
                comments: vec![],
            }],
        };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_let_expr() {
        let mut sa = SemanticAnalyzer::new();
        // f = let x = 42 in x + 1
        let mut program = Program {
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
                comments: vec![],
            }],
        };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_where_clause() {
        let mut sa = SemanticAnalyzer::new();
        // f x = y + 1 where y = x
        let mut program = Program {
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
                comments: vec![],
            }],
        };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());

        let scheme = sa.env.lookup("f").expect("f should be in env");
        let ty = sa.engine.finalize(&scheme.ty);
        assert_eq!(format!("{}", ty), "(I32 -> I32)");
    }

    #[test]
    fn test_lambda_inference() {
        let mut sa = SemanticAnalyzer::new();
        // f = \x -> x + 1
        let mut program = Program {
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
                comments: vec![],
            }],
        };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_case_expr_with_data_type() {
        let mut sa = SemanticAnalyzer::new();
        // data Bool2 = True2 | False2
        // f x = match x | True2 -> 1 | False2 -> 0
        let mut program = Program {
            decls: vec![
                Decl::DataDecl {
                    name: "Bool2".into(),
                    type_params: vec![],
                    constructors: vec![
                        ConDecl {
                            name: "True2".into(),
                            fields: ConFields::Empty,
                            discriminant: None,
                            span: span(),
                        },
                        ConDecl {
                            name: "False2".into(),
                            fields: ConFields::Empty,
                            discriminant: None,
                            span: span(),
                        },
                    ],
                    span: span(),
                    comments: vec![],
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
                    comments: vec![],
                },
            ],
        };
        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn test_builtin_option_result_and_tensor_utilities_are_registered() {
        let mut sa = SemanticAnalyzer::new();
        let mut program = Program { decls: vec![] };
        with_prelude(&mut program);
        sa.analyze(&program);

        for name in [
            "Some",
            "None",
            "Ok",
            "Err",
            "Pair",
            "sin",
            "cos",
            "normalize",
            "length",
            "vec2",
            "vec4",
            "load",
            "toF32",
            "writeAt",
        ] {
            assert!(sa.env.lookup(name).is_some(), "missing builtin: {}", name);
        }
    }

    #[test]
    fn test_builtin_option_result_and_map_type_check() {
        let mut sa = SemanticAnalyzer::new();
        let mut program = Program {
            decls: vec![
                // lift : Option I32 -> Option I32
                // lift value = match value | Some x -> Some (x + 1) | None -> None
                Decl::TypeSig {
                    name: "lift".into(),
                    ty: Type::Arrow(
                        Box::new(Type::App(
                            Box::new(Type::Con("Option".into(), span())),
                            Box::new(Type::Con("I32".into(), span())),
                            span(),
                        )),
                        Box::new(Type::App(
                            Box::new(Type::Con("Option".into(), span())),
                            Box::new(Type::Con("I32".into(), span())),
                            span(),
                        )),
                        span(),
                    ),
                    span: span(),
                    comments: vec![],
                },
                Decl::FunDecl {
                    name: "lift".into(),
                    params: vec![Pat::Var("value".into(), span())],
                    body: Expr::Case(
                        Box::new(Expr::Var("value".into(), span())),
                        vec![
                            (
                                Pat::Con("Some".into(), vec![Pat::Var("x".into(), span())], span()),
                                Expr::App(
                                    Box::new(Expr::Var("Some".into(), span())),
                                    Box::new(Expr::Infix(
                                        Box::new(Expr::Var("x".into(), span())),
                                        "+".into(),
                                        Box::new(Expr::Lit(Lit::Int(1), span())),
                                        span(),
                                    )),
                                    span(),
                                ),
                            ),
                            (
                                Pat::Con("None".into(), vec![], span()),
                                Expr::Var("None".into(), span()),
                            ),
                        ],
                        span(),
                    ),
                    where_binds: vec![],
                    span: span(),
                    comments: vec![],
                },
                Decl::TypeSig {
                    name: "unwrap".into(),
                    ty: Type::Arrow(
                        Box::new(Type::App(
                            Box::new(Type::Con("Result".into(), span())),
                            Box::new(Type::Con("I32".into(), span())),
                            span(),
                        )),
                        Box::new(Type::Con("I32".into(), span())),
                        span(),
                    ),
                    span: span(),
                    comments: vec![],
                },
                Decl::FunDecl {
                    name: "unwrap".into(),
                    params: vec![Pat::Var("value".into(), span())],
                    body: Expr::Case(
                        Box::new(Expr::Var("value".into(), span())),
                        vec![
                            (
                                Pat::Con("Ok".into(), vec![Pat::Var("x".into(), span())], span()),
                                Expr::Var("x".into(), span()),
                            ),
                            (
                                Pat::Con("Err".into(), vec![Pat::Wild(span())], span()),
                                Expr::Lit(Lit::Int(0), span()),
                            ),
                        ],
                        span(),
                    ),
                    where_binds: vec![],
                    span: span(),
                    comments: vec![],
                },
            ],
        };

        with_prelude(&mut program);
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }
}
