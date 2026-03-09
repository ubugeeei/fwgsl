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
        self.add_builtin_data_types();
        self.add_builtin_value_schemes();

        // Arithmetic: a -> a -> a
        let scalar = fresh_var_id(&mut self.engine);
        let numeric_binop = Scheme::poly(
            vec![scalar],
            Ty::arrow(Ty::Var(scalar), Ty::arrow(Ty::Var(scalar), Ty::Var(scalar))),
        );
        for op in ["+", "-", "*", "/", "%"] {
            self.env.insert(op.to_string(), numeric_binop.clone());
        }

        // Comparison: a -> a -> Bool
        let scalar = fresh_var_id(&mut self.engine);
        let numeric_cmp = Scheme::poly(
            vec![scalar],
            Ty::arrow(Ty::Var(scalar), Ty::arrow(Ty::Var(scalar), Ty::bool())),
        );
        for op in ["==", "/=", "<", ">", "<=", ">="] {
            self.env.insert(op.to_string(), numeric_cmp.clone());
        }

        // Boolean ops
        let bool_binop = Scheme::mono(Ty::arrow(Ty::bool(), Ty::arrow(Ty::bool(), Ty::bool())));
        self.env.insert("&&".to_string(), bool_binop.clone());
        self.env.insert("||".to_string(), bool_binop);
    }

    fn add_builtin_data_types(&mut self) {
        self.register_builtin_data_type(
            "Option",
            &["a"],
            vec![
                ConDecl {
                    name: "Some".into(),
                    fields: ConFields::Positional(vec![builtin_type_var("a")]),
                    span: builtin_span(),
                },
                ConDecl {
                    name: "None".into(),
                    fields: ConFields::Empty,
                    span: builtin_span(),
                },
            ],
        );

        self.register_builtin_data_type(
            "Result",
            &["a"],
            vec![
                ConDecl {
                    name: "Ok".into(),
                    fields: ConFields::Positional(vec![builtin_type_var("a")]),
                    span: builtin_span(),
                },
                ConDecl {
                    name: "Err".into(),
                    fields: ConFields::Positional(vec![builtin_type_con("String")]),
                    span: builtin_span(),
                },
            ],
        );

        self.register_builtin_data_type(
            "Pair",
            &["a", "b"],
            vec![ConDecl {
                name: "Pair".into(),
                fields: ConFields::Positional(vec![builtin_type_var("a"), builtin_type_var("b")]),
                span: builtin_span(),
            }],
        );
    }

    fn add_builtin_value_schemes(&mut self) {
        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["id"],
            Scheme::poly(vec![a], Ty::arrow(Ty::Var(a), Ty::Var(a))),
        );

        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["const"],
            Scheme::poly(
                vec![a, b],
                Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(b), Ty::Var(a))),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let c = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["flip"],
            Scheme::poly(
                vec![a, b, c],
                Ty::arrow(
                    Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(b), Ty::Var(c))),
                    Ty::arrow(Ty::Var(b), Ty::arrow(Ty::Var(a), Ty::Var(c))),
                ),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let c = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["compose"],
            Scheme::poly(
                vec![a, b, c],
                Ty::arrow(
                    Ty::arrow(Ty::Var(b), Ty::Var(c)),
                    Ty::arrow(
                        Ty::arrow(Ty::Var(a), Ty::Var(b)),
                        Ty::arrow(Ty::Var(a), Ty::Var(c)),
                    ),
                ),
            ),
        );

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["pure"],
            Scheme::poly(
                vec![f, a],
                Ty::arrow(Ty::Var(a), Ty::app(Ty::Var(f), Ty::Var(a))),
            ),
        );

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let fa = Ty::app(Ty::Var(f), Ty::Var(a));
        let fb = Ty::app(Ty::Var(f), Ty::Var(b));
        self.insert_builtin_names(
            &["bind"],
            Scheme::poly(
                vec![f, a, b],
                Ty::arrow(fa.clone(), Ty::arrow(Ty::arrow(Ty::Var(a), fb.clone()), fb)),
            ),
        );

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let fa = Ty::app(Ty::Var(f), Ty::Var(a));
        let fb = Ty::app(Ty::Var(f), Ty::Var(b));
        let map_scheme = Scheme::poly(
            vec![f, a, b],
            Ty::arrow(
                Ty::arrow(Ty::Var(a), Ty::Var(b)),
                Ty::arrow(fa.clone(), fb.clone()),
            ),
        );
        self.insert_builtin_names(&["fmap", "map", "$map"], map_scheme);

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let fa = Ty::app(Ty::Var(f), Ty::Var(a));
        let filter_scheme = Scheme::poly(
            vec![f, a],
            Ty::arrow(Ty::arrow(Ty::Var(a), Ty::bool()), Ty::arrow(fa.clone(), fa)),
        );
        self.insert_builtin_names(&["filter", "$filter"], filter_scheme);

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let fa = Ty::app(Ty::Var(f), Ty::Var(a));
        let fold_scheme = Scheme::poly(
            vec![f, a, b],
            Ty::arrow(
                Ty::arrow(Ty::Var(b), Ty::arrow(Ty::Var(a), Ty::Var(b))),
                Ty::arrow(Ty::Var(b), Ty::arrow(fa, Ty::Var(b))),
            ),
        );
        self.insert_builtin_names(&["foldl", "fold", "$fold"], fold_scheme);

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let fa = Ty::app(Ty::Var(f), Ty::Var(a));
        let foldr_scheme = Scheme::poly(
            vec![f, a, b],
            Ty::arrow(
                Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(b), Ty::Var(b))),
                Ty::arrow(Ty::Var(b), Ty::arrow(fa, Ty::Var(b))),
            ),
        );
        self.insert_builtin_names(&["foldr", "$foldr"], foldr_scheme);

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let nested = Ty::app(Ty::Var(f), Ty::app(Ty::Var(f), Ty::Var(a)));
        let flat_scheme = Scheme::poly(
            vec![f, a],
            Ty::arrow(nested, Ty::app(Ty::Var(f), Ty::Var(a))),
        );
        self.insert_builtin_names(&["flat", "$flat"], flat_scheme);

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let fb = Ty::app(Ty::Var(f), Ty::Var(b));
        let flat_map_scheme = Scheme::poly(
            vec![f, a, b],
            Ty::arrow(
                Ty::arrow(Ty::Var(a), fb.clone()),
                Ty::arrow(Ty::app(Ty::Var(f), Ty::Var(a)), fb),
            ),
        );
        self.insert_builtin_names(&["flatMap", "$flatMap"], flat_map_scheme);

        let f = fresh_var_id(&mut self.engine);
        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        let pair_ab = pair_ty(Ty::Var(a), Ty::Var(b));
        let zip_scheme = Scheme::poly(
            vec![f, a, b],
            Ty::arrow(
                Ty::app(Ty::Var(f), Ty::Var(a)),
                Ty::arrow(
                    Ty::app(Ty::Var(f), Ty::Var(b)),
                    Ty::app(Ty::Var(f), pair_ab),
                ),
            ),
        );
        self.insert_builtin_names(&["zip", "$zip"], zip_scheme);

        let f = fresh_var_id(&mut self.engine);
        let bools = Ty::app(Ty::Var(f), Ty::bool());
        let all_scheme = Scheme::poly(vec![f], Ty::arrow(bools.clone(), Ty::bool()));
        self.insert_builtin_names(&["all", "$all"], all_scheme);

        let f = fresh_var_id(&mut self.engine);
        let bools = Ty::app(Ty::Var(f), Ty::bool());
        let any_scheme = Scheme::poly(vec![f], Ty::arrow(bools, Ty::bool()));
        self.insert_builtin_names(&["any", "$any"], any_scheme);

        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["fst", "$fst"],
            Scheme::poly(
                vec![a, b],
                Ty::arrow(pair_ty(Ty::Var(a), Ty::Var(b)), Ty::Var(a)),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["snd", "$snd"],
            Scheme::poly(
                vec![a, b],
                Ty::arrow(pair_ty(Ty::Var(a), Ty::Var(b)), Ty::Var(b)),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["swap", "$swap"],
            Scheme::poly(
                vec![a, b],
                Ty::arrow(
                    pair_ty(Ty::Var(a), Ty::Var(b)),
                    pair_ty(Ty::Var(b), Ty::Var(a)),
                ),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["unwrapOr", "$unwrapOr"],
            Scheme::poly(
                vec![a],
                Ty::arrow(Ty::Var(a), Ty::arrow(option_ty(Ty::Var(a)), Ty::Var(a))),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        let unary_numeric = Scheme::poly(vec![a], Ty::arrow(Ty::Var(a), Ty::Var(a)));
        self.insert_builtin_names(
            &[
                "$sin", "$cos", "$abs", "$fract", "$floor", "$sign", "$sqrt", "negate",
            ],
            unary_numeric,
        );

        let dim = fresh_var_id(&mut self.engine);
        let scalar = fresh_var_id(&mut self.engine);
        let vector = Ty::app(
            Ty::app(Ty::Con("Vec".into()), Ty::Var(dim)),
            Ty::Var(scalar),
        );
        self.insert_builtin_names(
            &["$normalize"],
            Scheme::poly(vec![dim, scalar], Ty::arrow(vector.clone(), vector.clone())),
        );

        let a = fresh_var_id(&mut self.engine);
        let binary_numeric = Scheme::poly(
            vec![a],
            Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(a), Ty::Var(a))),
        );
        self.insert_builtin_names(
            &["$max", "$min", "$step", "$mod", "$pow", "$reflect", "$atan"],
            binary_numeric,
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$clamp"],
            Scheme::poly(
                vec![a],
                Ty::arrow(
                    Ty::Var(a),
                    Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(a), Ty::Var(a))),
                ),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        let b = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$mix", "$smoothstep"],
            Scheme::poly(
                vec![a, b],
                Ty::arrow(
                    Ty::Var(a),
                    Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(b), Ty::Var(a))),
                ),
            ),
        );

        let dim = fresh_var_id(&mut self.engine);
        let scalar = fresh_var_id(&mut self.engine);
        let vector = Ty::app(
            Ty::app(Ty::Con("Vec".into()), Ty::Var(dim)),
            Ty::Var(scalar),
        );
        self.insert_builtin_names(
            &["$length"],
            Scheme::poly(
                vec![dim, scalar],
                Ty::arrow(vector.clone(), Ty::Var(scalar)),
            ),
        );

        let dim = fresh_var_id(&mut self.engine);
        let scalar = fresh_var_id(&mut self.engine);
        let vector = Ty::app(
            Ty::app(Ty::Con("Vec".into()), Ty::Var(dim)),
            Ty::Var(scalar),
        );
        self.insert_builtin_names(
            &["$dot"],
            Scheme::poly(
                vec![dim, scalar],
                Ty::arrow(vector.clone(), Ty::arrow(vector.clone(), Ty::Var(scalar))),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$vec2"],
            Scheme::poly(
                vec![a],
                Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(a), vector_ty(2, Ty::Var(a)))),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$vec3"],
            Scheme::poly(
                vec![a],
                Ty::arrow(
                    Ty::Var(a),
                    Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(a), vector_ty(3, Ty::Var(a)))),
                ),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$vec4"],
            Scheme::poly(
                vec![a],
                Ty::arrow(
                    Ty::Var(a),
                    Ty::arrow(
                        Ty::Var(a),
                        Ty::arrow(Ty::Var(a), Ty::arrow(Ty::Var(a), vector_ty(4, Ty::Var(a)))),
                    ),
                ),
            ),
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$splat2"],
            Scheme::poly(vec![a], Ty::arrow(Ty::Var(a), vector_ty(2, Ty::Var(a)))),
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$splat3"],
            Scheme::poly(vec![a], Ty::arrow(Ty::Var(a), vector_ty(3, Ty::Var(a)))),
        );

        let a = fresh_var_id(&mut self.engine);
        self.insert_builtin_names(
            &["$splat4"],
            Scheme::poly(vec![a], Ty::arrow(Ty::Var(a), vector_ty(4, Ty::Var(a)))),
        );

        let dim = fresh_var_id(&mut self.engine);
        let scalar = fresh_var_id(&mut self.engine);
        let vector = Ty::app(
            Ty::app(Ty::Con("Vec".into()), Ty::Var(dim)),
            Ty::Var(scalar),
        );
        self.insert_builtin_names(
            &["$vecX", "$vecY", "$vecZ", "$vecW"],
            Scheme::poly(vec![dim, scalar], Ty::arrow(vector, Ty::Var(scalar))),
        );
    }

    fn register_builtin_data_type(
        &mut self,
        name: &str,
        type_params: &[&str],
        constructors: Vec<ConDecl>,
    ) {
        let type_params: Vec<String> = type_params
            .iter()
            .map(|param| (*param).to_string())
            .collect();
        self.register_data_type(name, &type_params, &constructors, builtin_span());
    }

    fn insert_builtin_names(&mut self, names: &[&str], scheme: Scheme) {
        for name in names {
            self.env.insert((*name).to_string(), scheme.clone());
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
                self.env.insert(name.clone(), alias_ty);
            }
        }

        // Pass 2: collect type signatures
        for decl in &program.decls {
            if let Decl::TypeSig { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type(ty);
                self.env.insert(name.clone(), inferred_ty);
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
                    for (_, field_ty) in fields.iter().rev() {
                        let ft = self.convert_syntax_type_with_scope(field_ty, &mut type_scope);
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
                        .map(|(n, t)| {
                            (
                                n.clone(),
                                self.convert_syntax_type_with_scope(t, &mut type_scope),
                            )
                        })
                        .collect(),
                ),
            };

            self.constructors.insert(
                con.name.clone(),
                ConstructorInfo {
                    type_name: name.to_string(),
                    tag: tag as u32,
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
            Type::Con(name, _) => Ty::Con(name.clone()),
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
                // Tuples as type application of a tuple constructor
                if elems.is_empty() {
                    Ty::unit()
                } else {
                    // Just use first element for now
                    self.convert_syntax_type_with_scope(&elems[0], scope)
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

        // Create types for parameters
        let mut param_types = Vec::new();
        for pat in params {
            let ty = self.engine.fresh_var();
            self.bind_pattern(pat, &ty, &mut local_env);
            param_types.push(ty);
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

fn builtin_span() -> Span {
    Span::new(0, 0)
}

fn builtin_type_con(name: &str) -> Type {
    Type::Con(name.to_string(), builtin_span())
}

fn builtin_type_var(name: &str) -> Type {
    Type::Var(name.to_string(), builtin_span())
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
    fn test_generic_type_signature_reuses_type_variable() {
        let mut sa = SemanticAnalyzer::new();
        let program = Program {
            decls: vec![
                Decl::TypeSig {
                    name: "id".into(),
                    ty: Type::Arrow(
                        Box::new(Type::Var("a".into(), span())),
                        Box::new(Type::Var("a".into(), span())),
                        span(),
                    ),
                    span: span(),
                },
                Decl::FunDecl {
                    name: "id".into(),
                    params: vec![Pat::Var("x".into(), span())],
                    body: Expr::Var("x".into(), span()),
                    where_binds: vec![],
                    span: span(),
                },
            ],
        };

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
        let program = Program {
            decls: vec![
                Decl::DataDecl {
                    name: "Box".into(),
                    type_params: vec!["a".into()],
                    constructors: vec![ConDecl {
                        name: "Box".into(),
                        fields: ConFields::Positional(vec![Type::Var("a".into(), span())]),
                        span: span(),
                    }],
                    span: span(),
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
                },
            ],
        };

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
        let program = Program {
            decls: vec![Decl::DataDecl {
                name: "Phantom".into(),
                type_params: vec!["a".into()],
                constructors: vec![ConDecl {
                    name: "Phantom".into(),
                    fields: ConFields::Empty,
                    span: span(),
                }],
                span: span(),
            }],
        };

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
    fn test_where_clause() {
        let mut sa = SemanticAnalyzer::new();
        // f x = y + 1 where y = x
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

    #[test]
    fn test_builtin_option_result_and_tensor_utilities_are_registered() {
        let sa = SemanticAnalyzer::new();

        for name in [
            "Some",
            "None",
            "Ok",
            "Err",
            "Pair",
            "$map",
            "$filter",
            "$fold",
            "$foldr",
            "$zip",
            "$flat",
            "$flatMap",
            "$all",
            "$any",
            "$unwrapOr",
        ] {
            assert!(sa.env.lookup(name).is_some(), "missing builtin: {}", name);
        }
    }

    #[test]
    fn test_builtin_option_result_and_map_type_check() {
        let mut sa = SemanticAnalyzer::new();
        let program = Program {
            decls: vec![
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
                },
                Decl::FunDecl {
                    name: "lift".into(),
                    params: vec![Pat::Var("value".into(), span())],
                    body: Expr::App(
                        Box::new(Expr::App(
                            Box::new(Expr::Var("$map".into(), span())),
                            Box::new(Expr::Lambda(
                                vec![Pat::Var("x".into(), span())],
                                Box::new(Expr::Infix(
                                    Box::new(Expr::Var("x".into(), span())),
                                    "+".into(),
                                    Box::new(Expr::Lit(Lit::Int(1), span())),
                                    span(),
                                )),
                                span(),
                            )),
                            span(),
                        )),
                        Box::new(Expr::Var("value".into(), span())),
                        span(),
                    ),
                    where_binds: vec![],
                    span: span(),
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
                },
            ],
        };

        sa.analyze(&program);
        assert!(!sa.has_errors());
    }
}
