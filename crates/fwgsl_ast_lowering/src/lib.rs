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
/// Bitfield field info used during AST lowering for construction/update.
#[derive(Debug, Clone)]
pub struct BitfieldFieldMeta {
    pub offset: u32,
    pub width: u32,
}

pub struct AstLowering {
    pub env: TypeEnv,
    pub engine: InferEngine,
    pub constructors: HashMap<String, ConstructorInfo>,
    pub data_types: HashMap<String, fwgsl_semantic::DataTypeInfo>,
    pub type_aliases: HashMap<String, Ty>,
    pub traits: HashMap<String, fwgsl_semantic::TraitInfo>,
    pub impls: Vec<fwgsl_semantic::ImplInfo>,
    /// Map from bitfield type name → ordered list of (field_name, meta).
    /// Populated during `lower_program` before expressions are lowered.
    pub bitfield_fields: HashMap<String, Vec<(String, BitfieldFieldMeta)>>,
}

impl AstLowering {
    /// Create a new lowering context from a completed semantic analyzer.
    pub fn new(sa: &fwgsl_semantic::SemanticAnalyzer) -> Self {
        let mut engine = InferEngine::new();
        if let Some(max_var_id) = sa.env.max_var_id() {
            engine.reserve_above(max_var_id + 1);
        }
        Self {
            env: sa.env.clone(),
            engine,
            constructors: sa.constructors.clone(),
            data_types: sa.data_types.clone(),
            type_aliases: sa.type_aliases.clone(),
            traits: sa.traits.clone(),
            impls: sa.impls.clone(),
            bitfield_fields: HashMap::new(),
        }
    }

    /// Lower the entire program.
    pub fn lower_program(&mut self, program: &Program) -> HirProgram {
        // Flatten CfgDecl nodes so we see declarations from both branches.
        let all_decls = Decl::flatten_cfg_decls(&program.decls);

        // Pass 1: register data types (re-populate env with constructor types)
        for decl in &all_decls {
            if let Decl::DataDecl {
                name,
                type_params,
                constructors: cons,
                span: _,
                ..
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
        for decl in &all_decls {
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
            if let Decl::ExternDecl { name, ty, .. } = decl {
                let inferred_ty = self.convert_syntax_type_scheme(ty);
                let flattened = Scheme {
                    vars: inferred_ty.vars.clone(),
                    ty: flatten_tuple_arrows(&inferred_ty.ty),
                };
                self.env.insert(name.clone(), flattened);
            }
        }

        // Collect comments from TypeSig decls so they can be attached to the
        // corresponding FunDecl (type signatures don't produce output themselves).
        let mut sig_comments: HashMap<String, Vec<String>> = HashMap::new();
        for decl in &all_decls {
            if let Decl::TypeSig { name, comments, .. } = decl {
                if !comments.is_empty() {
                    sig_comments.insert(name.clone(), comments.clone());
                }
            }
        }

        let mut functions = Vec::new();
        let mut data_types = Vec::new();
        let mut entry_points = Vec::new();
        let mut resources = Vec::new();
        let mut bitfields = Vec::new();
        let mut constants = Vec::new();

        for decl in &all_decls {
            match decl {
                Decl::FunDecl {
                    name,
                    params,
                    body,
                    where_binds,
                    span,
                    comments,
                } => {
                    let merged_comments = if comments.is_empty() {
                        sig_comments.get(name).cloned().unwrap_or_default()
                    } else {
                        let mut c = sig_comments.get(name).cloned().unwrap_or_default();
                        c.extend(comments.iter().cloned());
                        c
                    };
                    if let Some(f) =
                        self.lower_fun_decl(name, params, body, where_binds, *span, merged_comments)
                    {
                        functions.push(f);
                    }
                }
                Decl::EntryPoint {
                    attributes,
                    name,
                    params,
                    body,
                    span,
                    comments,
                } => {
                    if let Some(ep) = self.lower_entry_point(
                        attributes,
                        name,
                        params,
                        body,
                        *span,
                        comments.clone(),
                    ) {
                        entry_points.push(ep);
                    }
                }
                Decl::DataDecl {
                    name,
                    type_params,
                    constructors,
                    ..
                } => {
                    data_types.push(self.lower_data_decl(name, type_params, constructors));
                }
                Decl::BindingDecl {
                    name,
                    ty,
                    address_space,
                    group,
                    binding,
                    ..
                } => {
                    let scheme = self.convert_syntax_type_scheme(ty);
                    resources.push(fwgsl_hir::HirResource {
                        name: name.clone(),
                        ty: scheme.ty,
                        address_space: match address_space {
                            fwgsl_parser::parser::BindingAddressSpace::Uniform => "Uniform".to_string(),
                            fwgsl_parser::parser::BindingAddressSpace::StorageRead => "Storage".to_string(),
                            fwgsl_parser::parser::BindingAddressSpace::StorageReadWrite => "Storage".to_string(),
                        },
                        group: *group,
                        binding: *binding,
                    });
                }
                Decl::BitfieldDecl {
                    name,
                    base_ty,
                    fields,
                    span,
                    ..
                } => {
                    let base_scheme = self.convert_syntax_type_scheme(base_ty);
                    let mut offset = 0u32;
                    let mut bf_meta = Vec::new();
                    let hir_fields: Vec<fwgsl_hir::HirBitfieldField> = fields
                        .iter()
                        .map(|f| {
                            use fwgsl_parser::parser::BitfieldFieldKind;
                            let (width, field_type) = match &f.kind {
                                BitfieldFieldKind::Bare(w) => (*w, None),
                                BitfieldFieldKind::Bool => (1, Some("Bool".to_string())),
                                BitfieldFieldKind::Typed { ty, width } => {
                                    // Validate width against type if it's an enum
                                    if let Some(dt_info) = self.data_types.get(ty.as_str()) {
                                        let count = dt_info.constructors.len() as u32;
                                        let min_bits = if count <= 1 {
                                            1
                                        } else {
                                            (count as f64).log2().ceil() as u32
                                        };
                                        if *width < min_bits {
                                            self.engine.diagnostics.push(
                                                fwgsl_diagnostics::Diagnostic::error(format!(
                                                    "bitfield field '{}' needs at least {} bits for type '{}' ({} variants), but only {} specified",
                                                    f.name, min_bits, ty, count, width
                                                ))
                                                .with_label(fwgsl_diagnostics::Label::primary(f.span, "insufficient bit width"))
                                                .with_help(format!("use at least {} bits", min_bits)),
                                            );
                                        }
                                    }
                                    (*width, Some(ty.clone()))
                                }
                                BitfieldFieldKind::EnumInferred(type_name) => {
                                    // Look up the enum type to determine bit width
                                    if let Some(dt_info) = self.data_types.get(type_name.as_str()) {
                                        let count = dt_info.constructors.len() as u32;
                                        let bits = if count <= 1 {
                                            1
                                        } else {
                                            (count as f64).log2().ceil() as u32
                                        };
                                        (bits, Some(type_name.clone()))
                                    } else {
                                        // Unknown type — default to 1 bit
                                        self.engine.diagnostics.push(
                                            fwgsl_diagnostics::Diagnostic::error(format!(
                                                "unknown type '{}' in bitfield field '{}'",
                                                type_name, f.name
                                            ))
                                            .with_label(fwgsl_diagnostics::Label::primary(f.span, "unknown type")),
                                        );
                                        (1, None)
                                    }
                                }
                            };
                            bf_meta.push((f.name.clone(), BitfieldFieldMeta { offset, width }));
                            let hf = fwgsl_hir::HirBitfieldField {
                                name: f.name.clone(),
                                offset,
                                width,
                                field_type,
                            };
                            offset += width;
                            hf
                        })
                        .collect();
                    // Validate total bit width doesn't exceed base type
                    let max_bits: u32 = match &base_scheme.ty {
                        Ty::Con(n) if n == ty_name::U32 || n == ty_name::I32 => 32,
                        Ty::Con(n) if n == "U16" => 16,
                        Ty::Con(n) if n == "U8" => 8,
                        _ => 32,
                    };
                    if offset > max_bits {
                        self.engine.diagnostics.push(
                            fwgsl_diagnostics::Diagnostic::error(format!(
                                "bitfield '{}' uses {} bits, but base type allows only {}",
                                name, offset, max_bits
                            ))
                            .with_label(fwgsl_diagnostics::Label::primary(*span, "too many bits"))
                            .with_help("reduce the number of fields or use a wider base type"),
                        );
                    }
                    self.bitfield_fields.insert(name.clone(), bf_meta);
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
                    ..
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
                Decl::TypeSig { .. } | Decl::TypeAlias { .. } | Decl::ExternDecl { .. } => {}
                Decl::ModuleDecl { .. } | Decl::ImportDecl { .. } => {
                    // Module/import declarations are handled at the module resolution level.
                }
                Decl::CfgDecl { .. } => {
                    // CfgDecl nodes are flattened by flatten_cfg_decls above — unreachable here.
                }
                Decl::TraitDecl { .. } => {
                    // Trait declarations are type-level only — no HIR output.
                }
                Decl::ImplDecl {
                    trait_name,
                    ty,
                    methods,
                    span: _,
                    comments,
                } => {
                    let impl_ty_scheme = self.convert_syntax_type_scheme(ty);
                    let type_suffix = fwgsl_semantic::format_type_suffix(&impl_ty_scheme.ty);

                    if let Some(tname) = trait_name {
                        // Trait impl: look up trait method types
                        let mut method_info: Vec<(String, Ty)> = Vec::new();
                        if let Some(trait_info) = self.traits.get(tname) {
                            for m in methods {
                                let mangled =
                                    fwgsl_semantic::mangle_instance_method(&m.name, &type_suffix);
                                for (tmethod_name, tmethod_ty) in &trait_info.methods {
                                    if tmethod_name == &m.name {
                                        let concrete_ty = fwgsl_semantic::replace_all_vars(
                                            tmethod_ty,
                                            &impl_ty_scheme.ty,
                                        );
                                        method_info.push((mangled.clone(), concrete_ty));
                                    }
                                }
                            }
                        }
                        for (i, m) in methods.iter().enumerate() {
                            if let Some((mangled, concrete_ty)) = method_info.get(i) {
                                if let Some(f) = self.lower_impl_method(
                                    mangled,
                                    &m.params,
                                    &m.body,
                                    concrete_ty,
                                    m.span,
                                    comments.clone(),
                                ) {
                                    functions.push(f);
                                }
                            }
                        }
                    } else {
                        // Standalone impl: lower each method as a regular function
                        // with the impl type as the first parameter type.
                        for m in methods {
                            let mangled =
                                fwgsl_semantic::mangle_instance_method(&m.name, &type_suffix);
                            if let Some(f) = self.lower_standalone_impl_method(
                                &mangled,
                                &m.params,
                                &m.body,
                                &impl_ty_scheme.ty,
                                m.span,
                                comments.clone(),
                            ) {
                                // Register the inferred function type so subsequent
                                // call sites (method-call syntax, pipelines) resolve
                                // the correct return type.
                                let mut fun_ty = f.return_ty.clone();
                                for (_, pty) in f.params.iter().rev() {
                                    fun_ty = Ty::arrow(pty.clone(), fun_ty);
                                }
                                let scheme = Scheme::mono(fun_ty);
                                self.env.insert(m.name.clone(), scheme.clone());
                                self.env.insert(mangled.clone(), scheme);
                                functions.push(f);
                            }
                        }
                    }
                }
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
        comments: Vec<String>,
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

        // Unify parameter types with declared type signature BEFORE lowering
        // the body, so that record update expressions can resolve concrete types.
        let ret_ty_var = self.engine.fresh_var();
        let mut fun_ty = ret_ty_var.clone();
        for pt in param_types.iter().rev() {
            fun_ty = Ty::arrow(pt.clone(), fun_ty);
        }
        if let Some(scheme) = self.env.lookup(name) {
            let declared = self.engine.instantiate(scheme);
            self.engine.unify(&fun_ty, &declared, span);
        }

        let body = desugar_where(body, where_binds, span);
        let (hir_body, body_ty) = self.lower_expr(&body, &mut local_env);

        // Unify body type with the return type from the signature
        self.engine.unify(&body_ty, &ret_ty_var, span);

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
            comments,
        })
    }

    /// Lower an impl method body into a regular HIR function.
    fn lower_impl_method(
        &mut self,
        mangled_name: &str,
        params: &[Pat],
        body: &Expr,
        concrete_ty: &Ty,
        span: Span,
        comments: Vec<String>,
    ) -> Option<HirFunction> {
        let mut local_env = self.env.clone();

        // Extract parameter types from the concrete method type (which is curried arrows)
        let mut hir_params = Vec::new();
        let mut remaining_ty = concrete_ty.clone();
        for pat in params {
            let param_ty = match &remaining_ty {
                Ty::Arrow(arg, ret) => {
                    let pt = (**arg).clone();
                    remaining_ty = (**ret).clone();
                    pt
                }
                _ => self.engine.fresh_var(),
            };
            let pname = pat_name(pat);
            self.bind_pattern(pat, &param_ty, &mut local_env);
            hir_params.push((pname, param_ty));
        }

        let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);
        self.engine.unify(&body_ty, &remaining_ty, span);

        let final_params: Vec<(String, Ty)> = hir_params
            .into_iter()
            .map(|(n, ty)| (n, self.engine.finalize(&ty)))
            .collect();
        let return_ty = self.engine.finalize(&body_ty);
        let body = self.finalize_expr(hir_body);

        Some(HirFunction {
            name: mangled_name.to_string(),
            params: final_params,
            return_ty,
            body,
            span,
            comments,
        })
    }

    /// Lower a standalone impl method — infer types from parameters and body.
    fn lower_standalone_impl_method(
        &mut self,
        mangled_name: &str,
        params: &[Pat],
        body: &Expr,
        impl_ty: &Ty,
        span: Span,
        comments: Vec<String>,
    ) -> Option<HirFunction> {
        let mut local_env = self.env.clone();
        let mut hir_params = Vec::new();

        for (i, pat) in params.iter().enumerate() {
            // First parameter gets the impl type; rest are inferred.
            let param_ty = if i == 0 {
                impl_ty.clone()
            } else {
                self.engine.fresh_var()
            };
            let pname = pat_name(pat);
            self.bind_pattern(pat, &param_ty, &mut local_env);
            hir_params.push((pname, param_ty));
        }

        let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);

        let final_params: Vec<(String, Ty)> = hir_params
            .into_iter()
            .map(|(n, ty)| (n, self.engine.finalize(&ty)))
            .collect();
        let return_ty = self.engine.finalize(&body_ty);
        let body = self.finalize_expr(hir_body);

        Some(HirFunction {
            name: mangled_name.to_string(),
            params: final_params,
            return_ty,
            body,
            span,
            comments,
        })
    }

    fn lower_entry_point(
        &mut self,
        attributes: &[Attribute],
        name: &str,
        params: &[Pat],
        body: &Expr,
        span: Span,
        comments: Vec<String>,
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

        // Unify parameter types with declared type signature BEFORE lowering
        // the body, so that record update expressions can resolve concrete types.
        let ret_ty_var = self.engine.fresh_var();
        let mut fun_ty = ret_ty_var.clone();
        for pt in param_types.iter().rev() {
            fun_ty = Ty::arrow(pt.clone(), fun_ty);
        }

        if let Some(scheme) = self.env.lookup(name) {
            let declared = self.engine.instantiate(scheme);
            self.engine.unify(&fun_ty, &declared, span);
        }

        let (hir_body, body_ty) = self.lower_expr(body, &mut local_env);
        self.engine.unify(&body_ty, &ret_ty_var, span);

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
            comments,
        })
    }

    fn lower_data_decl(&self, name: &str, type_params: &[String], cons: &[ConDecl]) -> HirDataType {
        let mut hir_cons = Vec::new();
        for (tag, con) in cons.iter().enumerate() {
            let fields = match &con.fields {
                ConFields::Empty => vec![],
                ConFields::Positional(tys) => tys
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        let ty = self.convert_syntax_type_pure(t);
                        HirFieldDef {
                            name: format!("field{}", i),
                            ty,
                            attributes: vec![],
                        }
                    })
                    .collect(),
                ConFields::Record(fields) => fields
                    .iter()
                    .map(|f| {
                        let ty = self.convert_syntax_type_pure(&f.ty);
                        let attrs = f
                            .attributes
                            .iter()
                            .map(|a| HirAttribute {
                                name: a.name.clone(),
                                args: a.args.clone(),
                            })
                            .collect();
                        HirFieldDef {
                            name: f.name.clone(),
                            ty,
                            attributes: attrs,
                        }
                    })
                    .collect(),
            };
            let resolved_tag = con.discriminant.unwrap_or(tag as i64) as u32;
            hir_cons.push(HirConstructor {
                name: con.name.clone(),
                tag: resolved_tag,
                fields,
            });
        }
        HirDataType {
            name: name.to_string(),
            type_params: type_params.to_vec(),
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
                // Desugar foldRange: foldRange start end init (\acc i -> body)
                // → loop _fr (i = start) (acc = init) in if i >= end then acc else let acc = body in _fr (i + 1) acc
                if let Some(result) = self.try_lower_fold_range(func, arg, *span, env) {
                    return result;
                }

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
                            let inner_lambda =
                                Expr::Lambda(rest_pats.to_vec(), body.clone(), *lam_span);
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
                        HirExpr::App(Box::new(hir_func), Box::new(hir_arg), ret_ty.clone(), *span),
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
                for (pat, guard, body) in arms {
                    let mut arm_env = env.clone();
                    self.bind_pattern(pat, &scrut_ty, &mut arm_env);
                    let hir_pattern = self.lower_pattern(pat, &scrut_ty);
                    let hir_guard = guard.as_ref().map(|g| {
                        let (hir_g, guard_ty) = self.lower_expr(g, &mut arm_env);
                        self.engine.unify(&guard_ty, &Ty::bool(), *span);
                        hir_g
                    });
                    let (hir_body, body_ty) = self.lower_expr(body, &mut arm_env);
                    self.engine.unify(&result_ty, &body_ty, *span);
                    hir_arms.push(HirCaseArm {
                        pattern: hir_pattern,
                        guard: hir_guard,
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

            Expr::Record(name, fields, span) => {
                if let Some(bf_name) = name {
                    if let Some(bf_fields) = self.bitfield_fields.get(bf_name).cloned() {
                        // Bitfield construction: lower to shift+OR chain
                        return self
                            .lower_bitfield_construct(bf_name, &bf_fields, fields, env, *span);
                    }
                }
                // Regular record construction: Name { f1 = e1, f2 = e2 }
                if let Some(con_name) = name {
                    if let Some(con_info) = self.constructors.get(con_name).cloned() {
                        let con_info = con_info.instantiate(&mut self.engine);
                        if let ConstructorFields::Record(con_fields) = &con_info.fields {
                            let field_map: HashMap<&str, &Expr> =
                                fields.iter().map(|(n, e)| (n.as_str(), e)).collect();
                            let mut args = Vec::new();
                            for (field_name, field_ty) in con_fields {
                                if let Some(val_expr) = field_map.get(field_name.as_str()) {
                                    let (hir_val, val_ty) = self.lower_expr(val_expr, env);
                                    self.engine.unify(&val_ty, field_ty, *span);
                                    args.push(hir_val);
                                } else {
                                    self.engine.diagnostics.push(
                                        fwgsl_diagnostics::Diagnostic::error(format!(
                                            "missing field `{}` in record construction of `{}`",
                                            field_name, con_name
                                        ))
                                        .with_label(
                                            fwgsl_diagnostics::Label::primary(
                                                *span,
                                                "missing field",
                                            ),
                                        )
                                        .with_help("all record fields must be provided in construction; add the missing field"),
                                    );
                                    args.push(HirExpr::Lit(
                                        HirLit::Int(0),
                                        field_ty.clone(),
                                        *span,
                                    ));
                                }
                            }
                            let result_ty = con_info.result_ty.clone();
                            return (
                                HirExpr::ConstructorCall(
                                    con_name.clone(),
                                    con_info.tag,
                                    args,
                                    result_ty.clone(),
                                    *span,
                                ),
                                result_ty,
                            );
                        }
                    }
                }
                // Fallback for anonymous records or unknown constructors
                if let Some((_, expr)) = fields.first() {
                    self.lower_expr(expr, env)
                } else {
                    (HirExpr::Lit(HirLit::Int(0), Ty::unit(), *span), Ty::unit())
                }
            }

            Expr::FieldAccess(expr, field, span) => {
                let (hir_expr, expr_ty) = self.lower_expr(expr, env);
                let expr_ty_final = self.engine.finalize(&expr_ty);

                // 1. Check for Vec swizzle patterns
                if fwgsl_semantic::is_swizzle(field) {
                    if let Some((n, scalar)) = fwgsl_semantic::extract_vec_type(&expr_ty_final) {
                        if fwgsl_semantic::validate_swizzle(field, n) {
                            let result_ty = if field.len() == 1 {
                                scalar
                            } else {
                                Ty::app(
                                    Ty::app(
                                        Ty::Con(ty_name::VEC.into()),
                                        Ty::Nat(field.len() as u64),
                                    ),
                                    scalar,
                                )
                            };
                            return (
                                HirExpr::FieldAccess(
                                    Box::new(hir_expr),
                                    field.clone(),
                                    result_ty.clone(),
                                    *span,
                                ),
                                result_ty,
                            );
                        }
                    }
                }

                // 2. Method-call syntax sugar: `x.method` → `method x`
                //    If the field name is a function in the env (not a struct field),
                //    desugar to function application.
                if let Some(scheme) = env.lookup(field) {
                    let func_ty = self.engine.instantiate(scheme);
                    let ret_ty = self.engine.fresh_var();
                    let expected = Ty::arrow(expr_ty, ret_ty.clone());
                    self.engine.unify(&func_ty, &expected, *span);
                    return (
                        HirExpr::App(
                            Box::new(HirExpr::Var(field.clone(), func_ty, *span)),
                            Box::new(hir_expr),
                            ret_ty.clone(),
                            *span,
                        ),
                        ret_ty,
                    );
                }

                // 3. Regular field access (struct fields, bitfield fields)
                let result_ty = if let Some(ty) = self.resolve_record_field_type(&expr_ty_final, field) {
                    ty
                } else if let Ty::Con(ref type_name) = expr_ty_final {
                    // Check if it's a valid bitfield field
                    let bf_valid = self.bitfield_fields.get(type_name.as_str())
                        .is_some_and(|fields| fields.iter().any(|(n, _)| n == field));
                    if bf_valid {
                        // Bitfield field access — type inferred from context
                        self.engine.fresh_var()
                    } else {
                        // Check if the type is a known struct or bitfield with no matching field
                        let is_known_record = self.constructors.get(type_name.as_str())
                            .is_some_and(|c| matches!(&c.fields, ConstructorFields::Record(_)));
                        let is_known_bitfield = self.bitfield_fields.contains_key(type_name.as_str());
                        if is_known_record || is_known_bitfield {
                            self.engine.diagnostics.push(
                                fwgsl_diagnostics::Diagnostic::error(format!(
                                    "no field `{}` on type `{}`",
                                    field, type_name
                                ))
                                .with_label(
                                    fwgsl_diagnostics::Label::primary(*span, "unknown field"),
                                ),
                            );
                        }
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
                let vec_name = format!("vec{}", n);
                let result_ty = Ty::app(
                    Ty::app(Ty::Con(ty_name::VEC.into()), Ty::Nat(n)),
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

            Expr::Not(inner, span) => {
                let (hir_inner, _inner_ty) = self.lower_expr(inner, env);
                let bool_ty = Ty::bool();
                (
                    HirExpr::UnaryNot(Box::new(hir_inner), bool_ty.clone(), *span),
                    bool_ty,
                )
            }

            Expr::BitNot(inner, span) => {
                let (hir_inner, inner_ty) = self.lower_expr(inner, env);
                (
                    HirExpr::UnaryBitNot(Box::new(hir_inner), inner_ty.clone(), *span),
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

            Expr::RecordUpdate(base, fields, span) => {
                let (hir_base, base_ty) = self.lower_expr(base, env);
                let base_ty_final = self.engine.finalize(&base_ty);

                // Try to resolve bitfield type from the finalized base type
                let bf_match = if let Ty::Con(ref type_name) = base_ty_final {
                    self.bitfield_fields
                        .get(type_name)
                        .cloned()
                        .map(|bf| (type_name.clone(), bf))
                } else {
                    None
                };

                // Fallback: infer bitfield type from field names
                let bf_match = bf_match.or_else(|| {
                    if let Some((first_field, _)) = fields.first() {
                        for (bf_name, bf_fields) in &self.bitfield_fields {
                            if bf_fields.iter().any(|(n, _)| n == first_field) {
                                return Some((bf_name.clone(), bf_fields.clone()));
                            }
                        }
                    }
                    None
                });

                if let Some((type_name, bf_fields)) = bf_match {
                    let ty = Ty::Con(type_name.clone());
                    self.engine.unify(&base_ty, &ty, *span);
                    return self.lower_bitfield_update(
                        &type_name, hir_base, &bf_fields, fields, env, *span,
                    );
                }

                // Non-bitfield record update: expr { field1 = val1, field2 = val2 }
                // Look up the constructor for this record type.
                let base_ty_final = self.engine.finalize(&base_ty);
                let type_name = if let Ty::Con(ref name) = base_ty_final {
                    Some(name.clone())
                } else {
                    None
                };

                // Find a record constructor for this type
                let con_match = type_name.as_ref().and_then(|tn| {
                    let dt = self.data_types.get(tn)?;
                    for con_name in &dt.constructors {
                        if let Some(con_info) = self.constructors.get(con_name) {
                            if matches!(con_info.fields, ConstructorFields::Record(_)) {
                                return Some(con_info.clone());
                            }
                        }
                    }
                    None
                });

                if let Some(con_info) = con_match {
                    let con_info = con_info.instantiate(&mut self.engine);
                    if let ConstructorFields::Record(con_fields) = &con_info.fields {
                        self.engine.unify(&base_ty, &con_info.result_ty, *span);

                        // Bind base expression to a temp variable to avoid re-evaluation
                        let base_var = format!("_rec_base_{}", span.start);
                        let base_var_ty = base_ty.clone();

                        // Lower the updated field values
                        let update_map: HashMap<&str, &Expr> =
                            fields.iter().map(|(n, e)| (n.as_str(), e)).collect();

                        // Build constructor args: updated fields use new values,
                        // unchanged fields use FieldAccess on the base variable
                        let mut args = Vec::new();
                        for (field_name, field_ty) in con_fields {
                            if let Some(val_expr) = update_map.get(field_name.as_str()) {
                                let (hir_val, val_ty) = self.lower_expr(val_expr, env);
                                self.engine.unify(&val_ty, field_ty, *span);
                                args.push(hir_val);
                            } else {
                                // Copy from base: base_var.field_name
                                args.push(HirExpr::FieldAccess(
                                    Box::new(HirExpr::Var(
                                        base_var.clone(),
                                        base_var_ty.clone(),
                                        *span,
                                    )),
                                    field_name.clone(),
                                    field_ty.clone(),
                                    *span,
                                ));
                            }
                        }

                        let result_ty = con_info.result_ty.clone();
                        let constructor_call = HirExpr::ConstructorCall(
                            con_info.type_name.clone(),
                            con_info.tag,
                            args,
                            result_ty.clone(),
                            *span,
                        );

                        // Wrap in let to bind the base: let _rec_base = <base> in Constructor(...)
                        return (
                            HirExpr::Let(
                                vec![(base_var, hir_base)],
                                Box::new(constructor_call),
                                result_ty.clone(),
                                *span,
                            ),
                            result_ty,
                        );
                    }
                }

                // Fallback: not a record type
                self.engine.diagnostics.push(
                    fwgsl_diagnostics::Diagnostic::error(
                        "Record update syntax requires a record type (data type with named fields)",
                    )
                    .with_label(fwgsl_diagnostics::Label::primary(
                        *span,
                        "not a record type",
                    ))
                    .with_help("use `expr { field = value }` only on records defined with `data Name = Name { field : Type }`"),
                );
                (hir_base, base_ty)
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
            Pat::Or(alternatives, _) => {
                let hir_alts: Vec<HirPattern> = alternatives
                    .iter()
                    .map(|p| self.lower_pattern(p, _scrutinee_ty))
                    .collect();
                HirPattern::Or(hir_alts)
            }
        }
    }

    fn lower_lit(&self, lit: &Lit) -> (HirLit, Ty) {
        match lit {
            Lit::Int(v) => (HirLit::Int(*v), Ty::i32()),
            Lit::UInt(v) => (HirLit::UInt(*v), Ty::u32()),
            Lit::Float(v) => (HirLit::Float(*v), Ty::f32()),
            Lit::String(_) => (HirLit::Int(0), Ty::Con(ty_name::STRING.into())),
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
                        self.engine.fresh_var()
                    }
                    Lit::UInt(_) => Ty::u32(),
                    Lit::Float(_) => Ty::f32(),
                    Lit::String(_) => Ty::Con(ty_name::STRING.into()),
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
            Pat::Or(alternatives, _) => {
                for alt in alternatives {
                    self.bind_pattern(alt, ty, env);
                }
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
                    Ty::Tuple(
                        elems
                            .iter()
                            .map(|e| self.convert_syntax_type_pure(e))
                            .collect(),
                    )
                }
            }
            Type::Unit(_) => Ty::unit(),
        };
        normalize_type_aliases(&ty)
    }



    pub fn has_errors(&self) -> bool {
        self.engine.diagnostics.has_errors()
    }

    pub fn diagnostics(&self) -> &fwgsl_diagnostics::DiagnosticSink {
        &self.engine.diagnostics
    }

    /// Lower bitfield construction `Name { f1 = e1, f2 = e2 }`.
    /// Produces `HirExpr::BitfieldConstruct` which the MIR lowering converts
    /// to actual shift+OR bit manipulation.
    fn lower_bitfield_construct(
        &mut self,
        bf_name: &str,
        bf_fields: &[(String, BitfieldFieldMeta)],
        user_fields: &[(String, Expr)],
        env: &mut TypeEnv,
        span: fwgsl_span::Span,
    ) -> (HirExpr, Ty) {
        let ty = Ty::Con(bf_name.to_string());
        let mut hir_fields = Vec::new();

        for (field_name, field_expr) in user_fields {
            // Find the field metadata
            let meta = bf_fields.iter().find(|(n, _)| n == field_name);
            if meta.is_none() {
                self.engine.diagnostics.push(
                    fwgsl_diagnostics::Diagnostic::error(format!(
                        "unknown bitfield field '{}' in '{}'",
                        field_name, bf_name
                    ))
                    .with_label(fwgsl_diagnostics::Label::primary(span, "unknown field"))
                    .with_help("check the bitfield definition for available field names"),
                );
                continue;
            }

            let (hir_val, _val_ty) = self.lower_expr(field_expr, env);
            hir_fields.push((field_name.clone(), hir_val));
        }

        (
            HirExpr::BitfieldConstruct(bf_name.to_string(), hir_fields, ty.clone(), span),
            ty,
        )
    }

    /// Lower bitfield functional update `base { f1 = e1, f2 = e2 }`.
    /// Produces `HirExpr::BitfieldUpdate` which the MIR lowering converts to
    /// `(base & ~mask1 & ~mask2) | ((val1 & mask1) << offset1) | ...`
    fn lower_bitfield_update(
        &mut self,
        type_name: &str,
        hir_base: HirExpr,
        bf_fields: &[(String, BitfieldFieldMeta)],
        user_fields: &[(String, Expr)],
        env: &mut TypeEnv,
        span: fwgsl_span::Span,
    ) -> (HirExpr, Ty) {
        let ty = Ty::Con(type_name.to_string());
        let mut hir_fields = Vec::new();

        for (field_name, field_expr) in user_fields {
            let meta = bf_fields.iter().find(|(n, _)| n == field_name);
            if meta.is_none() {
                self.engine.diagnostics.push(
                    fwgsl_diagnostics::Diagnostic::error(format!(
                        "unknown bitfield field '{}' in '{}'",
                        field_name, type_name
                    ))
                    .with_label(fwgsl_diagnostics::Label::primary(span, "unknown field"))
                    .with_help("check the bitfield definition for available field names"),
                );
                continue;
            }

            let (hir_val, _val_ty) = self.lower_expr(field_expr, env);
            hir_fields.push((field_name.clone(), hir_val));
        }

        (
            HirExpr::BitfieldUpdate(
                type_name.to_string(),
                Box::new(hir_base),
                hir_fields,
                ty.clone(),
                span,
            ),
            ty,
        )
    }

    /// Try to desugar `foldRange start end init (\acc i -> body)` into a Loop.
    /// Returns Some((hir_expr, ty)) if the pattern matches, None otherwise.
    fn try_lower_fold_range(
        &mut self,
        func: &Expr,
        arg: &Expr,
        span: Span,
        env: &mut TypeEnv,
    ) -> Option<(HirExpr, Ty)> {
        // Flatten: App(App(App(App(Var("foldRange"), start), end), init), lambda)
        // The outermost App has func=App(App(App(Var("foldRange"), start), end), init), arg=lambda
        // We need to peel 3 layers of App to find Var("foldRange") at the core.
        let (f3, init) = match func {
            Expr::App(f, a, _) => (f.as_ref(), a.as_ref()),
            _ => return None,
        };
        let (f2, end_expr) = match f3 {
            Expr::App(f, a, _) => (f.as_ref(), a.as_ref()),
            _ => return None,
        };
        let (f1, start_expr) = match f2 {
            Expr::App(f, a, _) => (f.as_ref(), a.as_ref()),
            _ => return None,
        };
        // Unwrap parens around the function name
        let mut head = f1;
        while let Expr::Paren(inner, _) = head {
            head = inner.as_ref();
        }
        match head {
            Expr::Var(name, _) if name == "foldRange" => {}
            _ => return None,
        }

        // arg must be a Lambda with exactly 2 params, or a named function
        let mut lambda = arg;
        while let Expr::Paren(inner, _) = lambda {
            lambda = inner.as_ref();
        }

        // Extract acc/idx names from lambda, or use defaults for named function case
        let (acc_name, idx_name, is_lambda) = match lambda {
            Expr::Lambda(pats, _, _) if pats.len() == 2 => {
                let an = match &pats[0] {
                    Pat::Var(name, _) => name.clone(),
                    _ => "_fold_acc".to_string(),
                };
                let in_ = match &pats[1] {
                    Pat::Var(name, _) => name.clone(),
                    _ => "_fold_i".to_string(),
                };
                (an, in_, true)
            }
            // Named function or other expression — we'll build App(App(f, acc), i)
            _ => ("_fold_acc".to_string(), "_fold_i".to_string(), false),
        };

        // Lower start, end, init
        let (hir_start, start_ty) = self.lower_expr(start_expr, env);
        let (hir_end, end_ty) = self.lower_expr(end_expr, env);
        let (hir_init, init_ty) = self.lower_expr(init, env);

        // Unify start and end with I32
        self.engine.unify(&start_ty, &Ty::i32(), span);
        self.engine.unify(&end_ty, &Ty::i32(), span);

        // Set up loop environment with acc and i bound
        let mut loop_env = env.clone();
        loop_env.insert(idx_name.clone(), Scheme::mono(Ty::i32()));
        loop_env.insert(acc_name.clone(), Scheme::mono(init_ty.clone()));

        // The result type is the accumulator type
        let result_ty = init_ty.clone();

        // Build the loop name function type: I32 -> acc_ty -> result_ty
        let loop_name = "_foldRange".to_string();
        let loop_fn_ty = Ty::arrow(Ty::i32(), Ty::arrow(init_ty.clone(), result_ty.clone()));
        loop_env.insert(loop_name.clone(), Scheme::mono(loop_fn_ty));

        // Lower the fold body in the loop environment
        let (hir_body, body_ty) = if is_lambda {
            // Lambda case: lower the lambda body directly (params already bound in loop_env)
            let lambda_body = match lambda {
                Expr::Lambda(_, body, _) => body.as_ref(),
                _ => unreachable!(),
            };
            self.lower_expr(lambda_body, &mut loop_env)
        } else {
            // Named function case: lower as App(App(f, acc), i)
            let (hir_f, f_ty) = self.lower_expr(lambda, &mut loop_env);
            let ret1_ty = self.engine.fresh_var();
            let expected_f = Ty::arrow(init_ty.clone(), Ty::arrow(Ty::i32(), ret1_ty.clone()));
            self.engine.unify(&f_ty, &expected_f, span);
            let app1 = HirExpr::App(
                Box::new(hir_f),
                Box::new(HirExpr::Var(acc_name.clone(), init_ty.clone(), span)),
                Ty::arrow(Ty::i32(), ret1_ty.clone()),
                span,
            );
            let app2 = HirExpr::App(
                Box::new(app1),
                Box::new(HirExpr::Var(idx_name.clone(), Ty::i32(), span)),
                ret1_ty.clone(),
                span,
            );
            (app2, ret1_ty)
        };
        self.engine.unify(&result_ty, &body_ty, span);

        // Build: if i >= end then acc else _foldRange (i + 1) (body)
        // Condition: i >= end
        let cond = HirExpr::BinOp(
            BinOp::Ge,
            Box::new(HirExpr::Var(idx_name.clone(), Ty::i32(), span)),
            Box::new(hir_end),
            Ty::bool(),
            span,
        );

        // Then branch: acc (return the accumulator)
        let then_branch = HirExpr::Var(acc_name.clone(), init_ty.clone(), span);

        // Else branch: _foldRange (i + 1) (body)
        let i_plus_1 = HirExpr::BinOp(
            BinOp::Add,
            Box::new(HirExpr::Var(idx_name.clone(), Ty::i32(), span)),
            Box::new(HirExpr::Lit(HirLit::Int(1), Ty::i32(), span)),
            Ty::i32(),
            span,
        );
        let loop_var = HirExpr::Var(
            loop_name.clone(),
            Ty::arrow(Ty::i32(), Ty::arrow(init_ty.clone(), result_ty.clone())),
            span,
        );
        let app1 = HirExpr::App(
            Box::new(loop_var),
            Box::new(i_plus_1),
            Ty::arrow(init_ty.clone(), result_ty.clone()),
            span,
        );
        let else_branch = HirExpr::App(Box::new(app1), Box::new(hir_body), result_ty.clone(), span);

        let loop_body = HirExpr::If(
            Box::new(cond),
            Box::new(then_branch),
            Box::new(else_branch),
            result_ty.clone(),
            span,
        );

        // Build HirExpr::Loop
        let hir_loop = HirExpr::Loop(
            loop_name,
            vec![(idx_name, hir_start), (acc_name, hir_init)],
            Box::new(loop_body),
            result_ty.clone(),
            span,
        );

        Some((hir_loop, result_ty))
    }

    fn finalize_expr(&self, expr: HirExpr) -> HirExpr {
        match expr {
            HirExpr::Lit(lit, ty, span) => HirExpr::Lit(lit, self.engine.finalize(&ty), span),
            HirExpr::Var(name, ty, span) => {
                let final_ty = self.engine.finalize(&ty);
                // Trait method dispatch: if this var is a trait method and the
                // type resolves to a concrete type, rewrite to the mangled impl function.
                let resolved_name = self.resolve_trait_method(&name, &final_ty);
                HirExpr::Var(resolved_name, final_ty, span)
            }
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
                        guard: arm.guard.map(|g| self.finalize_expr(g)),
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
            HirExpr::BinOp(op, lhs, rhs, ty, span) => {
                let final_lhs = self.finalize_expr(*lhs);
                let final_rhs = self.finalize_expr(*rhs);
                let final_ty = self.engine.finalize(&ty);
                let lhs_ty = final_lhs.ty().clone();

                // Check if there's a trait instance for this operator on the lhs type.
                // Built-in operator-to-trait mapping: + → Add, - → Sub, * → Mul, / → Div, etc.
                let op_str = op.to_str();
                if let Some(mangled) = self.resolve_operator_trait(op_str, &lhs_ty) {
                    // Rewrite BinOp → App(App(Var(mangled), lhs), rhs)
                    let method_ty =
                        Ty::arrow(lhs_ty, Ty::arrow(final_rhs.ty().clone(), final_ty.clone()));
                    let var_expr = HirExpr::Var(mangled, method_ty, span);
                    let partial_ty = Ty::arrow(final_rhs.ty().clone(), final_ty.clone());
                    let app1 =
                        HirExpr::App(Box::new(var_expr), Box::new(final_lhs), partial_ty, span);
                    HirExpr::App(Box::new(app1), Box::new(final_rhs), final_ty, span)
                } else {
                    HirExpr::BinOp(op, Box::new(final_lhs), Box::new(final_rhs), final_ty, span)
                }
            }
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
            HirExpr::UnaryNeg(inner, ty, span) => {
                let final_inner = self.finalize_expr(*inner);
                let final_ty = self.engine.finalize(&ty);
                let inner_ty = final_inner.ty().clone();
                if let Some(mangled) = self.resolve_operator_trait("negate", &inner_ty) {
                    let method_ty = Ty::arrow(inner_ty, final_ty.clone());
                    let var_expr = HirExpr::Var(mangled, method_ty, span);
                    HirExpr::App(Box::new(var_expr), Box::new(final_inner), final_ty, span)
                } else {
                    HirExpr::UnaryNeg(Box::new(final_inner), final_ty, span)
                }
            }
            HirExpr::UnaryNot(inner, ty, span) => HirExpr::UnaryNot(
                Box::new(self.finalize_expr(*inner)),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::UnaryBitNot(inner, ty, span) => {
                let final_inner = self.finalize_expr(*inner);
                let final_ty = self.engine.finalize(&ty);
                let inner_ty = final_inner.ty().clone();
                if let Some(mangled) = self.resolve_operator_trait("bitnot", &inner_ty) {
                    let method_ty = Ty::arrow(inner_ty, final_ty.clone());
                    let var_expr = HirExpr::Var(mangled, method_ty, span);
                    HirExpr::App(Box::new(var_expr), Box::new(final_inner), final_ty, span)
                } else {
                    HirExpr::UnaryBitNot(Box::new(final_inner), final_ty, span)
                }
            }
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
            HirExpr::BitfieldConstruct(name, fields, ty, span) => HirExpr::BitfieldConstruct(
                name,
                fields
                    .into_iter()
                    .map(|(n, e)| (n, self.finalize_expr(e)))
                    .collect(),
                self.engine.finalize(&ty),
                span,
            ),
            HirExpr::BitfieldUpdate(name, base, fields, ty, span) => HirExpr::BitfieldUpdate(
                name,
                Box::new(self.finalize_expr(*base)),
                fields
                    .into_iter()
                    .map(|(n, e)| (n, self.finalize_expr(e)))
                    .collect(),
                self.engine.finalize(&ty),
                span,
            ),
        }
    }

    /// Check if an operator (e.g. "+") has a trait impl for the given operand type.
    /// Returns the mangled function name if found (e.g. "add_Fp64"), None otherwise.
    /// Resolve a record field type from the constructor info.
    /// Given a base type like `Fp64` and a field name like `high`, returns `Some(F32)`.
    fn resolve_record_field_type(&self, base_ty: &Ty, field: &str) -> Option<Ty> {
        let type_name = match base_ty {
            Ty::Con(name) => name.as_str(),
            _ => return None,
        };
        if let Some(con_info) = self.constructors.get(type_name) {
            if let ConstructorFields::Record(fields) = &con_info.fields {
                for (fname, fty) in fields {
                    if fname == field {
                        return Some(fty.clone());
                    }
                }
            }
        }
        None
    }

    fn resolve_operator_trait(&self, op: &str, operand_ty: &Ty) -> Option<String> {
        for trait_info in self.traits.values() {
            for (method_name, _) in &trait_info.methods {
                if method_name == op {
                    for inst in &self.impls {
                        if inst.trait_name.as_deref() == Some(trait_info.name.as_str())
                            && inst.ty == *operand_ty
                        {
                            if let Some(mangled) = inst.methods.get(op) {
                                return Some(mangled.clone());
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Resolve a trait or standalone impl method name to a concrete mangled name,
    /// if the resolved type is concrete and a matching impl exists.
    fn resolve_trait_method(&self, name: &str, ty: &Ty) -> String {
        // Check trait methods
        for trait_info in self.traits.values() {
            for (method_name, _) in &trait_info.methods {
                if method_name == name {
                    let concrete = extract_first_arg_type(ty);
                    if let Some(concrete_ty) = concrete {
                        for inst in &self.impls {
                            if inst.trait_name.as_deref() == Some(trait_info.name.as_str())
                                && inst.ty == concrete_ty
                            {
                                if let Some(mangled) = inst.methods.get(name) {
                                    return mangled.clone();
                                }
                            }
                        }
                    }
                }
            }
        }
        // Check standalone impl methods
        let concrete = extract_first_arg_type(ty);
        if let Some(concrete_ty) = concrete {
            for inst in &self.impls {
                if inst.trait_name.is_none() && inst.ty == concrete_ty {
                    if let Some(mangled) = inst.methods.get(name) {
                        return mangled.clone();
                    }
                }
            }
        }
        name.to_string()
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
            HirPattern::Or(alts) => {
                HirPattern::Or(alts.into_iter().map(|p| self.finalize_pattern(p)).collect())
            }
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

/// Extract the first argument type from a curried function type.
/// `(A -> B -> C)` → `Some(A)`
fn extract_first_arg_type(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Arrow(arg, _) => Some((**arg).clone()),
        _ => None,
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

    fn with_prelude(program: &mut Program) {
        let prelude = fwgsl_parser::prelude_program();
        let mut combined = prelude.decls.clone();
        combined.append(&mut program.decls);
        program.decls = combined;
    }

    #[test]
    fn test_lower_add_function() {
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
        let mut program = Program { decls: vec![] };
        with_prelude(&mut program);
        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        let mut lowering = AstLowering::new(&sa);
        let hir = lowering.lower_program(&program);
        // Prelude contributes data types but no user functions or entry points
        assert!(hir.entry_points.is_empty());
    }

    #[test]
    fn test_lower_where_clause() {
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
                        doc: None,
                    },
                    ConDecl {
                        name: "Green".into(),
                        fields: ConFields::Empty,
                        discriminant: None,
                        span: span(),
                        doc: None,
                    },
                    ConDecl {
                        name: "Blue".into(),
                        fields: ConFields::Empty,
                        discriminant: None,
                        span: span(),
                        doc: None,
                    },
                ],
                span: span(),
                comments: vec![],
            }],
        };
        with_prelude(&mut program);

        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        let mut lowering = AstLowering::new(&sa);
        let hir = lowering.lower_program(&program);

        let dt = hir
            .data_types
            .iter()
            .find(|dt| dt.name == "Color")
            .expect("Color data type");
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
