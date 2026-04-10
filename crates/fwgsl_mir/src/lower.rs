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

use std::collections::{HashMap, HashSet};

use fwgsl_hir::*;
use fwgsl_typechecker::{ty_name, Ty};

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
    /// Generic data type info: name → (type_params, constructors with original field Tys)
    generic_types: HashMap<String, Vec<String>>,
    /// Collected concrete instantiations: (type_name, vec of concrete Ty args)
    /// e.g., ("Box", [Ty::Con("I32")]) or ("Pair", [Ty::Con("F32"), Ty::Con("I32")])
    mono_instances: Vec<(String, Vec<Ty>)>,
}

impl LowerCtx {
    fn new(hir: &HirProgram) -> Self {
        let mut data_types = HashMap::new();
        let mut constructors = HashMap::new();
        let mut generic_types = HashMap::new();

        for dt in &hir.data_types {
            data_types.insert(dt.name.clone(), dt.constructors.clone());
            for con in &dt.constructors {
                constructors.insert(
                    con.name.clone(),
                    (dt.name.clone(), con.tag, con.fields.clone()),
                );
            }
            if !dt.type_params.is_empty() {
                generic_types.insert(dt.name.clone(), dt.type_params.clone());
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

        // Collect concrete instantiations of generic types from all HIR expressions
        let mut mono_instances = Vec::new();
        let mut seen = HashSet::new();
        for f in &hir.functions {
            collect_mono_instances_from_expr(
                &f.body,
                &generic_types,
                &mut mono_instances,
                &mut seen,
            );
            for (_, ty) in &f.params {
                collect_mono_instances_from_ty(ty, &generic_types, &mut mono_instances, &mut seen);
            }
            collect_mono_instances_from_ty(
                &f.return_ty,
                &generic_types,
                &mut mono_instances,
                &mut seen,
            );
        }
        for ep in &hir.entry_points {
            collect_mono_instances_from_expr(
                &ep.body,
                &generic_types,
                &mut mono_instances,
                &mut seen,
            );
            for (_, ty) in &ep.params {
                collect_mono_instances_from_ty(ty, &generic_types, &mut mono_instances, &mut seen);
            }
            collect_mono_instances_from_ty(
                &ep.return_ty,
                &generic_types,
                &mut mono_instances,
                &mut seen,
            );
        }

        LowerCtx {
            data_types,
            constructors,
            bitfields,
            generic_types,
            mono_instances,
        }
    }

    /// Look up a bitfield field by type name and field name.
    fn lookup_bitfield_field(
        &self,
        type_name: &str,
        field_name: &str,
    ) -> Option<&BitfieldFieldInfo> {
        self.bitfields.get(type_name).and_then(|fields| {
            fields
                .iter()
                .find(|(n, _)| n == field_name)
                .map(|(_, info)| info)
        })
    }

    fn is_sum_type(&self, name: &str) -> bool {
        self.data_types.get(name).is_some_and(|cons| cons.len() > 1)
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

    /// Resolve a data type name to its (possibly monomorphized) struct name
    /// using the result type from a constructor call.
    fn resolve_struct_name(&self, dt_name: &str, result_ty: &Ty) -> String {
        if let Some((name, args)) = extract_generic_instantiation(result_ty, &self.generic_types) {
            if name == dt_name {
                return mono_mangled_name(&name, &args);
            }
        }
        dt_name.to_string()
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

// ── Monomorphization: collect and specialize generic data types ──────────────

/// Extract concrete type arguments from a type application like `App(Con("Box"), Con("I32"))`.
/// Returns Some((type_name, [arg_types])) if this is an instantiation of a generic type.
fn extract_generic_instantiation(
    ty: &Ty,
    generic_types: &HashMap<String, Vec<String>>,
) -> Option<(String, Vec<Ty>)> {
    // Peel off nested App to find the base Con and collect all args
    let mut args = Vec::new();
    let mut cursor = ty;
    loop {
        match cursor {
            Ty::App(f, arg) => {
                args.push(arg.as_ref().clone());
                cursor = f.as_ref();
            }
            Ty::Con(name) if generic_types.contains_key(name.as_str()) => {
                args.reverse(); // Args were collected right-to-left
                let expected = generic_types[name.as_str()].len();
                if args.len() == expected {
                    return Some((name.clone(), args));
                }
                return None;
            }
            _ => return None,
        }
    }
}

/// Collect concrete instantiations from a Ty.
fn collect_mono_instances_from_ty(
    ty: &Ty,
    generic_types: &HashMap<String, Vec<String>>,
    instances: &mut Vec<(String, Vec<Ty>)>,
    seen: &mut HashSet<String>,
) {
    if let Some((name, args)) = extract_generic_instantiation(ty, generic_types) {
        let key = mono_mangled_name(&name, &args);
        if seen.insert(key) {
            instances.push((name, args));
        }
        return;
    }
    match ty {
        Ty::App(f, arg) => {
            collect_mono_instances_from_ty(f, generic_types, instances, seen);
            collect_mono_instances_from_ty(arg, generic_types, instances, seen);
        }
        Ty::Arrow(a, b) => {
            collect_mono_instances_from_ty(a, generic_types, instances, seen);
            collect_mono_instances_from_ty(b, generic_types, instances, seen);
        }
        Ty::Tuple(elems) => {
            for e in elems {
                collect_mono_instances_from_ty(e, generic_types, instances, seen);
            }
        }
        _ => {}
    }
}

/// Walk an HIR expression tree to find all concrete instantiations of generic types.
fn collect_mono_instances_from_expr(
    expr: &HirExpr,
    generic_types: &HashMap<String, Vec<String>>,
    instances: &mut Vec<(String, Vec<Ty>)>,
    seen: &mut HashSet<String>,
) {
    // Collect from the type annotation on this expression
    collect_mono_instances_from_ty(expr.ty(), generic_types, instances, seen);

    match expr {
        HirExpr::Lit(_, _, _) | HirExpr::Var(_, _, _) => {}
        HirExpr::App(f, arg, _, _) => {
            collect_mono_instances_from_expr(f, generic_types, instances, seen);
            collect_mono_instances_from_expr(arg, generic_types, instances, seen);
        }
        HirExpr::Let(bindings, body, _, _) => {
            for (_, e) in bindings {
                collect_mono_instances_from_expr(e, generic_types, instances, seen);
            }
            collect_mono_instances_from_expr(body, generic_types, instances, seen);
        }
        HirExpr::Case(scrut, arms, _, _) => {
            collect_mono_instances_from_expr(scrut, generic_types, instances, seen);
            for arm in arms {
                collect_mono_instances_from_expr(&arm.body, generic_types, instances, seen);
                if let Some(guard) = &arm.guard {
                    collect_mono_instances_from_expr(guard, generic_types, instances, seen);
                }
                collect_mono_instances_from_pattern(&arm.pattern, generic_types, instances, seen);
            }
        }
        HirExpr::If(cond, then_e, else_e, _, _) => {
            collect_mono_instances_from_expr(cond, generic_types, instances, seen);
            collect_mono_instances_from_expr(then_e, generic_types, instances, seen);
            collect_mono_instances_from_expr(else_e, generic_types, instances, seen);
        }
        HirExpr::BinOp(_, lhs, rhs, _, _) => {
            collect_mono_instances_from_expr(lhs, generic_types, instances, seen);
            collect_mono_instances_from_expr(rhs, generic_types, instances, seen);
        }
        HirExpr::UnaryNeg(inner, _, _) => {
            collect_mono_instances_from_expr(inner, generic_types, instances, seen);
        }
        HirExpr::ConstructorCall(_, _, args, _, _) => {
            for arg in args {
                collect_mono_instances_from_expr(arg, generic_types, instances, seen);
            }
        }
        HirExpr::FieldAccess(base, _, _, _) => {
            collect_mono_instances_from_expr(base, generic_types, instances, seen);
        }
        HirExpr::Index(base, idx, _, _) => {
            collect_mono_instances_from_expr(base, generic_types, instances, seen);
            collect_mono_instances_from_expr(idx, generic_types, instances, seen);
        }
        HirExpr::Loop(_, bindings, body, _, _) => {
            for (_, e) in bindings {
                collect_mono_instances_from_expr(e, generic_types, instances, seen);
            }
            collect_mono_instances_from_expr(body, generic_types, instances, seen);
        }
        HirExpr::UnaryNot(inner, _, _) => {
            collect_mono_instances_from_expr(inner, generic_types, instances, seen);
        }
        HirExpr::UnaryBitNot(inner, _, _) => {
            collect_mono_instances_from_expr(inner, generic_types, instances, seen);
        }
        HirExpr::BitfieldConstruct(_, fields, _, _) => {
            for (_, e) in fields {
                collect_mono_instances_from_expr(e, generic_types, instances, seen);
            }
        }
        HirExpr::BitfieldUpdate(_, base, fields, _, _) => {
            collect_mono_instances_from_expr(base, generic_types, instances, seen);
            for (_, e) in fields {
                collect_mono_instances_from_expr(e, generic_types, instances, seen);
            }
        }
    }
}

/// Collect from pattern types (pattern variables have type annotations).
fn collect_mono_instances_from_pattern(
    pat: &HirPattern,
    generic_types: &HashMap<String, Vec<String>>,
    instances: &mut Vec<(String, Vec<Ty>)>,
    seen: &mut HashSet<String>,
) {
    match pat {
        HirPattern::Var(_, ty) => {
            collect_mono_instances_from_ty(ty, generic_types, instances, seen);
        }
        HirPattern::Constructor(_, _, sub_pats) => {
            for p in sub_pats {
                collect_mono_instances_from_pattern(p, generic_types, instances, seen);
            }
        }
        HirPattern::Or(alts) => {
            for p in alts {
                collect_mono_instances_from_pattern(p, generic_types, instances, seen);
            }
        }
        HirPattern::Wild | HirPattern::Lit(_) => {}
    }
}

/// Generate a mangled name for a monomorphized type: "Box_i32", "Pair_f32_u32"
fn mono_mangled_name(type_name: &str, args: &[Ty]) -> String {
    let mut name = type_name.to_string();
    for arg in args {
        name.push('_');
        name.push_str(&ty_to_mono_suffix(arg));
    }
    name
}

/// Convert a Ty to a suffix string for mangling.
fn ty_to_mono_suffix(ty: &Ty) -> String {
    match ty {
        Ty::Con(name) => match name.as_str() {
            ty_name::I32 => "i32".to_string(),
            ty_name::U32 => "u32".to_string(),
            ty_name::F32 => "f32".to_string(),
            ty_name::BOOL => "bool".to_string(),
            ty_name::UNIT => "unit".to_string(),
            other => other.to_lowercase(),
        },
        Ty::App(_, _) => {
            // Nested generic: e.g., Box (Option I32) → box_option_i32
            let mut parts = Vec::new();
            let mut cursor = ty;
            loop {
                match cursor {
                    Ty::App(f, arg) => {
                        parts.push(ty_to_mono_suffix(arg));
                        cursor = f.as_ref();
                    }
                    other => {
                        parts.push(ty_to_mono_suffix(other));
                        break;
                    }
                }
            }
            parts.reverse();
            parts.join("_")
        }
        Ty::Var(v) => format!("t{}", v),
        Ty::Nat(n) => format!("{}", n),
        _ => "unknown".to_string(),
    }
}

/// Substitute type parameters in a field type with concrete types.
/// e.g., for `data Box a = Box a`, substitute `a` → `I32` in field type `Con("a")`.
fn substitute_type_params(ty: &Ty, subst: &HashMap<String, Ty>) -> Ty {
    match ty {
        Ty::Con(name) => {
            if let Some(replacement) = subst.get(name.as_str()) {
                replacement.clone()
            } else {
                ty.clone()
            }
        }
        Ty::App(f, arg) => Ty::app(
            substitute_type_params(f, subst),
            substitute_type_params(arg, subst),
        ),
        Ty::Arrow(a, b) => Ty::arrow(
            substitute_type_params(a, subst),
            substitute_type_params(b, subst),
        ),
        Ty::Tuple(elems) => Ty::Tuple(
            elems
                .iter()
                .map(|e| substitute_type_params(e, subst))
                .collect(),
        ),
        _ => ty.clone(),
    }
}

/// Convert a data type's constructors into a MirStruct, if it needs one.
fn lower_data_type_to_struct(
    name: &str,
    constructors: &[HirConstructor],
    ctx: &LowerCtx,
) -> Result<Option<MirStruct>, String> {
    if constructors.len() > 1 && constructors.iter().any(|c| !c.fields.is_empty()) {
        // Sum type with fields: emit one struct with tag + union of fields
        let mut fields = vec![MirField {
            name: "tag".to_string(),
            ty: MirType::U32,
            attributes: vec![],
        }];
        let max_con = constructors.iter().max_by_key(|c| c.fields.len()).unwrap();
        for f in &max_con.fields {
            let mir_ty = ty_to_mir_type_with_ctx(&f.ty, Some(ctx))?;
            fields.push(MirField {
                name: f.name.clone(),
                ty: mir_ty,
                attributes: f
                    .attributes
                    .iter()
                    .map(|a| MirAttribute {
                        name: a.name.clone(),
                        args: a.args.clone(),
                    })
                    .collect(),
            });
        }
        Ok(Some(MirStruct {
            name: name.to_string(),
            fields,
        }))
    } else if constructors.len() == 1 {
        let con = &constructors[0];
        if !con.fields.is_empty() {
            let fields = con
                .fields
                .iter()
                .map(|f| {
                    let mir_ty = ty_to_mir_type_with_ctx(&f.ty, Some(ctx))?;
                    Ok(MirField {
                        name: f.name.clone(),
                        ty: mir_ty,
                        attributes: f
                            .attributes
                            .iter()
                            .map(|a| MirAttribute {
                                name: a.name.clone(),
                                args: a.args.clone(),
                            })
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Some(MirStruct {
                name: name.to_string(),
                fields,
            }))
        } else {
            Ok(None)
        }
    } else {
        // Pure enums (all nullary constructors) → no struct needed
        Ok(None)
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
        // Skip generic data types — they'll be emitted as specialized versions below
        if !dt.type_params.is_empty() {
            continue;
        }
        match lower_data_type_to_struct(&dt.name, &dt.constructors, &ctx) {
            Ok(Some(s)) => structs.push(s),
            Ok(None) => {}
            Err(e) => errors.push(e),
        }
    }

    // Emit specialized structs for each monomorphized instance
    for (type_name, concrete_args) in &ctx.mono_instances {
        if let Some(dt) = hir.data_types.iter().find(|d| d.name == *type_name) {
            // Build substitution: type_param_name → concrete Ty
            let subst: HashMap<String, Ty> = dt
                .type_params
                .iter()
                .zip(concrete_args.iter())
                .map(|(param, arg)| (param.clone(), arg.clone()))
                .collect();

            // Create specialized constructors with substituted field types
            let specialized_cons: Vec<HirConstructor> = dt
                .constructors
                .iter()
                .map(|con| {
                    let fields = con
                        .fields
                        .iter()
                        .map(|f| HirFieldDef {
                            name: f.name.clone(),
                            ty: substitute_type_params(&f.ty, &subst),
                            attributes: f.attributes.clone(),
                        })
                        .collect();
                    HirConstructor {
                        name: con.name.clone(),
                        tag: con.tag,
                        fields,
                    }
                })
                .collect();

            let mangled = mono_mangled_name(type_name, concrete_args);
            match lower_data_type_to_struct(&mangled, &specialized_cons, &ctx) {
                Ok(Some(s)) => structs.push(s),
                Ok(None) => {}
                Err(e) => errors.push(e),
            }
        }
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

    // Lower explicit constants
    let mut constants = Vec::new();
    let mut known_consts: HashSet<String> = HashSet::new();
    for c in &hir.constants {
        let mir_ty = match ty_to_mir_type_with_ctx(&c.ty, Some(&ctx)) {
            Ok(t) => t,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };
        match lower_hir_expr(&c.value, &ctx) {
            Ok(mir_expr) => {
                known_consts.insert(c.name.clone());
                constants.push(MirConst {
                    name: c.name.clone(),
                    ty: mir_ty,
                    value: mir_expr,
                });
            }
            Err(e) => errors.push(e),
        }
    }

    // Promote zero-param functions with const-evaluable bodies to constants.
    // This turns `maxLights = 64` into `const maxLights: i32 = 64i;` instead
    // of `fn maxLights() -> i32 { return 64i; }`.
    let mut promoted_functions = Vec::new();
    for f in functions.drain(..) {
        if f.params.is_empty()
            && f.body.is_empty()
            && f.return_ty != MirType::Unit
            && f.return_expr
                .as_ref()
                .is_some_and(|e| is_const_expr(e, &known_consts))
        {
            known_consts.insert(f.name.clone());
            constants.push(MirConst {
                name: f.name,
                ty: f.return_ty,
                value: f.return_expr.unwrap(),
            });
        } else {
            promoted_functions.push(f);
        }
    }
    let functions = promoted_functions;

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
            ty_name::I32 => Ok(MirType::I32),
            ty_name::U32 => Ok(MirType::U32),
            ty_name::F32 => Ok(MirType::F32),
            ty_name::BOOL => Ok(MirType::Bool),
            ty_name::UNIT => Ok(MirType::Unit),
            other => {
                if let Some(ctx) = ctx {
                    if let Some(mir_ty) = ctx.resolve_type_con(other) {
                        return Ok(mir_ty);
                    }
                }
                Ok(MirType::Struct(other.to_string()))
            }
        },
        Ty::App(ref f, ref arg) => {
            // Check for monomorphized generic data type: e.g., App(Con("Box"), Con("I32"))
            if let Some(ctx) = ctx {
                if let Some((name, args)) = extract_generic_instantiation(&ty, &ctx.generic_types) {
                    let mangled = mono_mangled_name(&name, &args);
                    return Ok(MirType::Struct(mangled));
                }
            }
            match f.as_ref() {
                // Unsized array: Tensor<T> (single application, no Nat dimension)
                Ty::Con(name) if name == ty_name::TENSOR => {
                    let elem = ty_to_mir_type_with_ctx(arg, ctx)?;
                    Ok(MirType::RuntimeArray(Box::new(elem)))
                }
                Ty::App(ff, n) => match (ff.as_ref(), n.as_ref()) {
                    (Ty::App(fff, nn), Ty::Nat(m)) => {
                        if let (Ty::Con(name), Ty::Nat(n)) = (fff.as_ref(), nn.as_ref()) {
                            if name == ty_name::MAT {
                                let scalar = ty_to_mir_type_with_ctx(arg, ctx)?;
                                return Ok(MirType::Mat(*n as u8, *m as u8, Box::new(scalar)));
                            }
                        }
                        Err(format!("Cannot convert to MIR type: {}", ty))
                    }
                    (Ty::Con(name), Ty::Nat(n)) if name == ty_name::VEC => {
                        let scalar = ty_to_mir_type_with_ctx(arg, ctx)?;
                        Ok(MirType::Vec(*n as u8, Box::new(scalar)))
                    }
                    (Ty::Con(name), Ty::Nat(n)) if name == ty_name::TENSOR => {
                        let elem = ty_to_mir_type_with_ctx(arg, ctx)?;
                        let len = u32::try_from(*n)
                            .map_err(|_| format!("Tensor length out of range for MIR: {}", n))?;
                        Ok(MirType::Array(Box::new(elem), len))
                    }
                    // Handle surface syntax order: Tensor<T, N> (Array<T, N>)
                    (Ty::Con(name), _elem_ty) if name == ty_name::TENSOR => {
                        if let Ty::Nat(len) = arg.as_ref() {
                            let elem = ty_to_mir_type_with_ctx(n, ctx)?;
                            let len = u32::try_from(*len).map_err(|_| {
                                format!("Tensor length out of range for MIR: {}", len)
                            })?;
                            Ok(MirType::Array(Box::new(elem), len))
                        } else {
                            Err(format!("Cannot convert to MIR type: {}", ty))
                        }
                    }
                    _ => Err(format!("Cannot convert to MIR type: {}", ty)),
                },
                _ => Err(format!("Cannot convert to MIR type: {}", ty)),
            }
        }
        Ty::Arrow(_, _) => {
            // Function types can't be represented in WGSL. This is only
            // reached for higher-order values that haven't been applied yet.
            Err("Function types cannot be represented in WGSL".into())
        }
        Ty::Var(_) => {
            // Unresolved type variables arise from unannotated polymorphic
            // definitions (e.g. `add x y = x + y`). WGSL has no polymorphism,
            // so we default to I32, matching Haskell's numeric defaulting.
            Ok(MirType::I32)
        }
        Ty::Error => Err("cannot lower error type to MIR".into()),
        _ => Err(format!("Cannot convert to MIR type: {}", ty)),
    }
}

fn lower_hir_function(f: &HirFunction, ctx: &LowerCtx) -> Result<MirFunction, String> {
    let resolve = |ty: &Ty| ty_to_mir_type_with_ctx(ty, Some(ctx));
    let params: Vec<MirParam> = f
        .params
        .iter()
        .map(|(name, ty)| {
            Ok(MirParam {
                name: name.clone(),
                ty: resolve(ty)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let return_ty = resolve(&f.return_ty)?;

    let (stmts, return_expr) = lower_hir_expr_to_stmts(&f.body, ctx)?;

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

    let (stmts, return_expr) = lower_hir_expr_to_stmts(&ep.body, ctx)?;

    // Entry point params pass through directly. Bindings like @builtin and
    // @location are carried by the struct-typed parameter's field attributes
    // (declared via `data` with attributed fields), not injected here.
    let params: Vec<MirParam> = ep
        .params
        .iter()
        .map(|(name, ty)| {
            Ok(MirParam {
                name: name.clone(),
                ty: ty_to_mir_type_with_ctx(ty, Some(ctx))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Compute shaders always have void return type in WGSL
    let return_ty = if stage == ShaderStage::Compute {
        let orig_ty = ty_to_mir_type_with_ctx(&ep.return_ty, Some(ctx)).unwrap_or(MirType::Unit);
        if orig_ty != MirType::Unit {
            eprintln!(
                "warning: compute entry point '{}' has return type '{}' but WGSL compute shaders must return void; the return expression will be discarded",
                ep.name, orig_ty
            );
        }
        MirType::Unit
    } else {
        ty_to_mir_type_with_ctx(&ep.return_ty, Some(ctx))?
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

/// Lower a HIR expression, potentially producing statements (for Let, If, Case).
/// Returns (prefix_statements, result_expression).
fn lower_hir_expr_to_stmts(
    expr: &HirExpr,
    ctx: &LowerCtx,
) -> Result<(Vec<MirStmt>, MirExpr), String> {
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
                    .or_else(|| ty_to_mir_type_with_ctx(bind_expr.ty(), Some(ctx)).ok())
                    .ok_or_else(|| format!("cannot resolve type for let-binding '{}'", name))?;
                stmts.push(MirStmt::Let(name.clone(), bind_ty, bind_val));
            }
            let (mut body_stmts, body_expr) = lower_hir_expr_to_stmts(body, ctx)?;
            stmts.append(&mut body_stmts);
            Ok((stmts, body_expr))
        }

        HirExpr::If(cond, then_expr, else_expr, ty, _span) => {
            let cond_mir = lower_hir_expr(cond, ctx)?;
            let result_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;

            let (then_stmts, then_val) = lower_hir_expr_to_stmts(then_expr, ctx)?;
            let (else_stmts, else_val) = lower_hir_expr_to_stmts(else_expr, ctx)?;

            // Simple ternary: both branches are pure expressions with no statements.
            // Emit select(false_val, true_val, condition) instead of var/if/assign.
            if then_stmts.is_empty() && else_stmts.is_empty() {
                let select_expr = MirExpr::Call(
                    "select".to_string(),
                    vec![else_val, then_val, cond_mir],
                    result_ty,
                );
                return Ok((vec![], select_expr));
            }

            // Complex branches: fall back to var tmp; if (cond) { ... } else { ... }
            let tmp_name = format!("_if_tmp_{}", _span.start);

            let mut then_body = then_stmts;
            then_body.push(MirStmt::Assign(tmp_name.clone(), then_val));

            let mut else_body = else_stmts;
            else_body.push(MirStmt::Assign(tmp_name.clone(), else_val));

            let stmts = vec![
                MirStmt::Var(
                    tmp_name.clone(),
                    result_ty.clone(),
                    default_expr_for_type(&result_ty),
                ),
                MirStmt::If(cond_mir, then_body, else_body),
            ];

            Ok((stmts, MirExpr::Var(tmp_name, result_ty)))
        }

        HirExpr::Case(scrutinee, arms, ty, _span) => {
            let result_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
            let tmp_name = format!("_case_tmp_{}", _span.start);
            let scrut_mir = lower_hir_expr(scrutinee, ctx)?;
            let scrut_ty = ty_to_mir_type_with_ctx(scrutinee.ty(), Some(ctx))?;
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
                let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
                let result = lower_app_with_args(&func_name, mir_args, mir_ty, ctx)?;
                Ok((pre_stmts, result))
            }
        }

        HirExpr::Loop(loop_name, bindings, body, ty, _span) => {
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
            let result_var = format!("{}_result", loop_name);

            let mut pre_stmts = Vec::new();

            // Emit `var` for each binding
            let mut binding_names = Vec::new();
            let mut binding_types = Vec::new();
            for (bind_name, init_expr) in bindings {
                let (mut init_stmts, init_mir) = lower_hir_expr_to_stmts(init_expr, ctx)?;
                let bind_ty = init_mir
                    .result_type()
                    .or_else(|| ty_to_mir_type_with_ctx(init_expr.ty(), Some(ctx)).ok())
                    .ok_or_else(|| {
                        format!("cannot resolve type for loop binding '{}'", bind_name)
                    })?;
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
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
            Ok(MirExpr::Var(name.clone(), mir_ty))
        }

        HirExpr::BinOp(op, lhs, rhs, ty, _span) => {
            let mir_lhs = lower_hir_expr(lhs, ctx)?;
            let mir_rhs = lower_hir_expr(rhs, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
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
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
            let mir_args = mir_args?;
            lower_app_with_args(&func_name, mir_args, mir_ty, ctx)
        }

        HirExpr::ConstructorCall(name, _tag, args, _ty, _span) => {
            // Check if this constructor belongs to a sum type with fields
            if let Some((dt_name, tag, _fields)) = ctx.constructors.get(name.as_str()) {
                if ctx.is_sum_type(dt_name) && ctx.has_fields(dt_name) {
                    // Emit as DataType struct: DataType(tag, field0, ...)
                    let struct_name = ctx.resolve_struct_name(dt_name, _ty);
                    let mut all_args = vec![MirExpr::Lit(MirLit::U32(*tag))];
                    for arg in args {
                        all_args.push(lower_hir_expr(arg, ctx)?);
                    }
                    return Ok(MirExpr::ConstructStruct(struct_name, all_args));
                }
            }
            if args.is_empty() {
                // Nullary constructor: emit as u32 tag literal
                Ok(MirExpr::Lit(MirLit::U32(*_tag)))
            } else {
                // Record constructor: emit as struct construction
                let struct_name = if let Some((dt_name, _, _)) = ctx.constructors.get(name.as_str())
                {
                    ctx.resolve_struct_name(dt_name, _ty)
                } else {
                    name.clone()
                };
                let mir_args: Result<Vec<MirExpr>, String> =
                    args.iter().map(|a| lower_hir_expr(a, ctx)).collect();
                Ok(MirExpr::ConstructStruct(struct_name, mir_args?))
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
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
            Ok(MirExpr::FieldAccess(
                Box::new(mir_expr),
                field.clone(),
                mir_ty,
            ))
        }
        HirExpr::Index(base, index, ty, _span) => {
            let mir_base = lower_hir_expr(base, ctx)?;
            let mir_index = lower_hir_expr(index, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
            Ok(MirExpr::Index(
                Box::new(mir_base),
                Box::new(mir_index),
                mir_ty,
            ))
        }

        HirExpr::UnaryNeg(inner, ty, _span) => {
            let mir_inner = lower_hir_expr(inner, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
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

        HirExpr::UnaryBitNot(inner, ty, _span) => {
            let mir_inner = lower_hir_expr(inner, ctx)?;
            let mir_ty = ty_to_mir_type_with_ctx(ty, Some(ctx))?;
            Ok(MirExpr::UnaryOp(
                MirUnaryOp::BitNot,
                Box::new(mir_inner),
                mir_ty,
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
        let bf_info = ctx
            .lookup_bitfield_field(type_name, field_name)
            .ok_or_else(|| format!("unknown bitfield field '{}' in '{}'", field_name, type_name))?;

        let mir_val = lower_hir_expr(field_expr, ctx)?;
        let mask = (1u32 << bf_info.width) - 1;

        // For 1-bit bool fields: select(0u, 1u, val)  →  if val then 1u else 0u
        // At MIR level we just do: u32(val) which maps to a cast, but simpler:
        // just mask the value which already works for u32, and for bool we
        // wrap in a conditional
        let coerced = if bf_info.width == 1 {
            let val_ty = mir_val.result_type();
            if val_ty.as_ref() == Some(&MirType::Bool) {
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
                // Integer → u32: cast directly
                if val_ty.as_ref() == Some(&MirType::U32) {
                    mir_val
                } else {
                    MirExpr::Cast(Box::new(mir_val), MirType::U32)
                }
            }
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
        let bf_info = ctx
            .lookup_bitfield_field(type_name, field_name)
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
        let bf_info = ctx
            .lookup_bitfield_field(type_name, field_name)
            .ok_or_else(|| format!("unknown bitfield field '{}' in '{}'", field_name, type_name))?;

        let mir_val = lower_hir_expr(field_expr, ctx)?;
        let mask = (1u32 << bf_info.width) - 1;

        let coerced = if bf_info.width == 1 {
            let val_ty = mir_val.result_type();
            if val_ty.as_ref() == Some(&MirType::Bool) {
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
                // Integer → u32: cast directly
                if val_ty.as_ref() == Some(&MirType::U32) {
                    mir_val
                } else {
                    MirExpr::Cast(Box::new(mir_val), MirType::U32)
                }
            }
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
                then_branch,
                loop_name,
                binding_names,
                binding_types,
                result_var,
                result_ty,
                ctx,
            )?;
            let else_stmts = lower_loop_body(
                else_branch,
                loop_name,
                binding_names,
                binding_types,
                result_var,
                result_ty,
                ctx,
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
                let bind_ty = init_mir
                    .result_type()
                    .or_else(|| ty_to_mir_type_with_ctx(init_expr.ty(), Some(ctx)).ok())
                    .ok_or_else(|| {
                        format!("cannot resolve type for loop let-binding '{}'", name)
                    })?;
                stmts.append(&mut init_stmts);
                stmts.push(MirStmt::Let(name.clone(), bind_ty, init_mir));
            }
            let mut body_stmts = lower_loop_body(
                inner_body,
                loop_name,
                binding_names,
                binding_types,
                result_var,
                result_ty,
                ctx,
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
        ("mod", [lhs, rhs]) => Ok(MirExpr::BinOp(
            MirBinOp::Mod,
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
            mir_ty,
        )),
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
        ("bor", [lhs, rhs]) => Ok(MirExpr::BinOp(
            MirBinOp::BitOr,
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
            mir_ty,
        )),
        ("shr", [lhs, rhs]) => Ok(MirExpr::BinOp(
            MirBinOp::Shr,
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
            mir_ty,
        )),
        ("splat2", [arg]) => Ok(MirExpr::Call("vec2".to_string(), vec![arg.clone()], mir_ty)),
        ("splat3", [arg]) => Ok(MirExpr::Call("vec3".to_string(), vec![arg.clone()], mir_ty)),
        ("splat4", [arg]) => Ok(MirExpr::Call("vec4".to_string(), vec![arg.clone()], mir_ty)),
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
            // Check if this is a constructor call for a data type with fields
            if let Some((dt_name, tag, _fields)) = ctx.constructors.get(func_name) {
                // Use the monomorphized struct name from the result type if available
                let struct_name = if let MirType::Struct(name) = &mir_ty {
                    name.clone()
                } else {
                    dt_name.clone()
                };
                if ctx.is_sum_type(dt_name) && ctx.has_fields(dt_name) {
                    let mut all_args = vec![MirExpr::Lit(MirLit::U32(*tag))];
                    all_args.extend(mir_args);
                    return Ok(MirExpr::ConstructStruct(struct_name, all_args));
                } else if ctx.has_fields(dt_name) {
                    // Single-constructor type with fields
                    return Ok(MirExpr::ConstructStruct(struct_name, mir_args));
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
    // Guards disable the switch optimization (can't express arbitrary conditions in WGSL switch).
    let no_guards = arms.iter().all(|arm| arm.guard.is_none());
    if no_guards && matches!(scrut_ty, MirType::I32 | MirType::U32) {
        let all_int_or_wild = arms
            .iter()
            .all(|arm| is_switch_compatible_pattern(&arm.pattern));

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
                    HirPattern::Wild => {
                        default_body = body_stmts;
                    }
                    HirPattern::Var(name, var_ty) => {
                        let mir_ty =
                            ty_to_mir_type_with_ctx(var_ty, Some(ctx)).unwrap_or(scrut_ty.clone());
                        let mut stmts = vec![MirStmt::Let(
                            name.clone(),
                            mir_ty,
                            MirExpr::Var(scrut_name.to_string(), scrut_ty.clone()),
                        )];
                        stmts.extend(body_stmts);
                        default_body = stmts;
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

        // Lower the optional guard expression.
        // Guard may reference pattern-bound variables, so it must be evaluated
        // *after* pattern bindings are established.
        let guard_mir = if let Some(guard_expr) = &arm.guard {
            Some(lower_hir_expr_to_stmts(guard_expr, ctx)?)
        } else {
            None
        };

        /// Wrap body_stmts with a guard check if present.
        /// Returns: `guard_setup; if (guard) { body } else { fallthrough }` when guarded,
        /// or just `body` when unguarded.
        fn apply_guard(
            guard_mir: Option<(Vec<MirStmt>, MirExpr)>,
            body_stmts: Vec<MirStmt>,
            fallthrough: &mut Option<MirStmt>,
        ) -> Vec<MirStmt> {
            if let Some((guard_setup_stmts, guard_val)) = guard_mir {
                let else_stmts = take_else_stmts(fallthrough);
                let mut guarded = guard_setup_stmts;
                guarded.push(MirStmt::If(guard_val, body_stmts, else_stmts));
                guarded
            } else {
                body_stmts
            }
        }

        fn take_else_stmts(result: &mut Option<MirStmt>) -> Vec<MirStmt> {
            match result.take() {
                Some(MirStmt::If(c, t, e)) => vec![MirStmt::If(c, t, e)],
                Some(MirStmt::Block(stmts)) => stmts,
                Some(other) => vec![other],
                None => vec![],
            }
        }

        match &arm.pattern {
            HirPattern::Wild | HirPattern::Var(_, _) => {
                // Bind Var pattern to scrutinee so body/guard can reference it
                let mut binding = Vec::new();
                if let HirPattern::Var(name, var_ty) = &arm.pattern {
                    let mir_ty =
                        ty_to_mir_type_with_ctx(var_ty, Some(ctx)).unwrap_or(scrut_ty.clone());
                    binding.push(MirStmt::Let(
                        name.clone(),
                        mir_ty,
                        MirExpr::Var(scrut_name.to_string(), scrut_ty.clone()),
                    ));
                }
                let guarded_body = apply_guard(guard_mir, body_stmts, &mut result);
                let mut full = binding;
                full.extend(guarded_body);
                result = Some(MirStmt::Block(full));
            }

            HirPattern::Constructor(con_name, tag, sub_pats) => {
                // Bind pattern variables from the scrutinee's fields
                let mut bindings = Vec::new();
                let is_single_con =
                    if let Some((dt_name, _, _)) = ctx.constructors.get(con_name.as_str()) {
                        !ctx.is_sum_type(dt_name)
                    } else {
                        false
                    };

                if let Some((dt_name, _, con_fields)) = ctx.constructors.get(con_name.as_str()) {
                    if ctx.has_fields(dt_name) {
                        // Extract fields from struct (works for both sum and single-con types)
                        for (i, pat) in sub_pats.iter().enumerate() {
                            if let HirPattern::Var(var_name, var_ty) = pat {
                                let field_name = if i < con_fields.len() {
                                    con_fields[i].name.clone()
                                } else {
                                    format!("field{}", i)
                                };
                                let mir_ty = ty_to_mir_type_with_ctx(var_ty, Some(ctx))?;
                                bindings.push(MirStmt::Let(
                                    var_name.clone(),
                                    mir_ty.clone(),
                                    MirExpr::FieldAccess(
                                        Box::new(MirExpr::Var(
                                            scrut_name.to_string(),
                                            scrut_ty.clone(),
                                        )),
                                        field_name,
                                        mir_ty,
                                    ),
                                ));
                            }
                        }
                    }
                }

                if is_single_con {
                    // Single-constructor type: no tag check needed, just emit bindings + body
                    let guarded_body = apply_guard(guard_mir, body_stmts, &mut result);
                    let mut full = bindings;
                    full.extend(guarded_body);
                    result = Some(MirStmt::Block(full));
                } else {
                    // Sum type: match on tag
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

                    let mut full_body = bindings;
                    if guard_mir.is_some() {
                        let fallthrough_clone = result.clone();
                        let guarded_body = apply_guard(guard_mir, body_stmts, &mut result);
                        full_body.extend(guarded_body);
                        result = fallthrough_clone;
                    } else {
                        full_body.extend(body_stmts);
                    }

                    let else_stmts = take_else_stmts(&mut result);
                    result = Some(MirStmt::If(cond, full_body, else_stmts));
                }
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

                if guard_mir.is_some() {
                    let fallthrough_clone = result.clone();
                    let guarded_body = apply_guard(guard_mir, body_stmts, &mut result);
                    result = fallthrough_clone;
                    let else_stmts = take_else_stmts(&mut result);
                    result = Some(MirStmt::If(cond, guarded_body, else_stmts));
                } else {
                    let else_stmts = take_else_stmts(&mut result);
                    result = Some(MirStmt::If(cond, body_stmts, else_stmts));
                }
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
                    if guard_mir.is_some() {
                        let fallthrough_clone = result.clone();
                        let guarded_body = apply_guard(guard_mir, body_stmts, &mut result);
                        result = fallthrough_clone;
                        let else_stmts = take_else_stmts(&mut result);
                        result = Some(MirStmt::If(cond, guarded_body, else_stmts));
                    } else {
                        let else_stmts = take_else_stmts(&mut result);
                        result = Some(MirStmt::If(cond, body_stmts, else_stmts));
                    }
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
        BinOp::BitAnd => MirBinOp::BitAnd,
        BinOp::BitOr => MirBinOp::BitOr,
        BinOp::BitXor => MirBinOp::BitXor,
        BinOp::Shl => MirBinOp::Shl,
        BinOp::Shr => MirBinOp::Shr,
    }
}

fn lower_hir_lit(lit: &HirLit, ty: &Ty) -> MirLit {
    match lit {
        HirLit::Int(v) => match ty {
            Ty::Con(name) if name == ty_name::U32 => MirLit::U32(*v as u32),
            Ty::Con(name) if name == ty_name::F32 => MirLit::F32(*v as f64),
            _ => MirLit::I32(*v as i32),
        },
        HirLit::UInt(v) => MirLit::U32(*v as u32),
        HirLit::Float(v) => MirLit::F32(*v),
        HirLit::Bool(v) => MirLit::Bool(*v),
    }
}

/// Lower a HIR resource declaration to a MIR global binding.
/// The HIR type is already the inner type (no Uniform/Storage wrappers).
/// The address space is determined from the `address_space` hint string.
fn lower_hir_resource(res: &HirResource, ctx: &LowerCtx) -> Option<MirGlobal> {
    let mir_ty = ty_to_mir_type_with_ctx(&res.ty, Some(ctx)).ok()?;
    let address_space = match res.address_space.as_str() {
        "Uniform" => AddressSpace::Uniform,
        "StorageRead" => AddressSpace::StorageRead,
        "StorageReadWrite" => AddressSpace::StorageReadWrite,
        s if s.contains("Storage") => AddressSpace::StorageReadWrite,
        _ => AddressSpace::Uniform,
    };
    Some(MirGlobal {
        name: res.name.clone(),
        address_space,
        ty: mir_ty,
        group: res.group,
        binding: res.binding,
    })
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

// ---------------------------------------------------------------------------
// Const-expression analysis for zero-param function promotion
// ---------------------------------------------------------------------------

/// WGSL built-in functions that are const-evaluable (can appear in `const` declarations).
/// See WGSL spec §16.1 "Const-expressions".
fn is_wgsl_const_builtin(name: &str) -> bool {
    matches!(
        name,
        // Type constructors
        "vec2" | "vec3" | "vec4"
        | "mat2x2" | "mat2x3" | "mat2x4"
        | "mat3x2" | "mat3x3" | "mat3x4"
        | "mat4x2" | "mat4x3" | "mat4x4"
        | "array"
        // Numeric builtins
        | "abs" | "clamp" | "max" | "min" | "sign"
        | "countLeadingZeros" | "countOneBits" | "countTrailingZeros"
        | "extractBits" | "firstLeadingBit" | "firstTrailingBit"
        | "insertBits" | "reverseBits"
        // Logical
        | "all" | "any" | "select"
    )
}

/// Check whether a MIR expression is a valid WGSL const-expression.
///
/// `known_consts` contains names of other module-level constants that have
/// already been emitted (or will be emitted) as `const` declarations.
fn is_const_expr(expr: &MirExpr, known_consts: &HashSet<String>) -> bool {
    match expr {
        MirExpr::Lit(_) => true,
        MirExpr::Var(name, _) => known_consts.contains(name.as_str()),
        MirExpr::BinOp(_, lhs, rhs, _) => {
            is_const_expr(lhs, known_consts) && is_const_expr(rhs, known_consts)
        }
        MirExpr::UnaryOp(_, operand, _) => is_const_expr(operand, known_consts),
        MirExpr::Cast(inner, _) => is_const_expr(inner, known_consts),
        MirExpr::Call(name, args, _) => {
            is_wgsl_const_builtin(name) && args.iter().all(|a| is_const_expr(a, known_consts))
        }
        // Struct construction, field access, index — not const-evaluable in WGSL
        MirExpr::ConstructStruct(_, _)
        | MirExpr::FieldAccess(_, _, _)
        | MirExpr::Index(_, _, _) => false,
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
        // Simple if-then-else with no side effects should lower to a select() call
        // which becomes the return expression directly
        assert!(
            f.return_expr.is_some(),
            "should have a return expression (select call)"
        );
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

    #[test]
    fn test_zero_param_literal_promoted_to_const() {
        // maxLights = 64  →  should become `const maxLights: i32 = 64i;`
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "maxLights".into(),
                params: vec![],
                return_ty: Ty::i32(),
                body: HirExpr::Lit(HirLit::Int(64), Ty::i32(), span()),
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
        assert_eq!(
            mir.functions.len(),
            0,
            "zero-param literal function should be promoted"
        );
        assert_eq!(mir.constants.len(), 1);
        assert_eq!(mir.constants[0].name, "maxLights");
        assert_eq!(mir.constants[0].ty, MirType::I32);
        assert_eq!(mir.constants[0].value, MirExpr::Lit(MirLit::I32(64)));
    }

    #[test]
    fn test_zero_param_arithmetic_promoted_to_const() {
        // stride = 4 * 3  →  should become `const stride: i32 = (4i * 3i);`
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "stride".into(),
                params: vec![],
                return_ty: Ty::i32(),
                body: HirExpr::BinOp(
                    BinOp::Add,
                    Box::new(HirExpr::Lit(HirLit::Int(4), Ty::i32(), span())),
                    Box::new(HirExpr::Lit(HirLit::Int(3), Ty::i32(), span())),
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
        assert_eq!(
            mir.functions.len(),
            0,
            "zero-param arithmetic function should be promoted"
        );
        assert_eq!(mir.constants.len(), 1);
        assert_eq!(mir.constants[0].name, "stride");
    }

    #[test]
    fn test_zero_param_with_let_body_not_promoted() {
        // f = let x = 42 in x + 1  →  should stay as function (has statements)
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
        assert_eq!(
            mir.functions.len(),
            1,
            "let-body function should NOT be promoted"
        );
        assert_eq!(mir.constants.len(), 0);
    }

    #[test]
    fn test_function_with_params_not_promoted() {
        // f x = x  →  should stay as function
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "f".into(),
                params: vec![("x".into(), Ty::i32())],
                return_ty: Ty::i32(),
                body: HirExpr::Var("x".into(), Ty::i32(), span()),
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
        assert_eq!(
            mir.functions.len(),
            1,
            "function with params should NOT be promoted"
        );
        assert_eq!(mir.constants.len(), 0);
    }

    #[test]
    fn test_zero_param_negation_promoted_to_const() {
        // neg1 = -1  →  should become `const neg1: i32 = -(1i);`
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "neg1".into(),
                params: vec![],
                return_ty: Ty::i32(),
                body: HirExpr::UnaryNeg(
                    Box::new(HirExpr::Lit(HirLit::Int(1), Ty::i32(), span())),
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
        assert_eq!(
            mir.functions.len(),
            0,
            "zero-param negation should be promoted"
        );
        assert_eq!(mir.constants.len(), 1);
        assert_eq!(mir.constants[0].name, "neg1");
    }

    #[test]
    fn test_zero_param_unit_return_not_promoted() {
        // sideEffect = ()  →  should stay as function (Unit return type)
        let hir = HirProgram {
            functions: vec![HirFunction {
                name: "sideEffect".into(),
                params: vec![],
                return_ty: Ty::unit(),
                body: HirExpr::Lit(HirLit::Int(0), Ty::unit(), span()),
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
        assert_eq!(
            mir.functions.len(),
            1,
            "Unit-returning function should NOT be promoted"
        );
        assert_eq!(mir.constants.len(), 0);
    }
}
