//! Type system for fwgsl.
//!
//! Provides type representations, substitution, unification, and
//! Hindley-Milner type inference infrastructure for the fwgsl compiler.

use fwgsl_diagnostics::{Diagnostic, DiagnosticSink, Label};
use fwgsl_span::Span;
use std::collections::HashMap;
use std::fmt;

/// Unique type variable identifier.
pub type TyVarId = u32;

/// Type representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    /// Type variable (for inference).
    Var(TyVarId),
    /// Named type constructor: I32, F32, Bool, Vec, Mat, etc.
    Con(String),
    /// Type application: `Vec 3 F32` => App(App(Con("Vec"), Nat(3)), Con("F32"))
    App(Box<Ty>, Box<Ty>),
    /// Function arrow: a -> b
    Arrow(Box<Ty>, Box<Ty>),
    /// Universal quantification: forall a. ty
    Forall(Vec<String>, Box<Ty>),
    /// Nat literal for dependent dimensions (2, 3, 4).
    Nat(u64),
    /// Error type (produced during error recovery).
    Error,
}

impl Ty {
    pub fn i32() -> Ty {
        Ty::Con("I32".into())
    }
    pub fn f32() -> Ty {
        Ty::Con("F32".into())
    }
    pub fn u32() -> Ty {
        Ty::Con("U32".into())
    }
    pub fn bool() -> Ty {
        Ty::Con("Bool".into())
    }
    pub fn unit() -> Ty {
        Ty::Con("()".into())
    }

    pub fn arrow(from: Ty, to: Ty) -> Ty {
        Ty::Arrow(Box::new(from), Box::new(to))
    }

    pub fn app(f: Ty, arg: Ty) -> Ty {
        Ty::App(Box::new(f), Box::new(arg))
    }

    /// Check if this type contains a given type variable.
    pub fn contains_var(&self, var: TyVarId) -> bool {
        match self {
            Ty::Var(v) => *v == var,
            Ty::Con(_) | Ty::Nat(_) | Ty::Error => false,
            Ty::App(f, a) => f.contains_var(var) || a.contains_var(var),
            Ty::Arrow(a, b) => a.contains_var(var) || b.contains_var(var),
            Ty::Forall(_, body) => body.contains_var(var),
        }
    }

    /// Apply a substitution.
    pub fn apply_subst(&self, subst: &Substitution) -> Ty {
        match self {
            Ty::Var(v) => {
                if let Some(ty) = subst.lookup(*v) {
                    ty.clone()
                } else {
                    self.clone()
                }
            }
            Ty::Con(_) | Ty::Nat(_) | Ty::Error => self.clone(),
            Ty::App(f, a) => Ty::App(
                Box::new(f.apply_subst(subst)),
                Box::new(a.apply_subst(subst)),
            ),
            Ty::Arrow(a, b) => Ty::Arrow(
                Box::new(a.apply_subst(subst)),
                Box::new(b.apply_subst(subst)),
            ),
            Ty::Forall(vars, body) => Ty::Forall(vars.clone(), Box::new(body.apply_subst(subst))),
        }
    }

    /// Collect free type variables.
    pub fn free_vars(&self) -> Vec<TyVarId> {
        match self {
            Ty::Var(v) => vec![*v],
            Ty::Con(_) | Ty::Nat(_) | Ty::Error => vec![],
            Ty::App(f, a) => {
                let mut vs = f.free_vars();
                vs.extend(a.free_vars());
                vs.sort();
                vs.dedup();
                vs
            }
            Ty::Arrow(a, b) => {
                let mut vs = a.free_vars();
                vs.extend(b.free_vars());
                vs.sort();
                vs.dedup();
                vs
            }
            Ty::Forall(_, body) => body.free_vars(),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Var(v) => write!(f, "t{}", v),
            Ty::Con(name) => write!(f, "{}", name),
            Ty::App(func, arg) => write!(f, "({} {})", func, arg),
            Ty::Arrow(from, to) => write!(f, "({} -> {})", from, to),
            Ty::Forall(vars, body) => {
                write!(f, "forall")?;
                for v in vars {
                    write!(f, " {}", v)?;
                }
                write!(f, ". {}", body)
            }
            Ty::Nat(n) => write!(f, "{}", n),
            Ty::Error => write!(f, "<error>"),
        }
    }
}

/// Substitution: mapping from type variables to types.
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    map: HashMap<TyVarId, Ty>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, var: TyVarId, ty: Ty) {
        self.map.insert(var, ty);
    }

    pub fn lookup(&self, var: TyVarId) -> Option<&Ty> {
        self.map.get(&var)
    }

    /// Compose two substitutions: apply self first, then other.
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();
        for (&var, ty) in &self.map {
            result.insert(var, ty.apply_subst(other));
        }
        for (&var, ty) in &other.map {
            result.map.entry(var).or_insert_with(|| ty.clone());
        }
        result
    }
}

/// Type scheme (polymorphic type).
#[derive(Debug, Clone)]
pub struct Scheme {
    pub vars: Vec<TyVarId>,
    pub ty: Ty,
}

impl Scheme {
    pub fn mono(ty: Ty) -> Self {
        Scheme { vars: vec![], ty }
    }

    pub fn poly(vars: Vec<TyVarId>, ty: Ty) -> Self {
        Scheme { vars, ty }
    }
}

/// Type environment.
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    bindings: HashMap<String, Scheme>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: String, scheme: Scheme) {
        self.bindings.insert(name, scheme);
    }

    pub fn lookup(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name)
    }

    pub fn free_vars(&self) -> Vec<TyVarId> {
        let mut vars = Vec::new();
        for scheme in self.bindings.values() {
            for v in scheme.ty.free_vars() {
                if !scheme.vars.contains(&v) {
                    vars.push(v);
                }
            }
        }
        vars.sort();
        vars.dedup();
        vars
    }
}

/// Type inference engine.
pub struct InferEngine {
    next_var: TyVarId,
    pub subst: Substitution,
    pub diagnostics: DiagnosticSink,
}

impl Default for InferEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl InferEngine {
    pub fn new() -> Self {
        Self {
            next_var: 0,
            subst: Substitution::new(),
            diagnostics: DiagnosticSink::new(),
        }
    }

    /// Generate a fresh type variable.
    pub fn fresh_var(&mut self) -> Ty {
        let var = self.next_var;
        self.next_var += 1;
        Ty::Var(var)
    }

    /// Unify two types.
    pub fn unify(&mut self, a: &Ty, b: &Ty, span: Span) {
        let a = a.apply_subst(&self.subst);
        let b = b.apply_subst(&self.subst);

        match (&a, &b) {
            (Ty::Error, _) | (_, Ty::Error) => {}
            (Ty::Var(v1), Ty::Var(v2)) if v1 == v2 => {}
            (Ty::Var(v), ty) | (ty, Ty::Var(v)) => {
                if ty.contains_var(*v) {
                    self.diagnostics.push(
                        Diagnostic::error(format!("Infinite type: t{} ~ {}", v, ty))
                            .with_label(Label::primary(span, "occurs here")),
                    );
                } else {
                    self.subst.insert(*v, ty.clone());
                }
            }
            (Ty::Con(a), Ty::Con(b)) if a == b => {}
            (Ty::Nat(a), Ty::Nat(b)) if a == b => {}
            (Ty::Arrow(a1, a2), Ty::Arrow(b1, b2)) => {
                self.unify(a1, b1, span);
                self.unify(a2, b2, span);
            }
            (Ty::App(f1, a1), Ty::App(f2, a2)) => {
                self.unify(f1, f2, span);
                self.unify(a1, a2, span);
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error(format!("Type mismatch: {} vs {}", a, b))
                        .with_label(Label::primary(span, "type mismatch")),
                );
            }
        }
    }

    /// Instantiate a type scheme with fresh variables.
    pub fn instantiate(&mut self, scheme: &Scheme) -> Ty {
        let mut subst = Substitution::new();
        for &var in &scheme.vars {
            subst.insert(var, self.fresh_var());
        }
        scheme.ty.apply_subst(&subst)
    }

    /// Generalize a type over variables not free in the environment.
    pub fn generalize(&self, env: &TypeEnv, ty: &Ty) -> Scheme {
        let ty = ty.apply_subst(&self.subst);
        let env_vars = env.free_vars();
        let ty_vars = ty.free_vars();
        let gen_vars: Vec<TyVarId> = ty_vars
            .into_iter()
            .filter(|v| !env_vars.contains(v))
            .collect();
        Scheme::poly(gen_vars, ty)
    }

    /// Finalize a type by applying all accumulated substitutions.
    pub fn finalize(&self, ty: &Ty) -> Ty {
        ty.apply_subst(&self.subst)
    }
}

/// Constructor info used during type checking.
#[derive(Debug, Clone)]
pub struct ConstructorInfo {
    pub type_name: String,
    pub tag: u32,
    pub fields: ConstructorFields,
    pub result_ty: Ty,
}

#[derive(Debug, Clone)]
pub enum ConstructorFields {
    Positional(Vec<Ty>),
    Record(Vec<(String, Ty)>),
    Empty,
}

/// WGSL-compatible types (used in MIR and codegen).
#[derive(Debug, Clone, PartialEq)]
pub enum WgslType {
    I32,
    U32,
    F32,
    Bool,
    /// vec2<f32>, vec3<f32>, vec4<f32>
    Vec(u8, Box<WgslType>),
    /// mat2x2<f32>, etc.
    Mat(u8, u8, Box<WgslType>),
    Struct(String),
    /// Fixed-size or runtime array.
    Array(Box<WgslType>, Option<u32>),
    Unit,
}

impl fmt::Display for WgslType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgslType::I32 => write!(f, "i32"),
            WgslType::U32 => write!(f, "u32"),
            WgslType::F32 => write!(f, "f32"),
            WgslType::Bool => write!(f, "bool"),
            WgslType::Vec(n, elem) => write!(f, "vec{}<{}>", n, elem),
            WgslType::Mat(r, c, elem) => write!(f, "mat{}x{}<{}>", r, c, elem),
            WgslType::Struct(name) => write!(f, "{}", name),
            WgslType::Array(elem, Some(n)) => write!(f, "array<{}, {}>", elem, n),
            WgslType::Array(elem, None) => write!(f, "array<{}>", elem),
            WgslType::Unit => write!(f, "void"),
        }
    }
}

/// Convert high-level Ty to WgslType (after monomorphization).
pub fn ty_to_wgsl(ty: &Ty) -> Result<WgslType, String> {
    match ty {
        Ty::Con(name) => match name.as_str() {
            "I32" => Ok(WgslType::I32),
            "U32" => Ok(WgslType::U32),
            "F32" => Ok(WgslType::F32),
            "Bool" => Ok(WgslType::Bool),
            "()" => Ok(WgslType::Unit),
            other => Ok(WgslType::Struct(other.to_string())),
        },
        Ty::App(f, arg) => {
            // Handle Vec N Scalar and Mat N M Scalar
            match f.as_ref() {
                Ty::App(ff, n) => match (ff.as_ref(), n.as_ref()) {
                    // Mat N M Scalar
                    (Ty::App(fff, nn), Ty::Nat(m)) => {
                        if let (Ty::Con(name), Ty::Nat(n)) = (fff.as_ref(), nn.as_ref()) {
                            if name == "Mat" {
                                let scalar = ty_to_wgsl(arg)?;
                                return Ok(WgslType::Mat(*n as u8, *m as u8, Box::new(scalar)));
                            }
                        }
                        Err(format!("Cannot convert to WGSL: {}", ty))
                    }
                    // Vec N Scalar
                    (Ty::Con(name), Ty::Nat(n)) if name == "Vec" => {
                        let scalar = ty_to_wgsl(arg)?;
                        Ok(WgslType::Vec(*n as u8, Box::new(scalar)))
                    }
                    _ => Err(format!("Cannot convert to WGSL: {}", ty)),
                },
                Ty::Con(name) if name == "Vec" => {
                    // Vec applied to one arg (the dim), need another arg
                    Err(format!("Partially applied Vec: {}", ty))
                }
                _ => Err(format!("Cannot convert to WGSL: {}", ty)),
            }
        }
        Ty::Arrow(_, _) => Err("Function types cannot be represented in WGSL".into()),
        Ty::Var(v) => Err(format!("Unresolved type variable t{}", v)),
        _ => Err(format!("Cannot convert to WGSL: {}", ty)),
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
    fn test_ty_display_con() {
        assert_eq!(format!("{}", Ty::i32()), "I32");
        assert_eq!(format!("{}", Ty::f32()), "F32");
        assert_eq!(format!("{}", Ty::bool()), "Bool");
    }

    #[test]
    fn test_ty_display_arrow() {
        let ty = Ty::arrow(Ty::i32(), Ty::bool());
        assert_eq!(format!("{}", ty), "(I32 -> Bool)");
    }

    #[test]
    fn test_ty_display_app() {
        let ty = Ty::app(Ty::Con("Maybe".into()), Ty::i32());
        assert_eq!(format!("{}", ty), "(Maybe I32)");
    }

    #[test]
    fn test_ty_display_forall() {
        let ty = Ty::Forall(vec!["a".into()], Box::new(Ty::Var(0)));
        assert_eq!(format!("{}", ty), "forall a. t0");
    }

    #[test]
    fn test_ty_display_nat() {
        assert_eq!(format!("{}", Ty::Nat(3)), "3");
    }

    #[test]
    fn test_ty_display_var() {
        assert_eq!(format!("{}", Ty::Var(42)), "t42");
    }

    #[test]
    fn test_ty_display_error() {
        assert_eq!(format!("{}", Ty::Error), "<error>");
    }

    #[test]
    fn test_contains_var() {
        let ty = Ty::arrow(Ty::Var(0), Ty::i32());
        assert!(ty.contains_var(0));
        assert!(!ty.contains_var(1));
    }

    #[test]
    fn test_free_vars() {
        let ty = Ty::arrow(Ty::Var(0), Ty::arrow(Ty::Var(1), Ty::Var(0)));
        let mut fv = ty.free_vars();
        fv.sort();
        assert_eq!(fv, vec![0, 1]);
    }

    #[test]
    fn test_substitution_basic() {
        let mut subst = Substitution::new();
        subst.insert(0, Ty::i32());
        let ty = Ty::arrow(Ty::Var(0), Ty::Var(1));
        let result = ty.apply_subst(&subst);
        assert_eq!(result, Ty::arrow(Ty::i32(), Ty::Var(1)));
    }

    #[test]
    fn test_substitution_compose() {
        let mut s1 = Substitution::new();
        s1.insert(0, Ty::Var(1));

        let mut s2 = Substitution::new();
        s2.insert(1, Ty::i32());

        let composed = s1.compose(&s2);
        let ty = Ty::Var(0);
        let result = ty.apply_subst(&composed);
        assert_eq!(result, Ty::i32());
    }

    #[test]
    fn test_unify_same_con() {
        let mut engine = InferEngine::new();
        engine.unify(&Ty::i32(), &Ty::i32(), span());
        assert!(!engine.diagnostics.has_errors());
    }

    #[test]
    fn test_unify_var_with_con() {
        let mut engine = InferEngine::new();
        let var = engine.fresh_var();
        engine.unify(&var, &Ty::i32(), span());
        assert!(!engine.diagnostics.has_errors());
        let result = engine.finalize(&var);
        assert_eq!(result, Ty::i32());
    }

    #[test]
    fn test_unify_arrows() {
        let mut engine = InferEngine::new();
        let a = engine.fresh_var();
        let b = engine.fresh_var();
        let arrow1 = Ty::arrow(a.clone(), b.clone());
        let arrow2 = Ty::arrow(Ty::i32(), Ty::bool());
        engine.unify(&arrow1, &arrow2, span());
        assert!(!engine.diagnostics.has_errors());
        assert_eq!(engine.finalize(&a), Ty::i32());
        assert_eq!(engine.finalize(&b), Ty::bool());
    }

    #[test]
    fn test_unify_mismatch() {
        let mut engine = InferEngine::new();
        engine.unify(&Ty::i32(), &Ty::bool(), span());
        assert!(engine.diagnostics.has_errors());
    }

    #[test]
    fn test_unify_occurs_check() {
        let mut engine = InferEngine::new();
        let a = engine.fresh_var();
        let circular = Ty::arrow(a.clone(), Ty::i32());
        engine.unify(&a, &circular, span());
        assert!(engine.diagnostics.has_errors());
    }

    #[test]
    fn test_instantiate() {
        let mut engine = InferEngine::new();
        let scheme = Scheme::poly(vec![0], Ty::arrow(Ty::Var(0), Ty::Var(0)));
        let ty1 = engine.instantiate(&scheme);
        let ty2 = engine.instantiate(&scheme);
        // Each instantiation should produce different variables
        assert_ne!(ty1, ty2);
    }

    #[test]
    fn test_generalize() {
        let engine = InferEngine::new();
        let env = TypeEnv::new();
        let ty = Ty::arrow(Ty::Var(0), Ty::Var(0));
        let scheme = engine.generalize(&env, &ty);
        assert_eq!(scheme.vars, vec![0]);
    }

    #[test]
    fn test_generalize_with_env_constraint() {
        let engine = InferEngine::new();
        let mut env = TypeEnv::new();
        env.insert("x".to_string(), Scheme::mono(Ty::Var(0)));
        let ty = Ty::arrow(Ty::Var(0), Ty::Var(1));
        let scheme = engine.generalize(&env, &ty);
        // Only Var(1) should be generalized, Var(0) is free in env
        assert_eq!(scheme.vars, vec![1]);
    }

    #[test]
    fn test_ty_to_wgsl_scalars() {
        assert_eq!(ty_to_wgsl(&Ty::i32()), Ok(WgslType::I32));
        assert_eq!(ty_to_wgsl(&Ty::f32()), Ok(WgslType::F32));
        assert_eq!(ty_to_wgsl(&Ty::u32()), Ok(WgslType::U32));
        assert_eq!(ty_to_wgsl(&Ty::bool()), Ok(WgslType::Bool));
        assert_eq!(ty_to_wgsl(&Ty::unit()), Ok(WgslType::Unit));
    }

    #[test]
    fn test_ty_to_wgsl_vec() {
        // Vec 3 F32
        let ty = Ty::app(Ty::app(Ty::Con("Vec".into()), Ty::Nat(3)), Ty::f32());
        assert_eq!(
            ty_to_wgsl(&ty),
            Ok(WgslType::Vec(3, Box::new(WgslType::F32)))
        );
    }

    #[test]
    fn test_ty_to_wgsl_mat() {
        // Mat 4 4 F32
        let ty = Ty::app(
            Ty::app(Ty::app(Ty::Con("Mat".into()), Ty::Nat(4)), Ty::Nat(4)),
            Ty::f32(),
        );
        assert_eq!(
            ty_to_wgsl(&ty),
            Ok(WgslType::Mat(4, 4, Box::new(WgslType::F32)))
        );
    }

    #[test]
    fn test_ty_to_wgsl_function_error() {
        let ty = Ty::arrow(Ty::i32(), Ty::i32());
        assert!(ty_to_wgsl(&ty).is_err());
    }

    #[test]
    fn test_ty_to_wgsl_var_error() {
        assert!(ty_to_wgsl(&Ty::Var(0)).is_err());
    }

    #[test]
    fn test_wgsl_type_display() {
        assert_eq!(format!("{}", WgslType::I32), "i32");
        assert_eq!(format!("{}", WgslType::F32), "f32");
        assert_eq!(
            format!("{}", WgslType::Vec(3, Box::new(WgslType::F32))),
            "vec3<f32>"
        );
        assert_eq!(
            format!("{}", WgslType::Mat(4, 4, Box::new(WgslType::F32))),
            "mat4x4<f32>"
        );
        assert_eq!(
            format!("{}", WgslType::Array(Box::new(WgslType::F32), Some(16))),
            "array<f32, 16>"
        );
        assert_eq!(
            format!("{}", WgslType::Array(Box::new(WgslType::F32), None)),
            "array<f32>"
        );
        assert_eq!(format!("{}", WgslType::Unit), "void");
        assert_eq!(
            format!("{}", WgslType::Struct("MyStruct".into())),
            "MyStruct"
        );
    }

    #[test]
    fn test_scheme_mono() {
        let scheme = Scheme::mono(Ty::i32());
        assert!(scheme.vars.is_empty());
        assert_eq!(scheme.ty, Ty::i32());
    }

    #[test]
    fn test_type_env_basic() {
        let mut env = TypeEnv::new();
        env.insert("x".to_string(), Scheme::mono(Ty::i32()));
        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_none());
    }
}
