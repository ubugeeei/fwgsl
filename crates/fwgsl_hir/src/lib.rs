//! High-level Intermediate Representation (HIR) for fwgsl.
//!
//! HIR is the desugared, type-annotated intermediate representation.
//! It is produced from the parser AST after semantic analysis (type inference
//! and name resolution). Every expression node carries its resolved type.

use fwgsl_span::Span;
use fwgsl_typechecker::Ty;

/// HIR Program (desugared, type-annotated).
#[derive(Debug)]
pub struct HirProgram {
    pub functions: Vec<HirFunction>,
    pub data_types: Vec<HirDataType>,
    pub entry_points: Vec<HirEntryPoint>,
}

#[derive(Debug)]
pub struct HirFunction {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub return_ty: Ty,
    pub body: HirExpr,
    pub span: Span,
}

#[derive(Debug)]
pub struct HirEntryPoint {
    pub name: String,
    pub attributes: Vec<HirAttribute>,
    pub params: Vec<(String, Ty)>,
    pub return_ty: Ty,
    pub body: HirExpr,
    pub span: Span,
}

#[derive(Debug)]
pub struct HirAttribute {
    pub name: String,
    pub args: Vec<String>,
}

#[derive(Debug)]
pub struct HirDataType {
    pub name: String,
    pub constructors: Vec<HirConstructor>,
}

#[derive(Debug)]
pub struct HirConstructor {
    pub name: String,
    pub tag: u32,
    pub fields: Vec<(String, Ty)>,
}

#[derive(Debug)]
pub enum HirExpr {
    Lit(HirLit, Ty, Span),
    Var(String, Ty, Span),
    App(Box<HirExpr>, Box<HirExpr>, Ty, Span),
    Let(Vec<(String, HirExpr)>, Box<HirExpr>, Ty, Span),
    Case(Box<HirExpr>, Vec<HirCaseArm>, Ty, Span),
    If(Box<HirExpr>, Box<HirExpr>, Box<HirExpr>, Ty, Span),
    BinOp(BinOp, Box<HirExpr>, Box<HirExpr>, Ty, Span),
    /// Constructor call: name, tag, arguments, result type, span.
    ConstructorCall(String, u32, Vec<HirExpr>, Ty, Span),
    FieldAccess(Box<HirExpr>, String, Ty, Span),
}

impl HirExpr {
    pub fn ty(&self) -> &Ty {
        match self {
            HirExpr::Lit(_, ty, _) => ty,
            HirExpr::Var(_, ty, _) => ty,
            HirExpr::App(_, _, ty, _) => ty,
            HirExpr::Let(_, _, ty, _) => ty,
            HirExpr::Case(_, _, ty, _) => ty,
            HirExpr::If(_, _, _, ty, _) => ty,
            HirExpr::BinOp(_, _, _, ty, _) => ty,
            HirExpr::ConstructorCall(_, _, _, ty, _) => ty,
            HirExpr::FieldAccess(_, _, ty, _) => ty,
        }
    }
}

#[derive(Debug)]
pub struct HirCaseArm {
    pub pattern: HirPattern,
    pub body: HirExpr,
}

#[derive(Debug)]
pub enum HirPattern {
    Wild,
    Var(String, Ty),
    Constructor(String, u32, Vec<HirPattern>),
    Lit(HirLit),
}

#[derive(Debug, Clone)]
pub enum HirLit {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn parse(s: &str) -> Option<BinOp> {
        match s {
            "+" => Some(BinOp::Add),
            "-" => Some(BinOp::Sub),
            "*" => Some(BinOp::Mul),
            "/" => Some(BinOp::Div),
            "%" => Some(BinOp::Mod),
            "==" => Some(BinOp::Eq),
            "/=" => Some(BinOp::Ne),
            "<" => Some(BinOp::Lt),
            ">" => Some(BinOp::Gt),
            "<=" => Some(BinOp::Le),
            ">=" => Some(BinOp::Ge),
            "&&" => Some(BinOp::And),
            "||" => Some(BinOp::Or),
            _ => None,
        }
    }

    pub fn to_wgsl_str(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fwgsl_span::Span;
    use fwgsl_typechecker::Ty;

    #[test]
    fn test_hir_expr_ty() {
        let expr = HirExpr::Lit(HirLit::Int(42), Ty::i32(), Span::new(0, 2));
        assert_eq!(expr.ty(), &Ty::i32());
    }

    #[test]
    fn test_binop_from_str() {
        assert!(matches!(BinOp::parse("+"), Some(BinOp::Add)));
        assert!(matches!(BinOp::parse("-"), Some(BinOp::Sub)));
        assert!(matches!(BinOp::parse("*"), Some(BinOp::Mul)));
        assert!(matches!(BinOp::parse("/"), Some(BinOp::Div)));
        assert!(matches!(BinOp::parse("%"), Some(BinOp::Mod)));
        assert!(matches!(BinOp::parse("=="), Some(BinOp::Eq)));
        assert!(matches!(BinOp::parse("/="), Some(BinOp::Ne)));
        assert!(matches!(BinOp::parse("<"), Some(BinOp::Lt)));
        assert!(matches!(BinOp::parse(">"), Some(BinOp::Gt)));
        assert!(matches!(BinOp::parse("<="), Some(BinOp::Le)));
        assert!(matches!(BinOp::parse(">="), Some(BinOp::Ge)));
        assert!(matches!(BinOp::parse("&&"), Some(BinOp::And)));
        assert!(matches!(BinOp::parse("||"), Some(BinOp::Or)));
        assert!(BinOp::parse("???").is_none());
    }

    #[test]
    fn test_binop_to_wgsl_str() {
        assert_eq!(BinOp::Add.to_wgsl_str(), "+");
        assert_eq!(BinOp::Ne.to_wgsl_str(), "!=");
    }

    #[test]
    fn test_hir_program_construction() {
        let program = HirProgram {
            functions: vec![HirFunction {
                name: "add".into(),
                params: vec![("x".into(), Ty::i32()), ("y".into(), Ty::i32())],
                return_ty: Ty::i32(),
                body: HirExpr::BinOp(
                    BinOp::Add,
                    Box::new(HirExpr::Var("x".into(), Ty::i32(), Span::new(0, 1))),
                    Box::new(HirExpr::Var("y".into(), Ty::i32(), Span::new(4, 5))),
                    Ty::i32(),
                    Span::new(0, 5),
                ),
                span: Span::new(0, 20),
            }],
            data_types: vec![],
            entry_points: vec![],
        };
        assert_eq!(program.functions.len(), 1);
        assert_eq!(program.functions[0].name, "add");
    }

    #[test]
    fn test_hir_case_arm() {
        let arm = HirCaseArm {
            pattern: HirPattern::Lit(HirLit::Int(0)),
            body: HirExpr::Lit(HirLit::Bool(true), Ty::bool(), Span::new(0, 4)),
        };
        assert!(matches!(arm.pattern, HirPattern::Lit(HirLit::Int(0))));
    }

    #[test]
    fn test_hir_if_expr() {
        let expr = HirExpr::If(
            Box::new(HirExpr::Lit(
                HirLit::Bool(true),
                Ty::bool(),
                Span::new(0, 4),
            )),
            Box::new(HirExpr::Lit(HirLit::Int(1), Ty::i32(), Span::new(10, 11))),
            Box::new(HirExpr::Lit(HirLit::Int(0), Ty::i32(), Span::new(17, 18))),
            Ty::i32(),
            Span::new(0, 18),
        );
        assert_eq!(expr.ty(), &Ty::i32());
    }
}
