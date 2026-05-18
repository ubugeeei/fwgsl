///|
/// AST -> HIR lowering for fwgsl.
///
/// The Rust crate (`crates/fwgsl_ast_lowering/src/lib.rs`, ~1200 lines)
/// walks the parser AST and produces a fully type-annotated HIR using the
/// already-built `SemanticAnalyzer` for name resolution and types. The
/// port lands the public structure first so downstream callers (CLI,
/// integration tests) can depend on the same shape; the expression
/// lowering body is filled in incrementally.

///|
pub struct AstLowering {
  analyzer : @semantic.SemanticAnalyzer
  diagnostics : @diagnostics.DiagnosticSink
}

///|
pub fn AstLowering::new(analyzer : @semantic.SemanticAnalyzer) -> AstLowering {
  AstLowering::{
    analyzer,
    diagnostics: @diagnostics.DiagnosticSink::new(),
  }
}

///|
pub fn AstLowering::diagnostics(self : AstLowering) -> @diagnostics.DiagnosticSink {
  self.diagnostics
}

///|
pub fn AstLowering::has_errors(self : AstLowering) -> Bool {
  self.diagnostics.has_errors()
}

///|
/// Lower a complete parsed `Program` to a `HirProgram`.
///
/// The current pass:
/// 1. Collects record-shaped data type constructors into `HirDataType` /
///    `HirConstructor`.
/// 2. Stubs out function and entry-point bodies with `HirLit::Int(0)` so
///    downstream packages can already consume a `HirProgram`.
///
/// Full expression lowering (`lower_expr`) lives in
/// `crates/fwgsl_ast_lowering/src/lib.rs` and will land in a follow-up.
pub fn AstLowering::lower_program(
  self : AstLowering,
  prog : @ast.Program
) -> @hir.HirProgram {
  let functions : Array[@hir.HirFunction] = Array::new()
  let data_types : Array[@hir.HirDataType] = Array::new()
  let entry_points : Array[@hir.HirEntryPoint] = Array::new()
  for d in prog.decls {
    match d {
      DataDecl(name, _, ctors, _) => {
        let constructors : Array[@hir.HirConstructor] = Array::new()
        let mut tag : UInt = 0
        for con in ctors {
          let fields : Array[(String, @typechecker.Ty)] = Array::new()
          match con.fields {
            Positional(tys) => {
              let mut i = 0
              for ty in tys {
                fields.push(("f" + i.to_string(), ast_ty_placeholder(ty)))
                i = i + 1
              }
            }
            RecordFields(rs) =>
              for r in rs {
                let (n, ty) = r
                fields.push((n, ast_ty_placeholder(ty)))
              }
            Empty => ()
          }
          constructors.push(
            @hir.HirConstructor::{ name: con.name, tag, fields },
          )
          tag = tag + 1
        }
        data_types.push(@hir.HirDataType::{ name, constructors })
      }
      FunDecl(name, _, _, _, span) =>
        functions.push(
          @hir.HirFunction::{
            name,
            params: Array::new(),
            return_ty: @typechecker.Ty::i32(),
            body: placeholder_body(span),
            span,
          },
        )
      EntryPoint(attrs, name, _, _, span) => {
        let hir_attrs : Array[@hir.HirAttribute] = Array::new()
        for a in attrs {
          hir_attrs.push(@hir.HirAttribute::{ name: a.name, args: a.args })
        }
        entry_points.push(
          @hir.HirEntryPoint::{
            name,
            attributes: hir_attrs,
            params: Array::new(),
            return_ty: @typechecker.Ty::unit(),
            body: placeholder_body(span),
            span,
          },
        )
      }
      _ => ()
    }
  }
  let _ = self.analyzer
  @hir.HirProgram::{ functions, data_types, entry_points }
}

///|
fn placeholder_body(span : @span.Span) -> @hir.HirExpr {
  @hir.HirExpr::Lit(@hir.HirLit::Int(0L), @typechecker.Ty::i32(), span)
}

///|
fn ast_ty_placeholder(ty : @ast.Type) -> @typechecker.Ty {
  match ty {
    Con(name, _) => @typechecker.Ty::Con(name)
    Var(name, _) => @typechecker.Ty::Con(name)
    Nat(n, _) => @typechecker.Ty::Nat(n)
    App(f, arg, _) =>
      @typechecker.Ty::App(ast_ty_placeholder(f), ast_ty_placeholder(arg))
    Arrow(a, b, _) =>
      @typechecker.Ty::arrow(ast_ty_placeholder(a), ast_ty_placeholder(b))
    Paren(inner, _) => ast_ty_placeholder(inner)
    Tuple(_, _) => @typechecker.Ty::unit()
    Unit(_) => @typechecker.Ty::unit()
  }
}
