///|
/// Abstract syntax tree types for fwgsl.
///
/// In the Rust workspace these definitions live inside `fwgsl_parser`
/// (with `fwgsl_ast` left as a placeholder); the MoonBit port moves them
/// here so the parser package can stay focused on parsing.

///|
/// A literal value.
pub(all) enum Lit {
  Int(Int64)
  Float(Double)
  StringLit(String)
  Char(Char)
} derive(Show)

///|
/// A type expression.
pub(all) enum Type {
  Con(String, @span.Span)
  Var(String, @span.Span)
  Nat(UInt64, @span.Span)
  App(Type, Type, @span.Span)
  Arrow(Type, Type, @span.Span)
  Paren(Type, @span.Span)
  Tuple(Array[Type], @span.Span)
  Unit(@span.Span)
} derive(Show)

///|
/// Return the source span of a type expression.
pub fn Type::span(self : Type) -> @span.Span {
  match self {
    Con(_, s) => s
    Var(_, s) => s
    Nat(_, s) => s
    App(_, _, s) => s
    Arrow(_, _, s) => s
    Paren(_, s) => s
    Tuple(_, s) => s
    Unit(s) => s
  }
}

///|
/// A pattern.
pub(all) enum Pat {
  Wild(@span.Span)
  Var(String, @span.Span)
  Con(String, Array[Pat], @span.Span)
  Lit(Lit, @span.Span)
  Paren(Pat, @span.Span)
  Tuple(Array[Pat], @span.Span)
  Record(String, Array[(String, Pat?)], @span.Span)
  As(String, Pat, @span.Span)
} derive(Show)

///|
/// Return the source span of a pattern.
pub fn Pat::span(self : Pat) -> @span.Span {
  match self {
    Wild(s) => s
    Var(_, s) => s
    Con(_, _, s) => s
    Lit(_, s) => s
    Paren(_, s) => s
    Tuple(_, s) => s
    Record(_, _, s) => s
    As(_, _, s) => s
  }
}

///|
/// A single statement inside a `do` block.
pub(all) enum DoStmt {
  Bind(String, Expr, @span.Span)
  ExprStmt(Expr, @span.Span)
  LetStmt(String, Expr, @span.Span)
} derive(Show)

///|
/// An expression.
pub(all) enum Expr {
  Lit(Lit, @span.Span)
  Var(String, @span.Span)
  Con(String, @span.Span)
  App(Expr, Expr, @span.Span)
  Infix(Expr, String, Expr, @span.Span)
  Lambda(Array[Pat], Expr, @span.Span)
  Let(Array[(String, Expr)], Expr, @span.Span)
  Case(Expr, Array[(Pat, Expr)], @span.Span)
  If(Expr, Expr, Expr, @span.Span)
  Paren(Expr, @span.Span)
  Tuple(Array[Expr], @span.Span)
  Record(Array[(String, Expr)], @span.Span)
  FieldAccess(Expr, String, @span.Span)
  OpSection(String, @span.Span)
  Neg(Expr, @span.Span)
  Do(Array[DoStmt], @span.Span)
  /// Vector literal `[a, b, c]` — desugared to a `vecN` constructor call.
  VecLit(Array[Expr], @span.Span)
} derive(Show)

///|
/// Return the source span of an expression.
pub fn Expr::span(self : Expr) -> @span.Span {
  match self {
    Lit(_, s) => s
    Var(_, s) => s
    Con(_, s) => s
    App(_, _, s) => s
    Infix(_, _, _, s) => s
    Lambda(_, _, s) => s
    Let(_, _, s) => s
    Case(_, _, s) => s
    If(_, _, _, s) => s
    Paren(_, s) => s
    Tuple(_, s) => s
    Record(_, s) => s
    FieldAccess(_, _, s) => s
    OpSection(_, s) => s
    Neg(_, s) => s
    Do(_, s) => s
    VecLit(_, s) => s
  }
}

///|
/// The body of a constructor declaration.
pub(all) enum ConFields {
  Positional(Array[Type])
  RecordFields(Array[(String, Type)])
  Empty
} derive(Show)

///|
/// A single constructor inside a data declaration.
pub(all) struct ConDecl {
  name : String
  fields : ConFields
  span : @span.Span
} derive(Show)

///|
/// Create a new constructor declaration.
pub fn ConDecl::new(
  name : String,
  fields : ConFields,
  span : @span.Span
) -> ConDecl {
  ConDecl::{ name, fields, span }
}

///|
/// An attribute attached to an entry-point declaration.
pub(all) struct Attribute {
  name : String
  args : Array[String]
  span : @span.Span
} derive(Show)

///|
/// Create a new attribute.
pub fn Attribute::new(
  name : String,
  args : Array[String],
  span : @span.Span
) -> Attribute {
  Attribute::{ name, args, span }
}

///|
/// A top-level declaration.
pub(all) enum Decl {
  TypeSig(String, Type, @span.Span)
  FunDecl(
    String,
    Array[Pat],
    Expr,
    Array[(String, Expr)],
    @span.Span
  )
  DataDecl(String, Array[String], Array[ConDecl], @span.Span)
  EntryPoint(
    Array[Attribute],
    String,
    Array[Pat],
    Expr,
    @span.Span
  )
  TypeAlias(String, Array[String], Type, @span.Span)
} derive(Show)

///|
/// Return the source span of a declaration.
pub fn Decl::span(self : Decl) -> @span.Span {
  match self {
    TypeSig(_, _, s) => s
    FunDecl(_, _, _, _, s) => s
    DataDecl(_, _, _, s) => s
    EntryPoint(_, _, _, _, s) => s
    TypeAlias(_, _, _, s) => s
  }
}

///|
/// A parsed fwgsl program.
pub(all) struct Program {
  decls : Array[Decl]
} derive(Show)

///|
pub fn Program::new(decls : Array[Decl]) -> Program {
  Program::{ decls, }
}

///|
pub fn Program::empty() -> Program {
  Program::{ decls: Array::new() }
}
