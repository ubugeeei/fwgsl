///|
/// High-level Intermediate Representation (HIR) for fwgsl.
///
/// HIR is the desugared, type-annotated intermediate representation. It is
/// produced from the parser AST after semantic analysis (type inference and
/// name resolution). Every expression node carries its resolved type.

///|
pub(all) struct HirProgram {
  functions : Array[HirFunction]
  data_types : Array[HirDataType]
  entry_points : Array[HirEntryPoint]
}

///|
pub(all) struct HirFunction {
  name : String
  params : Array[(String, @typechecker.Ty)]
  return_ty : @typechecker.Ty
  body : HirExpr
  span : @span.Span
}

///|
pub(all) struct HirEntryPoint {
  name : String
  attributes : Array[HirAttribute]
  params : Array[(String, @typechecker.Ty)]
  return_ty : @typechecker.Ty
  body : HirExpr
  span : @span.Span
}

///|
pub(all) struct HirAttribute {
  name : String
  args : Array[String]
}

///|
pub(all) struct HirDataType {
  name : String
  constructors : Array[HirConstructor]
}

///|
pub(all) struct HirConstructor {
  name : String
  tag : UInt
  fields : Array[(String, @typechecker.Ty)]
}

///|
pub(all) enum HirLit {
  Int(Int64)
  Float(Double)
  Bool(Bool)
}

///|
pub(all) enum BinOp {
  Add
  Sub
  Mul
  Div
  Mod
  Eq
  Ne
  Lt
  Gt
  Le
  Ge
  And
  Or
} derive(Eq)

///|
pub fn BinOp::parse(s : String) -> BinOp? {
  match s {
    "+" => Some(Add)
    "-" => Some(Sub)
    "*" => Some(Mul)
    "/" => Some(Div)
    "%" => Some(Mod)
    "==" => Some(Eq)
    "/=" => Some(Ne)
    "<" => Some(Lt)
    ">" => Some(Gt)
    "<=" => Some(Le)
    ">=" => Some(Ge)
    "&&" => Some(And)
    "||" => Some(Or)
    _ => None
  }
}

///|
pub fn BinOp::to_wgsl_str(self : BinOp) -> String {
  match self {
    Add => "+"
    Sub => "-"
    Mul => "*"
    Div => "/"
    Mod => "%"
    Eq => "=="
    Ne => "!="
    Lt => "<"
    Gt => ">"
    Le => "<="
    Ge => ">="
    And => "&&"
    Or => "||"
  }
}

///|
pub(all) enum HirPattern {
  Wild
  Var(String, @typechecker.Ty)
  Constructor(String, UInt, Array[HirPattern])
  PatLit(HirLit)
}

///|
pub(all) struct HirCaseArm {
  pattern : HirPattern
  body : HirExpr
}

///|
pub(all) enum HirExpr {
  Lit(HirLit, @typechecker.Ty, @span.Span)
  Var(String, @typechecker.Ty, @span.Span)
  App(HirExpr, HirExpr, @typechecker.Ty, @span.Span)
  Let(Array[(String, HirExpr)], HirExpr, @typechecker.Ty, @span.Span)
  Case(HirExpr, Array[HirCaseArm], @typechecker.Ty, @span.Span)
  If(HirExpr, HirExpr, HirExpr, @typechecker.Ty, @span.Span)
  BinOpE(BinOp, HirExpr, HirExpr, @typechecker.Ty, @span.Span)
  ConstructorCall(String, UInt, Array[HirExpr], @typechecker.Ty, @span.Span)
  FieldAccess(HirExpr, String, @typechecker.Ty, @span.Span)
}

///|
pub fn HirExpr::ty(self : HirExpr) -> @typechecker.Ty {
  match self {
    Lit(_, ty, _) => ty
    Var(_, ty, _) => ty
    App(_, _, ty, _) => ty
    Let(_, _, ty, _) => ty
    Case(_, _, ty, _) => ty
    If(_, _, _, ty, _) => ty
    BinOpE(_, _, _, ty, _) => ty
    ConstructorCall(_, _, _, ty, _) => ty
    FieldAccess(_, _, ty, _) => ty
  }
}

///|
pub impl Show for BinOp with output(self, logger) {
  logger.write_string(self.to_wgsl_str())
}
