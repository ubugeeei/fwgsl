///|
/// Mid-level Intermediate Representation (MIR) for fwgsl.
///
/// MIR is a lowered representation suitable for code generation. It has
/// been type-checked, monomorphised, and desugared from HIR into a form
/// that maps closely to WGSL constructs.

// ---------------------------------------------------------------------------
// Top-level program
// ---------------------------------------------------------------------------

///|
pub(all) struct MirProgram {
  structs : Array[MirStruct]
  functions : Array[MirFunction]
  entry_points : Array[MirEntryPoint]
}

// ---------------------------------------------------------------------------
// Struct definitions
// ---------------------------------------------------------------------------

///|
pub(all) struct MirStruct {
  name : String
  fields : Array[MirField]
}

///|
pub(all) struct MirField {
  name : String
  ty : MirType
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

///|
/// Concrete types that map to WGSL types.
pub(all) enum MirType {
  I32
  U32
  F32
  Bool
  MirVec(UInt, MirType)
  MirMat(UInt, UInt, MirType)
  MirStructTy(String)
  MirArray(MirType, UInt)
  Unit
} derive(Eq)

///|
pub fn MirType::display(self : MirType) -> String {
  match self {
    I32 => "i32"
    U32 => "u32"
    F32 => "f32"
    Bool => "bool"
    MirVec(n, inner) => "vec" + n.to_string() + "<" + inner.display() + ">"
    MirMat(cols, rows, inner) =>
      "mat" + cols.to_string() + "x" + rows.to_string() + "<" + inner.display() + ">"
    MirStructTy(name) => name
    MirArray(inner, len) => "array<" + inner.display() + ", " + len.to_string() + ">"
    Unit => "void"
  }
}

///|
impl Show for MirType with output(self : MirType, logger : &Logger) -> Unit {
  logger.write_string(self.display())
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

///|
pub(all) struct MirFunction {
  name : String
  params : Array[MirParam]
  return_ty : MirType
  body : Array[MirStmt]
  return_expr : MirExpr?
}

///|
pub(all) struct MirParam {
  name : String
  ty : MirType
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

///|
pub(all) struct MirEntryPoint {
  name : String
  stage : ShaderStage
  workgroup_size : (UInt, UInt, UInt)?
  params : Array[MirParam]
  builtins : Array[(String, BuiltinBinding, MirType)]
  return_ty : MirType
  body : Array[MirStmt]
  return_expr : MirExpr?
}

///|
pub(all) enum ShaderStage {
  Compute
  Vertex
  Fragment
} derive(Eq)

///|
pub fn ShaderStage::display(self : ShaderStage) -> String {
  match self {
    Compute => "compute"
    Vertex => "vertex"
    Fragment => "fragment"
  }
}

///|
pub(all) enum BuiltinBinding {
  GlobalInvocationId
  LocalInvocationId
  WorkgroupId
  NumWorkgroups
  VertexIndex
  InstanceIndex
  Position
  FrontFacing
  FragDepth
  SampleIndex
  SampleMask
} derive(Eq)

///|
pub fn BuiltinBinding::display(self : BuiltinBinding) -> String {
  match self {
    GlobalInvocationId => "global_invocation_id"
    LocalInvocationId => "local_invocation_id"
    WorkgroupId => "workgroup_id"
    NumWorkgroups => "num_workgroups"
    VertexIndex => "vertex_index"
    InstanceIndex => "instance_index"
    Position => "position"
    FrontFacing => "front_facing"
    FragDepth => "frag_depth"
    SampleIndex => "sample_index"
    SampleMask => "sample_mask"
  }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

///|
pub(all) enum MirStmt {
  Let(String, MirType, MirExpr)
  VarDecl(String, MirType, MirExpr)
  Assign(String, MirExpr)
  IfStmt(MirExpr, Array[MirStmt], Array[MirStmt])
  Return(MirExpr)
  Block(Array[MirStmt])
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

///|
pub(all) enum MirExpr {
  Lit(MirLit)
  VarExpr(String, MirType)
  BinOpE(MirBinOp, MirExpr, MirExpr, MirType)
  UnaryOpE(MirUnaryOp, MirExpr, MirType)
  Call(String, Array[MirExpr], MirType)
  ConstructStruct(String, Array[MirExpr])
  FieldAccess(MirExpr, String, MirType)
  Index(MirExpr, MirExpr, MirType)
  Cast(MirExpr, MirType)
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

///|
pub(all) enum MirLit {
  I32Lit(Int)
  U32Lit(UInt)
  F32Lit(Double)
  BoolLit(Bool)
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

///|
pub(all) enum MirBinOp {
  Add
  Sub
  Mul
  Div
  Mod
  Eq
  Neq
  Lt
  Le
  Gt
  Ge
  And
  Or
  BitAnd
  BitOr
  BitXor
  Shl
  Shr
} derive(Eq)

///|
pub fn MirBinOp::display(self : MirBinOp) -> String {
  match self {
    Add => "+"
    Sub => "-"
    Mul => "*"
    Div => "/"
    Mod => "%"
    Eq => "=="
    Neq => "!="
    Lt => "<"
    Le => "<="
    Gt => ">"
    Ge => ">="
    And => "&&"
    Or => "||"
    BitAnd => "&"
    BitOr => "|"
    BitXor => "^"
    Shl => "<<"
    Shr => ">>"
  }
}

///|
pub(all) enum MirUnaryOp {
  Neg
  Not
  BitNot
} derive(Eq)

///|
pub fn MirUnaryOp::display(self : MirUnaryOp) -> String {
  match self {
    Neg => "-"
    Not => "!"
    BitNot => "~"
  }
}
