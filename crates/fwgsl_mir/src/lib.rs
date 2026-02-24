//! Mid-level Intermediate Representation (MIR) for fwgsl.
//!
//! MIR is a lowered representation suitable for code generation.
//! It has been type-checked, monomorphised, and desugared from HIR
//! into a form that maps closely to WGSL constructs.

pub mod lower;

use std::fmt;

// ---------------------------------------------------------------------------
// Top-level program
// ---------------------------------------------------------------------------

/// A complete MIR program ready for code generation.
#[derive(Debug, Clone, PartialEq)]
pub struct MirProgram {
    pub structs: Vec<MirStruct>,
    pub functions: Vec<MirFunction>,
    pub entry_points: Vec<MirEntryPoint>,
}

// ---------------------------------------------------------------------------
// Struct definitions
// ---------------------------------------------------------------------------

/// A struct type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct MirStruct {
    pub name: String,
    pub fields: Vec<MirField>,
}

/// A single field in a struct.
#[derive(Debug, Clone, PartialEq)]
pub struct MirField {
    pub name: String,
    pub ty: MirType,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Concrete types that map to WGSL types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MirType {
    I32,
    U32,
    F32,
    Bool,
    /// `vec{n}<T>` — e.g. `Vec(3, F32)` → `vec3<f32>`
    Vec(u8, Box<MirType>),
    /// `mat{cols}x{rows}<T>` — e.g. `Mat(4, 4, F32)` → `mat4x4<f32>`
    Mat(u8, u8, Box<MirType>),
    /// A user-defined struct type.
    Struct(String),
    /// `array<T, N>`
    Array(Box<MirType>, u32),
    /// The unit type — no WGSL representation (used for void returns).
    Unit,
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::I32 => write!(f, "i32"),
            MirType::U32 => write!(f, "u32"),
            MirType::F32 => write!(f, "f32"),
            MirType::Bool => write!(f, "bool"),
            MirType::Vec(n, inner) => write!(f, "vec{}<{}>", n, inner),
            MirType::Mat(cols, rows, inner) => write!(f, "mat{}x{}<{}>", cols, rows, inner),
            MirType::Struct(name) => write!(f, "{}", name),
            MirType::Array(inner, len) => write!(f, "array<{}, {}>", inner, len),
            MirType::Unit => write!(f, "void"),
        }
    }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// A regular (non-entry-point) function.
#[derive(Debug, Clone, PartialEq)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<MirParam>,
    pub return_ty: MirType,
    pub body: Vec<MirStmt>,
    pub return_expr: Option<MirExpr>,
}

/// A function parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct MirParam {
    pub name: String,
    pub ty: MirType,
}

// ---------------------------------------------------------------------------
// Entry points (shader stages)
// ---------------------------------------------------------------------------

/// An entry-point function annotated with a shader stage.
#[derive(Debug, Clone, PartialEq)]
pub struct MirEntryPoint {
    pub name: String,
    pub stage: ShaderStage,
    /// Workgroup size for compute shaders — `[x, y, z]`.
    pub workgroup_size: Option<[u32; 3]>,
    pub params: Vec<MirParam>,
    /// Built-in bindings such as `@builtin(global_invocation_id)`.
    pub builtins: Vec<(String, BuiltinBinding, MirType)>,
    pub return_ty: MirType,
    pub body: Vec<MirStmt>,
    pub return_expr: Option<MirExpr>,
}

/// Shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Compute,
    Vertex,
    Fragment,
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderStage::Compute => write!(f, "compute"),
            ShaderStage::Vertex => write!(f, "vertex"),
            ShaderStage::Fragment => write!(f, "fragment"),
        }
    }
}

/// Built-in variable bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinBinding {
    GlobalInvocationId,
    LocalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    VertexIndex,
    InstanceIndex,
    Position,
    FrontFacing,
    FragDepth,
    SampleIndex,
    SampleMask,
}

impl fmt::Display for BuiltinBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BuiltinBinding::GlobalInvocationId => "global_invocation_id",
            BuiltinBinding::LocalInvocationId => "local_invocation_id",
            BuiltinBinding::WorkgroupId => "workgroup_id",
            BuiltinBinding::NumWorkgroups => "num_workgroups",
            BuiltinBinding::VertexIndex => "vertex_index",
            BuiltinBinding::InstanceIndex => "instance_index",
            BuiltinBinding::Position => "position",
            BuiltinBinding::FrontFacing => "front_facing",
            BuiltinBinding::FragDepth => "frag_depth",
            BuiltinBinding::SampleIndex => "sample_index",
            BuiltinBinding::SampleMask => "sample_mask",
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// A MIR statement.
#[derive(Debug, Clone, PartialEq)]
pub enum MirStmt {
    /// `let name: ty = expr;`
    Let(String, MirType, MirExpr),
    /// `var name: ty = expr;`
    Var(String, MirType, MirExpr),
    /// `name = expr;`
    Assign(String, MirExpr),
    /// `if (cond) { then } else { else }`
    If(MirExpr, Vec<MirStmt>, Vec<MirStmt>),
    /// `return expr;`
    Return(MirExpr),
    /// A nested block `{ ... }`
    Block(Vec<MirStmt>),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// A MIR expression.
#[derive(Debug, Clone, PartialEq)]
pub enum MirExpr {
    /// A literal value.
    Lit(MirLit),
    /// A variable reference. The second element is the type.
    Var(String, MirType),
    /// A binary operation: `op(lhs, rhs) -> ty`.
    BinOp(MirBinOp, Box<MirExpr>, Box<MirExpr>, MirType),
    /// A unary operation: `op(operand) -> ty`.
    UnaryOp(MirUnaryOp, Box<MirExpr>, MirType),
    /// A function call: `name(args) -> ty`.
    Call(String, Vec<MirExpr>, MirType),
    /// Struct construction: `Name(field_exprs...)`.
    ConstructStruct(String, Vec<MirExpr>),
    /// Field access: `expr.field -> ty`.
    FieldAccess(Box<MirExpr>, String, MirType),
    /// Index access: `expr[index] -> ty`.
    Index(Box<MirExpr>, Box<MirExpr>, MirType),
    /// Type cast: `ty(expr)`.
    Cast(Box<MirExpr>, MirType),
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
pub enum MirLit {
    I32(i32),
    U32(u32),
    F32(f64), // stored as f64 for precision during compilation
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl fmt::Display for MirBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MirBinOp::Add => "+",
            MirBinOp::Sub => "-",
            MirBinOp::Mul => "*",
            MirBinOp::Div => "/",
            MirBinOp::Mod => "%",
            MirBinOp::Eq => "==",
            MirBinOp::Neq => "!=",
            MirBinOp::Lt => "<",
            MirBinOp::Le => "<=",
            MirBinOp::Gt => ">",
            MirBinOp::Ge => ">=",
            MirBinOp::And => "&&",
            MirBinOp::Or => "||",
            MirBinOp::BitAnd => "&",
            MirBinOp::BitOr => "|",
            MirBinOp::BitXor => "^",
            MirBinOp::Shl => "<<",
            MirBinOp::Shr => ">>",
        };
        write!(f, "{}", s)
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirUnaryOp {
    Neg,
    Not,
    BitNot,
}

impl fmt::Display for MirUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MirUnaryOp::Neg => "-",
            MirUnaryOp::Not => "!",
            MirUnaryOp::BitNot => "~",
        };
        write!(f, "{}", s)
    }
}
