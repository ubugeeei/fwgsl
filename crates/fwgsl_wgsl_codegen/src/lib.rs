//! WGSL code generation from MIR.
//!
//! This crate provides a tree-walk emitter that translates a [`MirProgram`]
//! into valid WGSL source text. The emitter walks each struct, function, and
//! entry point in the program and serialises them into a `String` buffer.

use fwgsl_mir::*;

/// Strip the `$` prefix from builtin identifiers for WGSL output.
fn strip_builtin_prefix(name: &str) -> &str {
    name.strip_prefix('$').unwrap_or(name)
}

/// A tree-walk emitter that writes WGSL text into a `String` buffer.
pub struct WgslEmitter {
    output: String,
    indent: usize,
}

impl Default for WgslEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl WgslEmitter {
    /// Create a new emitter with an empty output buffer.
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
        }
    }

    /// Consume the emitter and produce the WGSL text for the given program.
    pub fn emit_program(mut self, program: &MirProgram) -> String {
        // Emit structs
        for s in &program.structs {
            self.emit_struct(s);
            self.newline();
        }

        // Emit functions
        for f in &program.functions {
            self.emit_function(f);
            self.newline();
        }

        // Emit entry points
        for ep in &program.entry_points {
            self.emit_entry_point(ep);
            self.newline();
        }

        self.output.trim_end().to_string() + "\n"
    }

    // -----------------------------------------------------------------------
    // Struct emission
    // -----------------------------------------------------------------------

    fn emit_struct(&mut self, s: &MirStruct) {
        self.write(&format!("struct {} {{", s.name));
        self.newline();
        self.indent += 1;
        for field in &s.fields {
            self.write_indent();
            self.write(&format!("{}: {},", field.name, self.format_type(&field.ty)));
            self.newline();
        }
        self.indent -= 1;
        self.write("}");
        self.newline();
    }

    // -----------------------------------------------------------------------
    // Function emission
    // -----------------------------------------------------------------------

    fn emit_function(&mut self, f: &MirFunction) {
        self.write(&format!("fn {}(", f.name));
        for (i, param) in f.params.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.write(&format!("{}: {}", param.name, self.format_type(&param.ty)));
        }
        self.write(")");
        if f.return_ty != MirType::Unit {
            self.write(&format!(" -> {}", self.format_type(&f.return_ty)));
        }
        self.write(" {");
        self.newline();
        self.indent += 1;

        for stmt in &f.body {
            self.emit_stmt(stmt);
        }

        if let Some(ref expr) = f.return_expr {
            self.write_indent();
            self.write("return ");
            self.emit_expr(expr);
            self.write(";");
            self.newline();
        }

        self.indent -= 1;
        self.write("}");
        self.newline();
    }

    // -----------------------------------------------------------------------
    // Entry-point emission
    // -----------------------------------------------------------------------

    fn emit_entry_point(&mut self, ep: &MirEntryPoint) {
        // Stage attribute
        match ep.stage {
            ShaderStage::Compute => {
                if let Some(wg) = ep.workgroup_size {
                    self.write(&format!(
                        "@compute @workgroup_size({}, {}, {})",
                        wg[0], wg[1], wg[2]
                    ));
                } else {
                    self.write("@compute @workgroup_size(1, 1, 1)");
                }
            }
            ShaderStage::Vertex => self.write("@vertex"),
            ShaderStage::Fragment => self.write("@fragment"),
        }
        self.newline();

        self.write(&format!("fn {}(", ep.name));

        let mut first = true;

        // Built-in parameters
        for (name, binding, ty) in &ep.builtins {
            if !first {
                self.write(", ");
            }
            first = false;
            self.write(&format!(
                "@builtin({}) {}: {}",
                binding,
                name,
                self.format_type(ty)
            ));
        }

        // Regular parameters
        for param in &ep.params {
            if !first {
                self.write(", ");
            }
            first = false;
            self.write(&format!("{}: {}", param.name, self.format_type(&param.ty)));
        }

        self.write(")");
        if ep.return_ty != MirType::Unit {
            self.write(&format!(" -> {}", self.format_type(&ep.return_ty)));
        }
        self.write(" {");
        self.newline();
        self.indent += 1;

        for stmt in &ep.body {
            self.emit_stmt(stmt);
        }

        if let Some(ref expr) = ep.return_expr {
            self.write_indent();
            self.write("return ");
            self.emit_expr(expr);
            self.write(";");
            self.newline();
        }

        self.indent -= 1;
        self.write("}");
        self.newline();
    }

    // -----------------------------------------------------------------------
    // Statement emission
    // -----------------------------------------------------------------------

    fn emit_stmt(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Let(name, ty, expr) => {
                self.write_indent();
                self.write(&format!("let {}: {} = ", name, self.format_type(ty)));
                self.emit_expr(expr);
                self.write(";");
                self.newline();
            }
            MirStmt::Var(name, ty, expr) => {
                self.write_indent();
                self.write(&format!("var {}: {} = ", name, self.format_type(ty)));
                self.emit_expr(expr);
                self.write(";");
                self.newline();
            }
            MirStmt::Assign(name, expr) => {
                self.write_indent();
                self.write(&format!("{} = ", name));
                self.emit_expr(expr);
                self.write(";");
                self.newline();
            }
            MirStmt::If(cond, then_stmts, else_stmts) => {
                self.write_indent();
                self.write("if (");
                self.emit_expr(cond);
                self.write(") {");
                self.newline();
                self.indent += 1;
                for s in then_stmts {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                if else_stmts.is_empty() {
                    self.write_indent();
                    self.write("}");
                    self.newline();
                } else {
                    self.write_indent();
                    self.write("} else {");
                    self.newline();
                    self.indent += 1;
                    for s in else_stmts {
                        self.emit_stmt(s);
                    }
                    self.indent -= 1;
                    self.write_indent();
                    self.write("}");
                    self.newline();
                }
            }
            MirStmt::Return(expr) => {
                self.write_indent();
                self.write("return ");
                self.emit_expr(expr);
                self.write(";");
                self.newline();
            }
            MirStmt::Block(stmts) => {
                self.write_indent();
                self.write("{");
                self.newline();
                self.indent += 1;
                for s in stmts {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                self.write_indent();
                self.write("}");
                self.newline();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expression emission
    // -----------------------------------------------------------------------

    fn emit_expr(&mut self, expr: &MirExpr) {
        match expr {
            MirExpr::Lit(lit) => self.emit_lit(lit),
            MirExpr::Var(name, _) => self.write(strip_builtin_prefix(name)),
            MirExpr::BinOp(op, lhs, rhs, _) => {
                self.write("(");
                self.emit_expr(lhs);
                self.write(&format!(" {} ", op));
                self.emit_expr(rhs);
                self.write(")");
            }
            MirExpr::UnaryOp(op, operand, _) => {
                self.write(&format!("{}", op));
                self.emit_expr(operand);
            }
            MirExpr::Call(name, args, _) => {
                self.write(strip_builtin_prefix(name));
                self.write("(");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.emit_expr(arg);
                }
                self.write(")");
            }
            MirExpr::ConstructStruct(name, fields) => {
                self.write(name);
                self.write("(");
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.emit_expr(field);
                }
                self.write(")");
            }
            MirExpr::FieldAccess(expr, field, _) => {
                self.emit_expr(expr);
                self.write(&format!(".{}", field));
            }
            MirExpr::Index(array, index, _) => {
                self.emit_expr(array);
                self.write("[");
                self.emit_expr(index);
                self.write("]");
            }
            MirExpr::Cast(expr, ty) => {
                self.write(&format!("{}(", self.format_type(ty)));
                self.emit_expr(expr);
                self.write(")");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Literal emission
    // -----------------------------------------------------------------------

    fn emit_lit(&mut self, lit: &MirLit) {
        match lit {
            MirLit::I32(v) => {
                if *v < 0 {
                    // Wrap negative literals in parentheses for safety
                    self.write(&format!("({}i)", v));
                } else {
                    self.write(&format!("{}i", v));
                }
            }
            MirLit::U32(v) => self.write(&format!("{}u", v)),
            MirLit::F32(v) => {
                let s = format!("{}", v);
                if s.contains('.') {
                    self.write(&s);
                } else {
                    self.write(&format!("{}.0", s));
                }
            }
            MirLit::Bool(v) => self.write(if *v { "true" } else { "false" }),
        }
    }

    // -----------------------------------------------------------------------
    // Type formatting
    // -----------------------------------------------------------------------

    fn format_type(&self, ty: &MirType) -> String {
        ty.to_string()
    }

    // -----------------------------------------------------------------------
    // Buffer helpers
    // -----------------------------------------------------------------------

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.output.push_str("  ");
        }
    }

    fn newline(&mut self) {
        self.output.push('\n');
    }
}

/// Convenience function to emit a [`MirProgram`] as WGSL source text.
pub fn emit_wgsl(program: &MirProgram) -> String {
    WgslEmitter::new().emit_program(program)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_simple_function() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "add".to_string(),
                params: vec![
                    MirParam {
                        name: "x".to_string(),
                        ty: MirType::I32,
                    },
                    MirParam {
                        name: "y".to_string(),
                        ty: MirType::I32,
                    },
                ],
                return_ty: MirType::I32,
                body: vec![],
                return_expr: Some(MirExpr::BinOp(
                    MirBinOp::Add,
                    Box::new(MirExpr::Var("x".to_string(), MirType::I32)),
                    Box::new(MirExpr::Var("y".to_string(), MirType::I32)),
                    MirType::I32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("fn add(x: i32, y: i32) -> i32"));
        assert!(wgsl.contains("return (x + y)"));
    }

    #[test]
    fn test_emit_void_function() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "do_nothing".to_string(),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![],
                return_expr: None,
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("fn do_nothing() {"));
        assert!(!wgsl.contains("->"));
    }

    #[test]
    fn test_emit_struct() {
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "Particle".to_string(),
                fields: vec![
                    MirField {
                        name: "tag".to_string(),
                        ty: MirType::U32,
                    },
                    MirField {
                        name: "position".to_string(),
                        ty: MirType::Vec(3, Box::new(MirType::F32)),
                    },
                    MirField {
                        name: "life".to_string(),
                        ty: MirType::F32,
                    },
                ],
            }],
            functions: vec![],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("struct Particle {"));
        assert!(wgsl.contains("tag: u32,"));
        assert!(wgsl.contains("position: vec3<f32>,"));
        assert!(wgsl.contains("life: f32,"));
    }

    #[test]
    fn test_emit_compute_entry_point() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([64, 1, 1]),
                params: vec![],
                builtins: vec![(
                    "gid".to_string(),
                    BuiltinBinding::GlobalInvocationId,
                    MirType::Vec(3, Box::new(MirType::U32)),
                )],
                return_ty: MirType::Unit,
                body: vec![MirStmt::Let(
                    "idx".to_string(),
                    MirType::U32,
                    MirExpr::FieldAccess(
                        Box::new(MirExpr::Var(
                            "gid".to_string(),
                            MirType::Vec(3, Box::new(MirType::U32)),
                        )),
                        "x".to_string(),
                        MirType::U32,
                    ),
                )],
                return_expr: None,
            }],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@compute @workgroup_size(64, 1, 1)"));
        assert!(wgsl.contains("@builtin(global_invocation_id) gid: vec3<u32>"));
        assert!(wgsl.contains("let idx: u32 = gid.x;"));
    }

    #[test]
    fn test_emit_vertex_entry_point() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "vs_main".to_string(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
                params: vec![],
                builtins: vec![("vid".to_string(), BuiltinBinding::VertexIndex, MirType::U32)],
                return_ty: MirType::Vec(4, Box::new(MirType::F32)),
                body: vec![],
                return_expr: Some(MirExpr::Call(
                    "vec4".to_string(),
                    vec![
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(1.0)),
                    ],
                    MirType::Vec(4, Box::new(MirType::F32)),
                )),
            }],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@vertex"));
        assert!(wgsl.contains("fn vs_main("));
        assert!(wgsl.contains("@builtin(vertex_index) vid: u32"));
        assert!(wgsl.contains("-> vec4<f32>"));
        assert!(wgsl.contains("return vec4(0.0, 0.0, 0.0, 1.0);"));
    }

    #[test]
    fn test_emit_fragment_entry_point() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "fs_main".to_string(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
                params: vec![],
                builtins: vec![],
                return_ty: MirType::Vec(4, Box::new(MirType::F32)),
                body: vec![],
                return_expr: Some(MirExpr::Call(
                    "vec4".to_string(),
                    vec![
                        MirExpr::Lit(MirLit::F32(1.0)),
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(1.0)),
                    ],
                    MirType::Vec(4, Box::new(MirType::F32)),
                )),
            }],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@fragment"));
        assert!(wgsl.contains("fn fs_main()"));
        assert!(wgsl.contains("-> vec4<f32>"));
    }

    #[test]
    fn test_emit_if_statement() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "abs_val".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                }],
                return_ty: MirType::I32,
                body: vec![MirStmt::If(
                    MirExpr::BinOp(
                        MirBinOp::Lt,
                        Box::new(MirExpr::Var("x".to_string(), MirType::I32)),
                        Box::new(MirExpr::Lit(MirLit::I32(0))),
                        MirType::Bool,
                    ),
                    vec![MirStmt::Return(MirExpr::UnaryOp(
                        MirUnaryOp::Neg,
                        Box::new(MirExpr::Var("x".to_string(), MirType::I32)),
                        MirType::I32,
                    ))],
                    vec![MirStmt::Return(MirExpr::Var("x".to_string(), MirType::I32))],
                )],
                return_expr: None,
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("if ("));
        assert!(wgsl.contains("} else {"));
        assert!(wgsl.contains("return -x;"));
        assert!(wgsl.contains("return x;"));
    }

    #[test]
    fn test_emit_if_without_else() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "maybe_inc".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                }],
                return_ty: MirType::Unit,
                body: vec![
                    MirStmt::Var(
                        "r".to_string(),
                        MirType::I32,
                        MirExpr::Var("x".to_string(), MirType::I32),
                    ),
                    MirStmt::If(
                        MirExpr::BinOp(
                            MirBinOp::Gt,
                            Box::new(MirExpr::Var("x".to_string(), MirType::I32)),
                            Box::new(MirExpr::Lit(MirLit::I32(0))),
                            MirType::Bool,
                        ),
                        vec![MirStmt::Assign(
                            "r".to_string(),
                            MirExpr::BinOp(
                                MirBinOp::Add,
                                Box::new(MirExpr::Var("r".to_string(), MirType::I32)),
                                Box::new(MirExpr::Lit(MirLit::I32(1))),
                                MirType::I32,
                            ),
                        )],
                        vec![],
                    ),
                ],
                return_expr: None,
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("var r: i32 = x;"));
        assert!(wgsl.contains("if ("));
        assert!(!wgsl.contains("else"));
    }

    #[test]
    fn test_emit_let_and_var() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "f".to_string(),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![
                    MirStmt::Let("a".to_string(), MirType::I32, MirExpr::Lit(MirLit::I32(42))),
                    MirStmt::Var(
                        "b".to_string(),
                        MirType::F32,
                        MirExpr::Lit(MirLit::F32(3.14)),
                    ),
                ],
                return_expr: None,
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("let a: i32 = 42i;"));
        assert!(wgsl.contains("var b: f32 = 3.14;"));
    }

    #[test]
    fn test_emit_literals() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "lits".to_string(),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![
                    MirStmt::Let("a".to_string(), MirType::I32, MirExpr::Lit(MirLit::I32(-5))),
                    MirStmt::Let("b".to_string(), MirType::U32, MirExpr::Lit(MirLit::U32(10))),
                    MirStmt::Let(
                        "c".to_string(),
                        MirType::F32,
                        MirExpr::Lit(MirLit::F32(2.0)),
                    ),
                    MirStmt::Let(
                        "d".to_string(),
                        MirType::Bool,
                        MirExpr::Lit(MirLit::Bool(true)),
                    ),
                ],
                return_expr: None,
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("(-5i)"));
        assert!(wgsl.contains("10u"));
        assert!(wgsl.contains("2.0"));
        assert!(wgsl.contains("true"));
    }

    #[test]
    fn test_emit_call_expression() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "f".to_string(),
                params: vec![],
                return_ty: MirType::F32,
                body: vec![],
                return_expr: Some(MirExpr::Call(
                    "max".to_string(),
                    vec![
                        MirExpr::Lit(MirLit::F32(1.0)),
                        MirExpr::Lit(MirLit::F32(2.0)),
                    ],
                    MirType::F32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return max(1.0, 2.0);"));
    }

    #[test]
    fn test_emit_struct_construction() {
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "Vec2".to_string(),
                fields: vec![
                    MirField {
                        name: "x".to_string(),
                        ty: MirType::F32,
                    },
                    MirField {
                        name: "y".to_string(),
                        ty: MirType::F32,
                    },
                ],
            }],
            functions: vec![MirFunction {
                name: "make_vec".to_string(),
                params: vec![],
                return_ty: MirType::Struct("Vec2".to_string()),
                body: vec![],
                return_expr: Some(MirExpr::ConstructStruct(
                    "Vec2".to_string(),
                    vec![
                        MirExpr::Lit(MirLit::F32(1.0)),
                        MirExpr::Lit(MirLit::F32(2.0)),
                    ],
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return Vec2(1.0, 2.0);"));
    }

    #[test]
    fn test_emit_field_access() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "get_x".to_string(),
                params: vec![MirParam {
                    name: "v".to_string(),
                    ty: MirType::Vec(3, Box::new(MirType::F32)),
                }],
                return_ty: MirType::F32,
                body: vec![],
                return_expr: Some(MirExpr::FieldAccess(
                    Box::new(MirExpr::Var(
                        "v".to_string(),
                        MirType::Vec(3, Box::new(MirType::F32)),
                    )),
                    "x".to_string(),
                    MirType::F32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return v.x;"));
    }

    #[test]
    fn test_emit_index_access() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "get_elem".to_string(),
                params: vec![MirParam {
                    name: "arr".to_string(),
                    ty: MirType::Array(Box::new(MirType::F32), 4),
                }],
                return_ty: MirType::F32,
                body: vec![],
                return_expr: Some(MirExpr::Index(
                    Box::new(MirExpr::Var(
                        "arr".to_string(),
                        MirType::Array(Box::new(MirType::F32), 4),
                    )),
                    Box::new(MirExpr::Lit(MirLit::U32(0))),
                    MirType::F32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return arr[0u];"));
    }

    #[test]
    fn test_emit_cast() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "to_float".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                }],
                return_ty: MirType::F32,
                body: vec![],
                return_expr: Some(MirExpr::Cast(
                    Box::new(MirExpr::Var("x".to_string(), MirType::I32)),
                    MirType::F32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return f32(x);"));
    }

    #[test]
    fn test_emit_block_statement() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "f".to_string(),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![MirStmt::Block(vec![MirStmt::Let(
                    "x".to_string(),
                    MirType::I32,
                    MirExpr::Lit(MirLit::I32(1)),
                )])],
                return_expr: None,
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("  {\n    let x: i32 = 1i;\n  }"));
    }

    #[test]
    fn test_emit_mat_type() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "identity".to_string(),
                params: vec![],
                return_ty: MirType::Mat(4, 4, Box::new(MirType::F32)),
                body: vec![],
                return_expr: Some(MirExpr::Call(
                    "mat4x4".to_string(),
                    vec![],
                    MirType::Mat(4, 4, Box::new(MirType::F32)),
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("-> mat4x4<f32>"));
    }

    #[test]
    fn test_emit_array_type() {
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "Data".to_string(),
                fields: vec![MirField {
                    name: "values".to_string(),
                    ty: MirType::Array(Box::new(MirType::F32), 16),
                }],
            }],
            functions: vec![],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("values: array<f32, 16>,"));
    }

    #[test]
    fn test_emit_compute_default_workgroup() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: None,
                params: vec![],
                builtins: vec![],
                return_ty: MirType::Unit,
                body: vec![],
                return_expr: None,
            }],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@compute @workgroup_size(1, 1, 1)"));
    }

    #[test]
    fn test_emit_full_program() {
        // A complete small program: struct + helper function + compute entry point
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "Particle".to_string(),
                fields: vec![
                    MirField {
                        name: "pos".to_string(),
                        ty: MirType::Vec(3, Box::new(MirType::F32)),
                    },
                    MirField {
                        name: "vel".to_string(),
                        ty: MirType::Vec(3, Box::new(MirType::F32)),
                    },
                ],
            }],
            functions: vec![MirFunction {
                name: "step_particle".to_string(),
                params: vec![
                    MirParam {
                        name: "p".to_string(),
                        ty: MirType::Struct("Particle".to_string()),
                    },
                    MirParam {
                        name: "dt".to_string(),
                        ty: MirType::F32,
                    },
                ],
                return_ty: MirType::Struct("Particle".to_string()),
                body: vec![MirStmt::Let(
                    "new_pos".to_string(),
                    MirType::Vec(3, Box::new(MirType::F32)),
                    MirExpr::BinOp(
                        MirBinOp::Add,
                        Box::new(MirExpr::FieldAccess(
                            Box::new(MirExpr::Var(
                                "p".to_string(),
                                MirType::Struct("Particle".to_string()),
                            )),
                            "pos".to_string(),
                            MirType::Vec(3, Box::new(MirType::F32)),
                        )),
                        Box::new(MirExpr::BinOp(
                            MirBinOp::Mul,
                            Box::new(MirExpr::FieldAccess(
                                Box::new(MirExpr::Var(
                                    "p".to_string(),
                                    MirType::Struct("Particle".to_string()),
                                )),
                                "vel".to_string(),
                                MirType::Vec(3, Box::new(MirType::F32)),
                            )),
                            Box::new(MirExpr::Var("dt".to_string(), MirType::F32)),
                            MirType::Vec(3, Box::new(MirType::F32)),
                        )),
                        MirType::Vec(3, Box::new(MirType::F32)),
                    ),
                )],
                return_expr: Some(MirExpr::ConstructStruct(
                    "Particle".to_string(),
                    vec![
                        MirExpr::Var(
                            "new_pos".to_string(),
                            MirType::Vec(3, Box::new(MirType::F32)),
                        ),
                        MirExpr::FieldAccess(
                            Box::new(MirExpr::Var(
                                "p".to_string(),
                                MirType::Struct("Particle".to_string()),
                            )),
                            "vel".to_string(),
                            MirType::Vec(3, Box::new(MirType::F32)),
                        ),
                    ],
                )),
            }],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([256, 1, 1]),
                params: vec![],
                builtins: vec![(
                    "gid".to_string(),
                    BuiltinBinding::GlobalInvocationId,
                    MirType::Vec(3, Box::new(MirType::U32)),
                )],
                return_ty: MirType::Unit,
                body: vec![MirStmt::Let(
                    "idx".to_string(),
                    MirType::U32,
                    MirExpr::FieldAccess(
                        Box::new(MirExpr::Var(
                            "gid".to_string(),
                            MirType::Vec(3, Box::new(MirType::U32)),
                        )),
                        "x".to_string(),
                        MirType::U32,
                    ),
                )],
                return_expr: None,
            }],
        };

        let wgsl = emit_wgsl(&program);

        // Check that all major sections are present
        assert!(wgsl.contains("struct Particle {"));
        assert!(wgsl.contains("fn step_particle("));
        assert!(wgsl.contains("@compute @workgroup_size(256, 1, 1)"));
        assert!(wgsl.contains("fn main("));

        // Check that the output is properly ordered:
        // structs before functions, functions before entry points
        let struct_pos = wgsl.find("struct Particle").unwrap();
        let fn_pos = wgsl.find("fn step_particle").unwrap();
        let ep_pos = wgsl.find("@compute").unwrap();
        assert!(struct_pos < fn_pos);
        assert!(fn_pos < ep_pos);
    }
}
