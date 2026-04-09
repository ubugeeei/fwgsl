//! WGSL code generation from MIR.
//!
//! This crate provides a tree-walk emitter that translates a [`MirProgram`]
//! into valid WGSL source text. The emitter walks each struct, function, and
//! entry point in the program and serialises them into a `String` buffer.

use fwgsl_mir::*;

fn sanitize_identifier(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => out.push(ch),
            '\'' => out.push_str("_prime"),
            _ => out.push('_'),
        }
    }

    if out.is_empty() {
        "_".to_string()
    } else if out.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        format!("_{}", out)
    } else if is_reserved_identifier(&out) {
        format!("fw_{}", out)
    } else {
        out
    }
}

fn is_reserved_identifier(name: &str) -> bool {
    matches!(
        name,
        "alias"
            | "bitcast"
            | "bool"
            | "break"
            | "case"
            | "compute"
            | "const"
            | "const_assert"
            | "continue"
            | "continuing"
            | "default"
            | "diagnostic"
            | "discard"
            | "else"
            | "enable"
            | "f32"
            | "false"
            | "fn"
            | "for"
            | "fragment"
            | "i32"
            | "if"
            | "let"
            | "loop"
            | "mat2x2"
            | "mat3x3"
            | "mat4x4"
            | "override"
            | "private"
            | "read"
            | "read_write"
            | "requires"
            | "return"
            | "storage"
            | "struct"
            | "switch"
            | "true"
            | "u32"
            | "uniform"
            | "var"
            | "vec2"
            | "vec3"
            | "vec4"
            | "vertex"
            | "while"
            | "workgroup"
            | "write"
            | "final"
    )
}

/// A tree-walk emitter that writes WGSL text into a `String` buffer.
pub struct WgslEmitter {
    output: String,
    indent: usize,
    preserve_comments: bool,
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
            preserve_comments: false,
        }
    }

    /// Create a new emitter that preserves source comments.
    pub fn with_comments() -> Self {
        Self {
            output: String::new(),
            indent: 0,
            preserve_comments: true,
        }
    }

    /// Consume the emitter and produce the WGSL text for the given program.
    pub fn emit_program(mut self, program: &MirProgram) -> String {
        // Emit constants
        for c in &program.constants {
            self.emit_const(c);
            self.newline();
        }

        // Emit structs
        for s in &program.structs {
            self.emit_struct(s);
            self.newline();
        }

        // Emit global bindings (resources)
        for g in &program.globals {
            self.emit_global(g);
            self.newline();
        }

        // Emit functions
        for f in &program.functions {
            self.emit_comments(&f.comments);
            self.emit_function(f);
            self.newline();
        }

        // Emit entry points
        for ep in &program.entry_points {
            self.emit_comments(&ep.comments);
            self.emit_entry_point(ep);
            self.newline();
        }

        self.output.trim_end().to_string() + "\n"
    }

    // -----------------------------------------------------------------------
    // Struct emission
    // -----------------------------------------------------------------------
    // Constant emission
    // -----------------------------------------------------------------------

    fn emit_comments(&mut self, comments: &[String]) {
        if !self.preserve_comments || comments.is_empty() {
            return;
        }
        for comment in comments {
            self.write(&format!("//{}", comment));
            self.newline();
        }
    }

    fn emit_const(&mut self, c: &MirConst) {
        self.write(&format!(
            "const {}: {} = ",
            sanitize_identifier(&c.name),
            self.format_type(&c.ty)
        ));
        self.emit_expr(&c.value);
        self.write(";");
        self.newline();
    }

    // -----------------------------------------------------------------------
    // Struct emission
    // -----------------------------------------------------------------------

    fn emit_struct(&mut self, s: &MirStruct) {
        self.write(&format!("struct {} {{", sanitize_identifier(&s.name)));
        self.newline();
        self.indent += 1;
        for field in &s.fields {
            self.write_indent();
            // Emit field attributes (e.g. @location(0), @builtin(position))
            for attr in &field.attributes {
                self.write("@");
                self.write(&attr.name);
                if !attr.args.is_empty() {
                    self.write("(");
                    self.write(&attr.args.join(", "));
                    self.write(")");
                }
                self.write(" ");
            }
            self.write(&format!(
                "{}: {},",
                sanitize_identifier(&field.name),
                self.format_type(&field.ty)
            ));
            self.newline();
        }
        self.indent -= 1;
        self.write("}");
        self.newline();
    }

    // -----------------------------------------------------------------------
    // Global variable emission (resource bindings)
    // -----------------------------------------------------------------------

    fn emit_global(&mut self, g: &MirGlobal) {
        let addr_space = match g.address_space {
            AddressSpace::Uniform => "uniform",
            AddressSpace::StorageRead => "storage, read",
            AddressSpace::StorageReadWrite => "storage, read_write",
        };
        self.write(&format!(
            "@group({}) @binding({}) var<{}> {}: {};",
            g.group,
            g.binding,
            addr_space,
            sanitize_identifier(&g.name),
            self.format_type(&g.ty),
        ));
        self.newline();
    }

    // -----------------------------------------------------------------------
    // Function emission
    // -----------------------------------------------------------------------

    fn emit_function(&mut self, f: &MirFunction) {
        self.write(&format!("fn {}(", sanitize_identifier(&f.name)));
        for (i, param) in f.params.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.write(&format!(
                "{}: {}",
                sanitize_identifier(&param.name),
                self.format_type(&param.ty)
            ));
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

        self.write(&format!("fn {}(", sanitize_identifier(&ep.name)));

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
                sanitize_identifier(name),
                self.format_type(ty)
            ));
        }

        // Regular parameters
        for param in &ep.params {
            if !first {
                self.write(", ");
            }
            first = false;
            if let Some(loc) = param.location {
                self.write(&format!(
                    "@location({}) {}: {}",
                    loc,
                    sanitize_identifier(&param.name),
                    self.format_type(&param.ty)
                ));
            } else {
                self.write(&format!(
                    "{}: {}",
                    sanitize_identifier(&param.name),
                    self.format_type(&param.ty)
                ));
            }
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
                self.write(&format!(
                    "let {}: {} = ",
                    sanitize_identifier(name),
                    self.format_type(ty)
                ));
                self.emit_expr(expr);
                self.write(";");
                self.newline();
            }
            MirStmt::Var(name, ty, expr) => {
                self.write_indent();
                self.write(&format!(
                    "var {}: {} = ",
                    sanitize_identifier(name),
                    self.format_type(ty)
                ));
                self.emit_expr(expr);
                self.write(";");
                self.newline();
            }
            MirStmt::Assign(name, expr) => {
                self.write_indent();
                self.write(&format!("{} = ", sanitize_identifier(name)));
                self.emit_expr(expr);
                self.write(";");
                self.newline();
            }
            MirStmt::IndexAssign(base, index, value) => {
                self.write_indent();
                self.emit_expr(base);
                self.write("[");
                self.emit_expr(index);
                self.write("] = ");
                self.emit_expr(value);
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
            MirStmt::Switch(expr, cases, default_body) => {
                self.write_indent();
                self.write("switch (");
                self.emit_expr(expr);
                self.write(") {");
                self.newline();
                self.indent += 1;
                for case in cases {
                    self.write_indent();
                    self.write("case ");
                    for (i, val) in case.values.iter().enumerate() {
                        if i > 0 {
                            self.write(", ");
                        }
                        self.emit_lit(val);
                    }
                    self.write(": {");
                    self.newline();
                    self.indent += 1;
                    for s in &case.body {
                        self.emit_stmt(s);
                    }
                    self.indent -= 1;
                    self.write_indent();
                    self.write("}");
                    self.newline();
                }
                if !default_body.is_empty() {
                    self.write_indent();
                    self.write("default: {");
                    self.newline();
                    self.indent += 1;
                    for s in default_body {
                        self.emit_stmt(s);
                    }
                    self.indent -= 1;
                    self.write_indent();
                    self.write("}");
                    self.newline();
                }
                self.indent -= 1;
                self.write_indent();
                self.write("}");
                self.newline();
            }
            MirStmt::Loop(body) => {
                self.write_indent();
                self.write("loop {");
                self.newline();
                self.indent += 1;
                for s in body {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                self.write_indent();
                self.write("}");
                self.newline();
            }
            MirStmt::Break => {
                self.write_indent();
                self.write("break;");
                self.newline();
            }
            MirStmt::Continue => {
                self.write_indent();
                self.write("continue;");
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
            MirExpr::Var(name, _) => self.write(&sanitize_identifier(name)),
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
            MirExpr::Call(name, args, ty) => {
                if is_type_constructor_call(name, ty) {
                    self.write(&self.format_type(ty));
                } else {
                    self.write(&sanitize_identifier(name));
                }
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
                self.write(&sanitize_identifier(name));
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
                self.write(&format!(".{}", sanitize_identifier(field)));
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

fn is_type_constructor_call(name: &str, ty: &MirType) -> bool {
    matches!(
        (name, ty),
        ("vec2", MirType::Vec(2, _))
            | ("vec3", MirType::Vec(3, _))
            | ("vec4", MirType::Vec(4, _))
            | ("mat2x2", MirType::Mat(2, 2, _))
            | ("mat3x3", MirType::Mat(3, 3, _))
            | ("mat4x4", MirType::Mat(4, 4, _))
    )
}

/// Convenience function to emit a [`MirProgram`] as WGSL source text.
pub fn emit_wgsl(program: &MirProgram) -> String {
    WgslEmitter::new().emit_program(program)
}

/// Emit a [`MirProgram`] as WGSL, preserving source comments.
pub fn emit_wgsl_with_comments(program: &MirProgram) -> String {
    WgslEmitter::with_comments().emit_program(program)
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
            globals: vec![],
            functions: vec![MirFunction {
                name: "add".to_string(),
                params: vec![
                    MirParam {
                        name: "x".to_string(),
                        ty: MirType::I32,
                        location: None,
                    },
                    MirParam {
                        name: "y".to_string(),
                        ty: MirType::I32,
                        location: None,
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("fn add(x: i32, y: i32) -> i32"));
        assert!(wgsl.contains("return (x + y)"));
    }

    #[test]
    fn test_emit_void_function() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
            functions: vec![MirFunction {
                name: "do_nothing".to_string(),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![],
                return_expr: None,
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
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
                        attributes: vec![],
                    },
                    MirField {
                        name: "position".to_string(),
                        ty: MirType::Vec(3, Box::new(MirType::F32)),
                        attributes: vec![],
                    },
                    MirField {
                        name: "life".to_string(),
                        ty: MirType::F32,
                        attributes: vec![],
                    },
                ],
            }],
            globals: vec![],
            functions: vec![],
            entry_points: vec![],
            constants: vec![],
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
            globals: vec![],
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
                comments: vec![],
            }],
            constants: vec![],
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
            globals: vec![],
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
                comments: vec![],
            }],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@vertex"));
        assert!(wgsl.contains("fn vs_main("));
        assert!(wgsl.contains("@builtin(vertex_index) vid: u32"));
        assert!(wgsl.contains("-> vec4<f32>"));
        assert!(wgsl.contains("return vec4<f32>(0.0, 0.0, 0.0, 1.0);"));
    }

    #[test]
    fn test_emit_fragment_entry_point() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
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
                comments: vec![],
            }],
            constants: vec![],
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
            globals: vec![],
            functions: vec![MirFunction {
                name: "abs_val".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                    location: None,
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
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
            globals: vec![],
            functions: vec![MirFunction {
                name: "maybe_inc".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                    location: None,
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
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
            globals: vec![],
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("let a: i32 = 42i;"));
        assert!(wgsl.contains("var b: f32 = 3.14;"));
    }

    #[test]
    fn test_emit_literals() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
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
            globals: vec![],
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
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
                        attributes: vec![],
                    },
                    MirField {
                        name: "y".to_string(),
                        ty: MirType::F32,
                        attributes: vec![],
                    },
                ],
            }],
            globals: vec![],
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return Vec2(1.0, 2.0);"));
    }

    #[test]
    fn test_emit_field_access() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
            functions: vec![MirFunction {
                name: "get_x".to_string(),
                params: vec![MirParam {
                    name: "v".to_string(),
                    ty: MirType::Vec(3, Box::new(MirType::F32)),
                    location: None,
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return v.x;"));
    }

    #[test]
    fn test_emit_index_access() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
            functions: vec![MirFunction {
                name: "get_elem".to_string(),
                params: vec![MirParam {
                    name: "arr".to_string(),
                    ty: MirType::Array(Box::new(MirType::F32), 4),
                    location: None,
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return arr[0u];"));
    }

    #[test]
    fn test_emit_cast() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
            functions: vec![MirFunction {
                name: "to_float".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                    location: None,
                }],
                return_ty: MirType::F32,
                body: vec![],
                return_expr: Some(MirExpr::Cast(
                    Box::new(MirExpr::Var("x".to_string(), MirType::I32)),
                    MirType::F32,
                )),
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("return f32(x);"));
    }

    #[test]
    fn test_emit_block_statement() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("  {\n    let x: i32 = 1i;\n  }"));
    }

    #[test]
    fn test_emit_mat_type() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
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
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
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
                    attributes: vec![],
                }],
            }],
            globals: vec![],
            functions: vec![],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("values: array<f32, 16>,"));
    }

    #[test]
    fn test_emit_compute_default_workgroup() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
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
                comments: vec![],
            }],
            constants: vec![],
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
                        attributes: vec![],
                    },
                    MirField {
                        name: "vel".to_string(),
                        ty: MirType::Vec(3, Box::new(MirType::F32)),
                        attributes: vec![],
                    },
                ],
            }],
            globals: vec![],
            functions: vec![MirFunction {
                name: "step_particle".to_string(),
                params: vec![
                    MirParam {
                        name: "p".to_string(),
                        ty: MirType::Struct("Particle".to_string()),
                        location: None,
                    },
                    MirParam {
                        name: "dt".to_string(),
                        ty: MirType::F32,
                        location: None,
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
                comments: vec![],
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
                comments: vec![],
            }],
            constants: vec![],
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

    #[test]
    fn test_emit_struct_field_attributes() {
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "VertexOutput".to_string(),
                fields: vec![
                    MirField {
                        name: "clip_position".to_string(),
                        ty: MirType::Vec(4, Box::new(MirType::F32)),
                        attributes: vec![MirAttribute {
                            name: "builtin".to_string(),
                            args: vec!["position".to_string()],
                        }],
                    },
                    MirField {
                        name: "color".to_string(),
                        ty: MirType::Vec(4, Box::new(MirType::F32)),
                        attributes: vec![MirAttribute {
                            name: "location".to_string(),
                            args: vec!["0".to_string()],
                        }],
                    },
                    MirField {
                        name: "uv".to_string(),
                        ty: MirType::Vec(2, Box::new(MirType::F32)),
                        attributes: vec![
                            MirAttribute {
                                name: "location".to_string(),
                                args: vec!["1".to_string()],
                            },
                            MirAttribute {
                                name: "interpolate".to_string(),
                                args: vec!["linear".to_string(), "center".to_string()],
                            },
                        ],
                    },
                ],
            }],
            globals: vec![],
            functions: vec![],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(
            wgsl.contains("@builtin(position) clip_position: vec4<f32>,"),
            "expected @builtin(position) attribute, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("@location(0) color: vec4<f32>,"),
            "expected @location(0) attribute, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("@location(1) @interpolate(linear, center) uv: vec2<f32>,"),
            "expected multiple attributes, got:\n{}",
            wgsl
        );
    }

    #[test]
    fn test_emit_switch_statement() {
        // Build a function that contains a switch on a u32 variable
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
            functions: vec![MirFunction {
                name: "classify".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::U32,
                    location: None,
                }],
                return_ty: MirType::I32,
                body: vec![
                    MirStmt::Var(
                        "result".to_string(),
                        MirType::I32,
                        MirExpr::Lit(MirLit::I32(0)),
                    ),
                    MirStmt::Switch(
                        MirExpr::Var("x".to_string(), MirType::U32),
                        vec![
                            MirSwitchCase {
                                values: vec![MirLit::U32(0)],
                                body: vec![MirStmt::Assign(
                                    "result".to_string(),
                                    MirExpr::Lit(MirLit::I32(10)),
                                )],
                            },
                            MirSwitchCase {
                                values: vec![MirLit::U32(1)],
                                body: vec![MirStmt::Assign(
                                    "result".to_string(),
                                    MirExpr::Lit(MirLit::I32(20)),
                                )],
                            },
                            MirSwitchCase {
                                values: vec![MirLit::U32(2)],
                                body: vec![MirStmt::Assign(
                                    "result".to_string(),
                                    MirExpr::Lit(MirLit::I32(30)),
                                )],
                            },
                        ],
                        // default body
                        vec![MirStmt::Assign(
                            "result".to_string(),
                            MirExpr::Lit(MirLit::I32(-1)),
                        )],
                    ),
                ],
                return_expr: Some(MirExpr::Var("result".to_string(), MirType::I32)),
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(
            wgsl.contains("switch (x)"),
            "expected switch statement, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("case 0u:"),
            "expected case 0u, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("case 1u:"),
            "expected case 1u, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("case 2u:"),
            "expected case 2u, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("default:"),
            "expected default case, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("result = 10i;"),
            "expected result = 10i in case body, got:\n{}",
            wgsl
        );
        assert!(
            wgsl.contains("result = (-1i);"),
            "expected result = (-1i) in default body, got:\n{}",
            wgsl
        );
    }
}
