///|
/// WGSL code generation from MIR.
///
/// A tree-walk emitter that translates a `MirProgram` into valid WGSL source
/// text. The emitter walks each struct, function, and entry point in the
/// program and serialises them into a `String` buffer.

///|
fn strip_builtin_prefix(name : String) -> String {
  if name.length() > 0 && name.unsafe_charcode_at(0) == 0x24 {
    name.substring(start=1, end=name.length())
  } else {
    name
  }
}

///|
fn sanitize_identifier(name : String) -> String {
  let len = name.length()
  let mut out = ""
  let mut i = 0
  while i < len {
    let c = name.unsafe_charcode_at(i)
    let is_alnum = (c >= 0x30 && c <= 0x39) ||
      (c >= 0x41 && c <= 0x5A) ||
      (c >= 0x61 && c <= 0x7A) ||
      c == 0x5F
    if is_alnum {
      out = out + name.substring(start=i, end=i + 1)
    } else if c == 0x27 {
      out = out + "_prime"
    } else {
      out = out + "_"
    }
    i = i + 1
  }
  if out.length() == 0 {
    return "_"
  }
  let first = out.unsafe_charcode_at(0)
  if first >= 0x30 && first <= 0x39 {
    "_" + out
  } else if is_reserved_identifier(out) {
    "fw_" + out
  } else {
    out
  }
}

///|
fn is_reserved_identifier(name : String) -> Bool {
  match name {
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
    | "final" => true
    _ => false
  }
}

///|
/// Tree-walk emitter that writes WGSL text into a `String` buffer.
pub struct WgslEmitter {
  mut output : String
  mut indent : Int
}

///|
pub fn WgslEmitter::new() -> WgslEmitter {
  WgslEmitter::{ output: "", indent: 0 }
}

///|
fn WgslEmitter::write(self : WgslEmitter, s : String) -> Unit {
  self.output = self.output + s
}

///|
fn WgslEmitter::write_indent(self : WgslEmitter) -> Unit {
  let mut i = 0
  while i < self.indent {
    self.output = self.output + "  "
    i = i + 1
  }
}

///|
fn WgslEmitter::newline(self : WgslEmitter) -> Unit {
  self.output = self.output + "\n"
}

///|
fn trim_trailing_whitespace(s : String) -> String {
  let mut end = s.length()
  while end > 0 {
    let c = s.unsafe_charcode_at(end - 1)
    if c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D {
      end = end - 1
    } else {
      break
    }
  }
  s.substring(start=0, end=end)
}

///|
/// Emit the WGSL text for a complete program.
pub fn WgslEmitter::emit_program(
  self : WgslEmitter,
  program : @mir.MirProgram
) -> String {
  for s in program.structs {
    self.emit_struct(s)
    self.newline()
  }
  for f in program.functions {
    self.emit_function(f)
    self.newline()
  }
  for ep in program.entry_points {
    self.emit_entry_point(ep)
    self.newline()
  }
  trim_trailing_whitespace(self.output) + "\n"
}

///|
fn WgslEmitter::emit_struct(self : WgslEmitter, s : @mir.MirStruct) -> Unit {
  self.write("struct " + sanitize_identifier(s.name) + " {")
  self.newline()
  self.indent = self.indent + 1
  for field in s.fields {
    self.write_indent()
    self.write(sanitize_identifier(field.name) + ": " + field.ty.display() + ",")
    self.newline()
  }
  self.indent = self.indent - 1
  self.write("}")
  self.newline()
}

///|
fn WgslEmitter::emit_function(
  self : WgslEmitter,
  f : @mir.MirFunction
) -> Unit {
  self.write("fn " + sanitize_identifier(f.name) + "(")
  let mut first = true
  for param in f.params {
    if not(first) {
      self.write(", ")
    }
    first = false
    self.write(sanitize_identifier(param.name) + ": " + param.ty.display())
  }
  self.write(")")
  if f.return_ty != @mir.MirType::Unit {
    self.write(" -> " + f.return_ty.display())
  }
  self.write(" {")
  self.newline()
  self.indent = self.indent + 1
  for stmt in f.body {
    self.emit_stmt(stmt)
  }
  match f.return_expr {
    Some(expr) => {
      self.write_indent()
      self.write("return ")
      self.emit_expr(expr)
      self.write(";")
      self.newline()
    }
    None => ()
  }
  self.indent = self.indent - 1
  self.write("}")
  self.newline()
}

///|
fn WgslEmitter::emit_entry_point(
  self : WgslEmitter,
  ep : @mir.MirEntryPoint
) -> Unit {
  match ep.stage {
    Compute => {
      match ep.workgroup_size {
        Some(wg) => {
          let (x, y, z) = wg
          self.write(
            "@compute @workgroup_size(" +
            x.to_string() + ", " + y.to_string() + ", " + z.to_string() + ")",
          )
        }
        None => self.write("@compute @workgroup_size(1, 1, 1)")
      }
    }
    Vertex => self.write("@vertex")
    Fragment => self.write("@fragment")
  }
  self.newline()
  self.write("fn " + sanitize_identifier(ep.name) + "(")
  let mut first = true
  for triple in ep.builtins {
    if not(first) {
      self.write(", ")
    }
    first = false
    let (name, binding, ty) = triple
    self.write(
      "@builtin(" + binding.display() + ") " +
      sanitize_identifier(name) + ": " + ty.display(),
    )
  }
  for param in ep.params {
    if not(first) {
      self.write(", ")
    }
    first = false
    self.write(sanitize_identifier(param.name) + ": " + param.ty.display())
  }
  self.write(")")
  if ep.return_ty != @mir.MirType::Unit {
    self.write(" -> " + ep.return_ty.display())
  }
  self.write(" {")
  self.newline()
  self.indent = self.indent + 1
  for stmt in ep.body {
    self.emit_stmt(stmt)
  }
  match ep.return_expr {
    Some(expr) => {
      self.write_indent()
      self.write("return ")
      self.emit_expr(expr)
      self.write(";")
      self.newline()
    }
    None => ()
  }
  self.indent = self.indent - 1
  self.write("}")
  self.newline()
}

///|
fn WgslEmitter::emit_stmt(self : WgslEmitter, stmt : @mir.MirStmt) -> Unit {
  match stmt {
    Let(name, ty, expr) => {
      self.write_indent()
      self.write("let " + sanitize_identifier(name) + ": " + ty.display() + " = ")
      self.emit_expr(expr)
      self.write(";")
      self.newline()
    }
    VarDecl(name, ty, expr) => {
      self.write_indent()
      self.write("var " + sanitize_identifier(name) + ": " + ty.display() + " = ")
      self.emit_expr(expr)
      self.write(";")
      self.newline()
    }
    Assign(name, expr) => {
      self.write_indent()
      self.write(sanitize_identifier(name) + " = ")
      self.emit_expr(expr)
      self.write(";")
      self.newline()
    }
    IfStmt(cond, then_stmts, else_stmts) => {
      self.write_indent()
      self.write("if (")
      self.emit_expr(cond)
      self.write(") {")
      self.newline()
      self.indent = self.indent + 1
      for s in then_stmts {
        self.emit_stmt(s)
      }
      self.indent = self.indent - 1
      if else_stmts.length() == 0 {
        self.write_indent()
        self.write("}")
        self.newline()
      } else {
        self.write_indent()
        self.write("} else {")
        self.newline()
        self.indent = self.indent + 1
        for s in else_stmts {
          self.emit_stmt(s)
        }
        self.indent = self.indent - 1
        self.write_indent()
        self.write("}")
        self.newline()
      }
    }
    Return(expr) => {
      self.write_indent()
      self.write("return ")
      self.emit_expr(expr)
      self.write(";")
      self.newline()
    }
    Block(stmts) => {
      self.write_indent()
      self.write("{")
      self.newline()
      self.indent = self.indent + 1
      for s in stmts {
        self.emit_stmt(s)
      }
      self.indent = self.indent - 1
      self.write_indent()
      self.write("}")
      self.newline()
    }
  }
}

///|
fn WgslEmitter::emit_expr(self : WgslEmitter, expr : @mir.MirExpr) -> Unit {
  match expr {
    Lit(lit) => self.emit_lit(lit)
    VarExpr(name, _) => self.write(sanitize_identifier(strip_builtin_prefix(name)))
    BinOpE(op, lhs, rhs, _) => {
      self.write("(")
      self.emit_expr(lhs)
      self.write(" " + op.display() + " ")
      self.emit_expr(rhs)
      self.write(")")
    }
    UnaryOpE(op, operand, _) => {
      self.write(op.display())
      self.emit_expr(operand)
    }
    Call(name, args, ty) => {
      let callee = strip_builtin_prefix(name)
      if is_type_constructor_call(callee, ty) {
        self.write(ty.display())
      } else {
        self.write(sanitize_identifier(callee))
      }
      self.write("(")
      let mut first = true
      for arg in args {
        if not(first) {
          self.write(", ")
        }
        first = false
        self.emit_expr(arg)
      }
      self.write(")")
    }
    ConstructStruct(name, fields) => {
      self.write(sanitize_identifier(name))
      self.write("(")
      let mut first = true
      for field in fields {
        if not(first) {
          self.write(", ")
        }
        first = false
        self.emit_expr(field)
      }
      self.write(")")
    }
    FieldAccess(inner, field, _) => {
      self.emit_expr(inner)
      self.write("." + sanitize_identifier(field))
    }
    Index(array, index, _) => {
      self.emit_expr(array)
      self.write("[")
      self.emit_expr(index)
      self.write("]")
    }
    Cast(inner, ty) => {
      self.write(ty.display() + "(")
      self.emit_expr(inner)
      self.write(")")
    }
  }
}

///|
fn WgslEmitter::emit_lit(self : WgslEmitter, lit : @mir.MirLit) -> Unit {
  match lit {
    I32Lit(v) =>
      if v < 0 {
        self.write("(" + v.to_string() + "i)")
      } else {
        self.write(v.to_string() + "i")
      }
    U32Lit(v) => self.write(v.to_string() + "u")
    F32Lit(v) => {
      let s = v.to_string()
      if string_contains(s, ".") || string_contains(s, "e") || string_contains(s, "E") {
        self.write(s)
      } else {
        self.write(s + ".0")
      }
    }
    BoolLit(v) => self.write(if v { "true" } else { "false" })
  }
}

///|
fn string_contains(haystack : String, needle : String) -> Bool {
  let hlen = haystack.length()
  let nlen = needle.length()
  if nlen == 0 {
    return true
  }
  let mut i = 0
  while i + nlen <= hlen {
    if haystack.substring(start=i, end=i + nlen) == needle {
      return true
    }
    i = i + 1
  }
  false
}

///|
fn is_type_constructor_call(name : String, ty : @mir.MirType) -> Bool {
  match (name, ty) {
    ("vec2", MirVec(2, _)) => true
    ("vec3", MirVec(3, _)) => true
    ("vec4", MirVec(4, _)) => true
    ("mat2x2", MirMat(2, 2, _)) => true
    ("mat3x3", MirMat(3, 3, _)) => true
    ("mat4x4", MirMat(4, 4, _)) => true
    _ => false
  }
}

///|
/// Convenience: emit a `MirProgram` as WGSL source text.
pub fn emit_wgsl(program : @mir.MirProgram) -> String {
  WgslEmitter::new().emit_program(program)
}
