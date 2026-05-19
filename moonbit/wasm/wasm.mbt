///|
/// WASM bindings for the fwgsl playground.
///
/// In the Rust workspace `fwgsl_wasm` uses `wasm-bindgen` to expose the
/// compiler to a JavaScript host. MoonBit compiles natively to wasm-gc,
/// so the equivalent is to export plain MoonBit functions and have the
/// JavaScript host call them through the wasm-gc ABI.
///
/// - `wasm_check(source)` -> JSON array of diagnostics.
/// - `wasm_compile(source)` -> JSON object with `diagnostics` and `wgsl`.

///|
/// Run parser + semantic diagnostics and return them as a JSON array.
///
/// Includes lint warnings alongside parser and semantic errors so the
/// playground can surface unused-let / shadowed-binding / redundant-
/// wildcard hints in the same panel as compile errors.
pub fn wasm_check(source : String) -> String {
  let parser = @parser.Parser::new(source)
  let prog = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(prog)
  let diags : Array[@diagnostics.Diagnostic] = Array::new()
  for d in parser.diagnostics().all() {
    diags.push(d)
  }
  for d in analyzer.diagnostics().all() {
    diags.push(d)
  }
  for d in @linter.lint_program(prog) {
    diags.push(d)
  }
  diagnostics_to_json(diags, source)
}

///|
/// Run the full parse -> semantic -> AST lowering -> MIR -> WGSL
/// pipeline. Returns a JSON object with `diagnostics` (parser +
/// semantic + AST-lowering + MIR + lint diagnostics) and `wgsl` (the
/// generated shader text, or `null` if any error-severity diagnostic
/// was reported).
pub fn wasm_compile(source : String) -> String {
  let parser = @parser.Parser::new(source)
  let prog = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(prog)
  let diags : Array[@diagnostics.Diagnostic] = Array::new()
  for d in parser.diagnostics().all() {
    diags.push(d)
  }
  for d in analyzer.diagnostics().all() {
    diags.push(d)
  }
  let lowering = @ast_lowering.AstLowering::new(analyzer)
  let hir = lowering.lower_program(prog)
  for d in lowering.diagnostics().all() {
    diags.push(d)
  }
  let has_errors = any_error(diags)
  let wgsl : String? = if has_errors {
    None
  } else {
    match @mir.lower_hir_to_mir(hir) {
      Ok(mir_program) => Some(@wgsl_codegen.emit_wgsl(mir_program))
      Err(errs) => {
        for e in errs {
          diags.push(@diagnostics.Diagnostic::error(e))
        }
        None
      }
    }
  }
  // Lint warnings come after the compile diagnostics so an editor that
  // groups by severity keeps the compile errors at the top.
  for d in @linter.lint_program(prog) {
    diags.push(d)
  }
  let mut out = "{\"diagnostics\":"
  out = out + diagnostics_to_json(diags, source)
  out = out + ",\"wgsl\":"
  match wgsl {
    Some(w) => out = out + json_string(w)
    None => out = out + "null"
  }
  out + "}"
}

///|
fn any_error(diags : Array[@diagnostics.Diagnostic]) -> Bool {
  for d in diags {
    if d.severity == @diagnostics.Severity::Error {
      return true
    }
  }
  false
}

///|
fn diagnostics_to_json(
  diags : Array[@diagnostics.Diagnostic],
  _source : String
) -> String {
  let mut out = "["
  let mut first = true
  for diag in diags {
    if !first {
      out = out + ","
    }
    first = false
    let severity = match diag.severity {
      Error => "error"
      Warning => "warning"
      Info => "info"
      Hint => "hint"
    }
    out = out + "{\"severity\":\"" + severity + "\","
    out = out + "\"message\":" + json_string(diag.message)
    match diag.code {
      Some(c) => out = out + ",\"code\":" + json_string(c)
      None => ()
    }
    if diag.labels.length() > 0 {
      out = out + ",\"labels\":["
      let mut lfirst = true
      for label in diag.labels {
        if !lfirst {
          out = out + ","
        }
        lfirst = false
        out = out +
          "{\"start\":" + label.span.start.to_string() +
          ",\"end\":" + label.span.end.to_string() +
          ",\"message\":" + json_string(label.message) + "}"
      }
      out = out + "]"
    }
    match diag.help {
      Some(h) => out = out + ",\"help\":" + json_string(h)
      None => ()
    }
    out = out + "}"
  }
  out + "]"
}

///|
fn json_string(s : String) -> String {
  let len = s.length()
  let mut out = "\""
  let mut i = 0
  while i < len {
    let c = s.unsafe_charcode_at(i)
    if c == 0x22 {
      out = out + "\\\""
    } else if c == 0x5C {
      out = out + "\\\\"
    } else if c == 0x0A {
      out = out + "\\n"
    } else if c == 0x0D {
      out = out + "\\r"
    } else if c == 0x09 {
      out = out + "\\t"
    } else if c < 0x20 {
      out = out + "\\u00" + hex_byte(c)
    } else {
      out = out + s.unsafe_substring(start=i, end=i + 1)
    }
    i = i + 1
  }
  out + "\""
}

///|
fn hex_byte(b : Int) -> String {
  let hex = "0123456789abcdef"
  let hi = (b / 16) % 16
  let lo = b % 16
  hex.unsafe_substring(start=hi, end=hi + 1) + hex.unsafe_substring(start=lo, end=lo + 1)
}
