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
pub fn wasm_check(source : String) -> String {
  let parser = @parser.Parser::new(source)
  let prog = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(prog)
  let combined = combine(parser.diagnostics(), analyzer.diagnostics())
  diagnostics_to_json(combined, source)
}

///|
/// Run parse + semantic analysis. Returns a JSON object with
/// `diagnostics` (an array) and `wgsl` (a string or `null`). When any
/// stage produces an error-severity diagnostic, `wgsl` is `null`.
pub fn wasm_compile(source : String) -> String {
  let parser = @parser.Parser::new(source)
  let prog = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(prog)
  let combined = combine(parser.diagnostics(), analyzer.diagnostics())
  let mut out = "{\"diagnostics\":"
  out = out + diagnostics_to_json(combined, source)
  out = out + ",\"wgsl\":null}"
  out
}

///|
fn combine(
  a : @diagnostics.DiagnosticSink,
  b : @diagnostics.DiagnosticSink
) -> Array[@diagnostics.Diagnostic] {
  let out : Array[@diagnostics.Diagnostic] = Array::new()
  for d in a.all() {
    out.push(d)
  }
  for d in b.all() {
    out.push(d)
  }
  out
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
