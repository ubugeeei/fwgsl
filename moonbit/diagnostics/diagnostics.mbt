///|
/// Diagnostic reporting for fwgsl.
///
/// Provides structured diagnostics with severity levels, source labels,
/// and a renderer that formats diagnostics as human-readable text. The
/// Rust crate used `miette` for terminal rendering; here we render directly
/// since MoonBit's wasm-gc target has no equivalent crate.

///|
/// Severity level for a diagnostic.
pub(all) enum Severity {
  Error
  Warning
  Info
  Hint
} derive(Eq, Show)

///|
/// A label pointing to a source location with an associated message.
pub(all) struct Label {
  span : @span.Span
  message : String
} derive(Show)

///|
/// Create a new label at the given span with a message.
pub fn Label::new(span : @span.Span, message : String) -> Label {
  Label::{ span, message }
}

///|
/// Create a primary label (alias for `new`).
pub fn Label::primary(span : @span.Span, message : String) -> Label {
  Label::{ span, message }
}

///|
/// A structured diagnostic message.
pub(all) struct Diagnostic {
  mut severity : Severity
  mut message : String
  mut code : String?
  mut labels : Array[Label]
  mut help : String?
} derive(Show)

///|
/// Create a new error diagnostic.
pub fn Diagnostic::error(message : String) -> Diagnostic {
  Diagnostic::{
    severity: Error,
    message,
    code: None,
    labels: Array::new(),
    help: None,
  }
}

///|
/// Create a new warning diagnostic.
pub fn Diagnostic::warning(message : String) -> Diagnostic {
  Diagnostic::{
    severity: Warning,
    message,
    code: None,
    labels: Array::new(),
    help: None,
  }
}

///|
/// Add a label to the diagnostic (builder pattern).
pub fn Diagnostic::with_label(self : Diagnostic, label : Label) -> Diagnostic {
  self.labels.push(label)
  self
}

///|
/// Set the help message (builder pattern).
pub fn Diagnostic::with_help(self : Diagnostic, help : String) -> Diagnostic {
  self.help = Some(help)
  self
}

///|
/// Set the error code (builder pattern).
pub fn Diagnostic::with_code(self : Diagnostic, code : String) -> Diagnostic {
  self.code = Some(code)
  self
}

///|
/// A collection of diagnostics accumulated during compilation.
pub(all) struct DiagnosticSink {
  diagnostics : Array[Diagnostic]
}

///|
/// Create a new empty diagnostic sink.
pub fn DiagnosticSink::new() -> DiagnosticSink {
  DiagnosticSink::{ diagnostics: Array::new() }
}

///|
/// Push a diagnostic into the sink.
pub fn DiagnosticSink::push(self : DiagnosticSink, diagnostic : Diagnostic) -> Unit {
  self.diagnostics.push(diagnostic)
}

///|
/// Return all diagnostics in insertion order.
pub fn DiagnosticSink::all(self : DiagnosticSink) -> Array[Diagnostic] {
  self.diagnostics
}

///|
/// Check whether any error-level diagnostics have been reported.
pub fn DiagnosticSink::has_errors(self : DiagnosticSink) -> Bool {
  for d in self.diagnostics {
    if d.severity == Error {
      return true
    }
  }
  false
}

///|
/// Number of diagnostics in the sink.
pub fn DiagnosticSink::len(self : DiagnosticSink) -> Int {
  self.diagnostics.length()
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

///|
/// Render a single diagnostic against the source text as a multi-line
/// string with `miette` / `rustc`-style source-line excerpts:
///
/// ```text
/// error: Unbound variable: y
///  --> file.fwgsl:1:7
///   |
/// 1 | f x = y
///   |       ^ not in scope
///   |
///   help: bind the name in a parameter, let, where, or import declaration
/// ```
pub fn render(
  diag : Diagnostic,
  source_name : String,
  source : String
) -> String {
  let buf = StringBuilder::new()
  buf.write_string(severity_prefix(diag.severity))
  match diag.code {
    Some(code) => {
      buf.write_string("[")
      buf.write_string(code)
      buf.write_string("] ")
    }
    None => ()
  }
  buf.write_string(diag.message)
  buf.write_string("\n")
  for label in diag.labels {
    let (line, col) = offset_to_line_col(source, label.span.start)
    let (_, end_col) = offset_to_line_col(source, label.span.end)
    let line_num_str = line.to_string()
    let gutter_width = line_num_str.length()
    let gutter_pad = repeat_char(' ', gutter_width)
    buf.write_string(" --> ")
    buf.write_string(source_name)
    buf.write_string(":")
    buf.write_string(line_num_str)
    buf.write_string(":")
    buf.write_string(col.to_string())
    buf.write_string("\n")
    buf.write_string(gutter_pad)
    buf.write_string(" |\n")
    let snippet = extract_line(source, label.span.start)
    buf.write_string(line_num_str)
    buf.write_string(" | ")
    buf.write_string(snippet)
    buf.write_string("\n")
    buf.write_string(gutter_pad)
    buf.write_string(" | ")
    buf.write_string(repeat_char(' ', col - 1))
    let caret_len = if end_col > col { end_col - col } else { 1 }
    buf.write_string(repeat_char('^', caret_len))
    if label.message.length() > 0 {
      buf.write_string(" ")
      buf.write_string(label.message)
    }
    buf.write_string("\n")
    buf.write_string(gutter_pad)
    buf.write_string(" |\n")
  }
  match diag.help {
    Some(help) => {
      buf.write_string("  help: ")
      buf.write_string(help)
      buf.write_string("\n")
    }
    None => ()
  }
  buf.to_string()
}

///|
/// Repeat `c` `n` times. Returns an empty string when `n` <= 0.
fn repeat_char(c : Char, n : Int) -> String {
  if n <= 0 {
    return ""
  }
  let buf = StringBuilder::new()
  let mut i = 0
  while i < n {
    buf.write_char(c)
    i = i + 1
  }
  buf.to_string()
}

///|
/// Extract the source line containing `offset` (without the trailing
/// newline).
fn extract_line(source : String, offset : UInt) -> String {
  let off = offset.reinterpret_as_int()
  let len = source.length()
  let safe_off = if off > len { len } else { off }
  let mut start = safe_off
  while start > 0 && source.unsafe_charcode_at(start - 1) != 0x0A {
    start = start - 1
  }
  let mut end = safe_off
  while end < len && source.unsafe_charcode_at(end) != 0x0A {
    end = end + 1
  }
  source.unsafe_substring(start~, end~)
}

///|
fn severity_prefix(s : Severity) -> String {
  match s {
    Error => "error: "
    Warning => "warning: "
    Info => "info: "
    Hint => "hint: "
  }
}

///|
/// Convert a byte offset into a 1-based (line, column) pair.
fn offset_to_line_col(source : String, offset : UInt) -> (Int, Int) {
  let target = offset.reinterpret_as_int()
  let mut line = 1
  let mut col = 1
  let mut i = 0
  let len = source.length()
  while i < len && i < target {
    let ch = source.unsafe_charcode_at(i)
    if ch == '\n' {
      line = line + 1
      col = 1
    } else {
      col = col + 1
    }
    i = i + 1
  }
  (line, col)
}
