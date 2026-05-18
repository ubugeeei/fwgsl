///|
/// `fwgsl` CLI entry point (scaffold).
///
/// The Rust `fwgsl_cli/src/main.rs` reads files from disk, drives the
/// parser / semantic / lowering / codegen pipeline, and prints the WGSL
/// output. This MoonBit port currently exposes a `compile` function that
/// takes the source as a `String` so it can be exercised from MoonBit
/// tests without depending on a particular I/O API. A future PR will
/// wire it up to MoonBit's `@fs` / `@sys` modules to read files and
/// expose the binary subcommands (`compile`, `check`, `fmt`, `version`).

///|
pub const VERSION = "0.1.0"

///|
/// Drive the front half of the pipeline (parse + semantic) on an
/// in-memory source string and return any diagnostics it produced.
pub fn check_source(source : String) -> @diagnostics.DiagnosticSink {
  let parser = @parser.Parser::new(source)
  let _ = parser.parse_program()
  parser.diagnostics()
}

///|
/// Drive parse + semantic registration. Returns the `Program` plus the
/// parser's diagnostic sink. Once AST -> HIR lowering, MIR lowering, and
/// codegen are wired in, this will be extended to return the WGSL output.
pub fn compile_source(
  source : String
) -> (@ast.Program, @diagnostics.DiagnosticSink) {
  let parser = @parser.Parser::new(source)
  let program = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(program)
  let _ = analyzer
  (program, parser.diagnostics())
}

///|
/// Format a diagnostic with a `file:line:col: severity: message` header
/// and an optional `help:` follow-up line.
pub fn format_diagnostic(
  diag : @diagnostics.Diagnostic,
  file : String,
  source : String
) -> String {
  let severity = match diag.severity {
    Error => "error"
    Warning => "warning"
    Info => "info"
    Hint => "hint"
  }
  let location = if diag.labels.length() > 0 {
    let label = diag.labels[0]
    let (line, col) = offset_to_line_col(source, label.span.start)
    file + ":" + line.to_string() + ":" + col.to_string()
  } else {
    file
  }
  let mut out = location + ": " + severity + ": " + diag.message
  match diag.help {
    Some(help) => out = out + "\n  help: " + help
    None => ()
  }
  out
}

///|
fn offset_to_line_col(source : String, offset : UInt) -> (Int, Int) {
  let target = offset.reinterpret_as_int()
  let mut line = 1
  let mut col = 1
  let mut i = 0
  let len = source.length()
  while i < len && i < target {
    let c = source.charcode_at(i)
    if c == 0x0A {
      line = line + 1
      col = 1
    } else {
      col = col + 1
    }
    i = i + 1
  }
  (line, col)
}

///|
/// Print the CLI usage banner to stdout. Called from the binary entry
/// point in `moonbit/cli/cmd/main`.
pub fn print_help() -> Unit {
  println("fwgsl " + VERSION + " (MoonBit rewrite, scaffold)")
  println("USAGE:")
  println("    fwgsl <COMMAND> [OPTIONS]")
  println("")
  println("COMMANDS:")
  println("    compile, c  <file>    Compile .fwgsl to .wgsl (not wired up)")
  println("    check       <file>    Type-check without emitting   (not wired up)")
  println("    version               Print version")
  println("    help                  Print this help")
}
