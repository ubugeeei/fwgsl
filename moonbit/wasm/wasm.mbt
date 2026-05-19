///|
/// WASM bindings for the fwgsl playground.
///
/// MoonBit compiles natively to `wasm-gc`, so the public functions here
/// are listed in `moon.pkg.json` under `link.wasm-gc.exports` to surface
/// them to the JavaScript host. `link.wasm-gc.use-js-builtin-string`
/// makes MoonBit `String` values pass to / from JS as native JS strings
/// without manual marshaling.
///
/// Public exports:
///
/// - `wasm_check(source)` -> JSON array of diagnostics (legacy alias).
/// - `wasm_compile(source)` -> JSON object with `diagnostics` and `wgsl`
///    (legacy alias).
/// - `compile(source)` -> identical to `wasm_compile` — the playground
///    calls this name.
/// - `format(source)` -> reformatted source (the token-stream formatter).
/// - `editor_completions(source, line, column)` -> JSON array of
///    completion items.
/// - `editor_hover(source, line, column)` -> JSON object with `contents`
///    and `range`, or `null`.
/// - `editor_definition(source, line, column)` -> JSON array of
///    `{ range }` locations.
/// - `editor_references(source, line, column, includeDeclaration)` ->
///    JSON array of `{ range }` locations.

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

// ---------------------------------------------------------------------------
// Playground-facing exports
// ---------------------------------------------------------------------------
//
// The playground (`playground/app.js`) calls these directly. Names are
// short on purpose — `compile(source)`, `format(source)`,
// `editor_completions(source, line, column)`, etc. The legacy
// `wasm_check` / `wasm_compile` entry points remain for backward
// compatibility.

///|
/// Compile alias for the playground. Same shape as `wasm_compile`:
/// JSON object with `diagnostics` and `wgsl`.
pub fn compile(source : String) -> String {
  wasm_compile(source)
}

///|
/// Reformat source with the token-stream formatter. Backs the
/// playground's "Format" button and the LSP `textDocument/formatting`
/// request.
pub fn format(source : String) -> String {
  @formatter.format(source)
}

///|
/// Editor completion list for `source` at `(line, column)` (both
/// 1-based, matching Monaco's positions). Returns a JSON array of
/// completion items shaped like `{ label, kind, detail?, range }`.
pub fn editor_completions(
  source : String,
  line : Int,
  column : Int
) -> String {
  let offset = line_col_to_offset(source, line, column)
  let items : Array[String] = Array::new()
  for it in @ide.completions_at(source, offset) {
    items.push(completion_item_to_json(it, line, column))
  }
  join_json_array(items)
}

///|
/// Editor hover at `(line, column)`. Returns a JSON object
/// `{ contents, range }` or the string `"null"`.
pub fn editor_hover(source : String, line : Int, column : Int) -> String {
  let offset = line_col_to_offset(source, line, column)
  match @ide.hover_at(source, offset) {
    None => "null"
    Some(info) => {
      let range = span_to_monaco_range(source, info.span)
      "{\"contents\":" +
      json_string(info.contents) +
      ",\"range\":" +
      range +
      "}"
    }
  }
}

///|
/// Editor goto-definition at `(line, column)`. Returns a JSON array
/// of `{ range }` locations (empty array when nothing matches).
pub fn editor_definition(
  source : String,
  line : Int,
  column : Int
) -> String {
  let offset = line_col_to_offset(source, line, column)
  match @ide.goto_definition_at(source, offset) {
    None => "[]"
    Some(target) => {
      let range = span_to_monaco_range(source, target.span)
      "[{\"range\":" + range + "}]"
    }
  }
}

///|
/// Editor find-references at `(line, column)`. The current IDE layer
/// only provides goto-definition, so for now we return a single entry
/// pointing at the definition (which is what most editors do as a
/// fallback). Empty array when nothing matches.
pub fn editor_references(
  source : String,
  line : Int,
  column : Int,
  _include_declaration : Bool
) -> String {
  editor_definition(source, line, column)
}

///|
/// Convert a 1-based `(line, column)` pair into a byte offset for use
/// with the byte-offset-based IDE APIs.
fn line_col_to_offset(source : String, line : Int, column : Int) -> UInt {
  if line <= 0 || column <= 0 {
    return 0U
  }
  let target_line = line - 1
  let target_col = column - 1
  let len = source.length()
  let mut i = 0
  let mut cur_line = 0
  let mut cur_col = 0
  while i < len {
    if cur_line == target_line && cur_col == target_col {
      break
    }
    let c = source.unsafe_charcode_at(i)
    if c == 0x0A {
      if cur_line == target_line {
        // Asked for a column past the end of this line; clamp here.
        break
      }
      cur_line = cur_line + 1
      cur_col = 0
    } else {
      cur_col = cur_col + 1
    }
    i = i + 1
  }
  i.reinterpret_as_uint()
}

///|
/// Render `(@span.Span)` as a Monaco-shaped `{ startLineNumber, startColumn,
/// endLineNumber, endColumn }` JSON object (both 1-based).
fn span_to_monaco_range(source : String, span : @span.Span) -> String {
  let (sl, sc) = offset_to_line_col(source, span.start)
  let (el, ec) = offset_to_line_col(source, span.end)
  "{\"startLineNumber\":" +
  sl.to_string() +
  ",\"startColumn\":" +
  sc.to_string() +
  ",\"endLineNumber\":" +
  el.to_string() +
  ",\"endColumn\":" +
  ec.to_string() +
  "}"
}

///|
fn offset_to_line_col(source : String, offset : UInt) -> (Int, Int) {
  let target = offset.reinterpret_as_int()
  let mut line = 1
  let mut col = 1
  let mut i = 0
  let len = source.length()
  while i < len && i < target {
    let c = source.unsafe_charcode_at(i)
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
fn completion_item_to_json(
  it : @ide.CompletionItem,
  line : Int,
  column : Int
) -> String {
  let kind_num = match it.kind {
    Keyword => 17 // monaco.languages.CompletionItemKind.Keyword
    Function => 1 // CompletionItemKind.Function -> Method (closest)
    Constructor => 2 // CompletionItemKind.Constructor -> Function
    Type => 14 // CompletionItemKind.Class
    Variable => 4 // CompletionItemKind.Variable
    Attribute => 5 // CompletionItemKind.Property
  }
  let range = "{\"startLineNumber\":" +
    line.to_string() +
    ",\"startColumn\":" +
    column.to_string() +
    ",\"endLineNumber\":" +
    line.to_string() +
    ",\"endColumn\":" +
    column.to_string() +
    "}"
  let mut out = "{\"label\":" +
    json_string(it.label) +
    ",\"kind\":" +
    kind_num.to_string() +
    ",\"range\":" +
    range
  match it.detail {
    Some(d) => out = out + ",\"detail\":" + json_string(d)
    None => ()
  }
  out + "}"
}

///|
fn join_json_array(items : Array[String]) -> String {
  let mut out = "["
  let mut first = true
  for item in items {
    if !first {
      out = out + ","
    }
    first = false
    out = out + item
  }
  out + "]"
}
