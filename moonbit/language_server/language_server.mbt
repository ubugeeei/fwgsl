///|
/// LSP server skeleton for fwgsl.
///
/// MoonBit doesn't ship an `lsp-types` equivalent yet, so the port lands
/// transport-agnostic helpers a JSON-RPC loop can drive:
///
/// - `DocumentStore` tracks `did_open`/`did_change`/`did_close` state.
/// - `compute_diagnostics` runs the parse + semantic pipeline and
///   returns flat `LspDiagnostic` records ready for serialisation to a
///   `textDocument/publishDiagnostics` notification.
/// - `semantic_tokens_for` / `completions_at` / `hover_at` /
///   `goto_definition_at` proxy through the `ide` library so a future
///   transport layer only has to think about JSON-RPC framing.

///|
pub(all) struct OpenDocument {
  uri : String
  mut text : String
  mut version : Int
}

///|
pub(all) struct DocumentStore {
  docs : Map[String, OpenDocument]
}

///|
pub fn DocumentStore::new() -> DocumentStore {
  DocumentStore::{ docs: Map::new() }
}

///|
pub fn DocumentStore::open(
  self : DocumentStore,
  uri : String,
  text : String,
  version : Int
) -> Unit {
  self.docs[uri] = OpenDocument::{ uri, text, version }
}

///|
pub fn DocumentStore::update(
  self : DocumentStore,
  uri : String,
  text : String,
  version : Int
) -> Unit {
  match self.docs.get(uri) {
    Some(doc) => {
      doc.text = text
      doc.version = version
    }
    None => self.open(uri, text, version)
  }
}

///|
pub fn DocumentStore::close(self : DocumentStore, uri : String) -> Unit {
  self.docs.remove(uri)
}

///|
pub fn DocumentStore::get(self : DocumentStore, uri : String) -> OpenDocument? {
  self.docs.get(uri)
}

// ---------------------------------------------------------------------------
// Diagnostics for publishDiagnostics
// ---------------------------------------------------------------------------

///|
/// LSP-shaped diagnostic. Mirrors the relevant subset of the
/// `lsp-types::Diagnostic` record so a transport layer can serialise
/// without having to touch the lower-level `@diagnostics.Diagnostic`.
pub(all) struct LspDiagnostic {
  message : String
  severity : LspSeverity
  start_line : Int
  start_character : Int
  end_line : Int
  end_character : Int
  help : String?
} derive(Show)

///|
pub(all) enum LspSeverity {
  ErrorSev
  WarningSev
  InfoSev
  HintSev
} derive(Eq, Show)

///|
/// Compute publishDiagnostics payload for a document.
///
/// Includes parser errors, semantic errors, and lint warnings; lint
/// warnings come last so a UI grouping by severity keeps the more
/// important compile errors at the top.
pub fn compute_diagnostics(text : String) -> Array[LspDiagnostic] {
  let parser = @parser.Parser::new(text)
  let program = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(program)
  let out : Array[LspDiagnostic] = Array::new()
  for d in parser.diagnostics().all() {
    out.push(to_lsp_diagnostic(d, text))
  }
  for d in analyzer.diagnostics().all() {
    out.push(to_lsp_diagnostic(d, text))
  }
  for d in @linter.lint_program(program) {
    out.push(to_lsp_diagnostic(d, text))
  }
  out
}

///|
fn to_lsp_diagnostic(
  d : @diagnostics.Diagnostic,
  source : String
) -> LspDiagnostic {
  let severity : LspSeverity = match d.severity {
    Error => ErrorSev
    Warning => WarningSev
    Info => InfoSev
    Hint => HintSev
  }
  let (start_span, end_span) = if d.labels.length() > 0 {
    let l = d.labels[0]
    (l.span.start, l.span.end)
  } else {
    (0U, 0U)
  }
  let (sl, sc) = offset_to_line_col(source, start_span)
  let (el, ec) = offset_to_line_col(source, end_span)
  LspDiagnostic::{
    message: d.message,
    severity,
    start_line: sl - 1,
    start_character: sc - 1,
    end_line: el - 1,
    end_character: ec - 1,
    help: d.help,
  }
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

// ---------------------------------------------------------------------------
// IDE proxies
// ---------------------------------------------------------------------------

///|
pub fn semantic_tokens_for(text : String) -> Array[@ide.SemanticToken] {
  @ide.semantic_tokens(text)
}

///|
pub fn completions_at(text : String, offset : UInt) -> Array[@ide.CompletionItem] {
  @ide.completions_at(text, offset)
}

///|
pub fn hover_at(text : String, offset : UInt) -> @ide.HoverInfo? {
  @ide.hover_at(text, offset)
}

///|
pub fn goto_definition_at(text : String, offset : UInt) -> @ide.GotoTarget? {
  @ide.goto_definition_at(text, offset)
}

///|
/// Format the document. Backs `textDocument/formatting`.
pub fn format_document(text : String) -> String {
  @formatter.format(text)
}

// Code actions
// ---------------------------------------------------------------------------

///|
/// A single LSP-shaped text edit, with `(line, character)` positions.
pub(all) struct LspTextEdit {
  start_line : Int
  start_character : Int
  end_line : Int
  end_character : Int
  new_text : String
} derive(Show)

///|
/// LSP-shaped code action ready for serialisation to a
/// `textDocument/codeAction` response.
pub(all) struct LspCodeAction {
  title : String
  /// LSP `CodeActionKind` string (`"quickfix"`, `"refactor"`, ...).
  kind : String
  edits : Array[LspTextEdit]
} derive(Show)

///|
/// Compute code actions for the document at offset `offset`. Backs
/// `textDocument/codeAction`.
pub fn code_actions_for(
  text : String,
  offset : UInt
) -> Array[LspCodeAction] {
  let out : Array[LspCodeAction] = Array::new()
  for a in @ide.code_actions_at(text, offset) {
    let lsp_edits : Array[LspTextEdit] = Array::new()
    for e in a.edits {
      let (sl, sc) = offset_to_line_col(text, e.start)
      let (el, ec) = offset_to_line_col(text, e.end)
      lsp_edits.push(
        LspTextEdit::{
          start_line: sl - 1,
          start_character: sc - 1,
          end_line: el - 1,
          end_character: ec - 1,
          new_text: e.new_text,
        },
      )
    }
    out.push(
      LspCodeAction::{ title: a.title, kind: a.kind.lsp_str(), edits: lsp_edits },
    )
  }
  out
}

// ---------------------------------------------------------------------------
// Inlay hints
// ---------------------------------------------------------------------------

///|
/// LSP-shaped inlay hint. Mirrors the subset of `lsp-types::InlayHint`
/// the transport layer needs to serialise the response for
/// `textDocument/inlayHint`.
pub(all) struct LspInlayHint {
  /// `(line, character)` position to anchor the hint (zero-based, as
  /// the LSP protocol expects).
  line : Int
  character : Int
  label : String
} derive(Show)

///|
/// Compute inlay hints for the document. Backs `textDocument/inlayHint`.
pub fn inlay_hints_for(text : String) -> Array[LspInlayHint] {
  let out : Array[LspInlayHint] = Array::new()
  for h in @ide.inlay_hints(text) {
    let (line, character) = offset_to_line_col(text, h.position)
    out.push(
      LspInlayHint::{
        line: line - 1,
        character: character - 1,
        label: h.label,
      },
    )
  }
  out
}

// ---------------------------------------------------------------------------
// JSON-RPC serialisation helpers
// ---------------------------------------------------------------------------

///|
/// Serialise an LSP diagnostic as a JSON object body (without surrounding
/// envelope). Good enough for the simple stdin/stdout transport layer
/// that will land in a follow-up.
pub fn diagnostic_to_json(d : LspDiagnostic) -> String {
  let severity_n = match d.severity {
    ErrorSev => 1
    WarningSev => 2
    InfoSev => 3
    HintSev => 4
  }
  let mut out = "{\"message\":" + json_string(d.message) +
    ",\"severity\":" + severity_n.to_string() +
    ",\"range\":{" +
    "\"start\":{\"line\":" + d.start_line.to_string() +
    ",\"character\":" + d.start_character.to_string() + "}," +
    "\"end\":{\"line\":" + d.end_line.to_string() +
    ",\"character\":" + d.end_character.to_string() + "}}"
  match d.help {
    Some(h) => out = out + ",\"help\":" + json_string(h)
    None => ()
  }
  out + "}"
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

///|
/// Render `compute_diagnostics(text)` as a JSON array of objects, ready
/// to drop into the `params.diagnostics` field of a
/// `textDocument/publishDiagnostics` notification.
pub fn diagnostics_json_for(text : String) -> String {
  let diags = compute_diagnostics(text)
  let mut out = "["
  let mut first = true
  for d in diags {
    if !first {
      out = out + ","
    }
    first = false
    out = out + diagnostic_to_json(d)
  }
  out + "]"
}
