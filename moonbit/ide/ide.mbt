///|
/// IDE-facing analysis APIs for fwgsl.
///
/// The Rust `fwgsl_ide` crate (~2300 lines including the catalog) drives
/// completion, hover, goto-definition, and semantic-token computation
/// shared between the LSP and WASM playground. The MoonBit port lands
/// the public surface here; the catalog and individual analysis bodies
/// will follow as we finish porting the underlying compiler stages.

///|
pub(all) enum CompletionKind {
  Keyword
  Function
  Constructor
  Type
  Variable
  Attribute
} derive(Eq, Show)

///|
pub(all) struct CompletionItem {
  label : String
  kind : CompletionKind
  detail : String?
  documentation : String?
  snippet : String?
} derive(Show)

///|
pub fn CompletionItem::new(label : String, kind : CompletionKind) -> CompletionItem {
  CompletionItem::{ label, kind, detail: None, documentation: None, snippet: None }
}

///|
pub(all) struct HoverInfo {
  contents : String
  span : @span.Span
} derive(Show)

///|
pub(all) struct GotoTarget {
  span : @span.Span
} derive(Show)

///|
pub(all) enum SemanticTokenKind {
  TokKeyword
  TokOperator
  TokDecorator
  TokType
  TokFunction
  TokVariable
  TokParameter
  TokString
  TokNumber
  TokComment
} derive(Eq, Show)

///|
pub(all) struct SemanticToken {
  kind : SemanticTokenKind
  span : @span.Span
} derive(Show)

///|
/// Compute completion items for a given source position.
///
/// Placeholder implementation: returns the static keyword list so the
/// downstream packages can already wire up an IDE without depending on
/// the still-incomplete inference engine.
pub fn completions_at(
  _source : String,
  _offset : UInt
) -> Array[CompletionItem] {
  let items : Array[CompletionItem] = Array::new()
  for kw in [
      "let", "in", "where", "data", "type", "case", "of", "match", "if",
      "then", "else", "do", "forall", "infixl", "infixr", "infix", "deriving",
      "class", "instance", "import", "module",
    ] {
    items.push(CompletionItem::new(kw, Keyword))
  }
  items
}

///|
/// Compute the hover summary for the symbol under `offset`.
///
/// Placeholder: returns `None` until name resolution is wired up.
pub fn hover_at(_source : String, _offset : UInt) -> HoverInfo? {
  None
}

///|
/// Resolve a goto-definition request.
///
/// Placeholder: returns `None` until name resolution is wired up.
pub fn goto_definition_at(_source : String, _offset : UInt) -> GotoTarget? {
  None
}

///|
/// Compute semantic tokens for the entire document.
///
/// The Rust crate emits a much richer stream; this placeholder turns the
/// raw lexer output into a coarse `keyword | operator | comment` colouring
/// so the LSP can already show *something*.
pub fn semantic_tokens(source : String) -> Array[SemanticToken] {
  let out : Array[SemanticToken] = Array::new()
  for tok in @parser.lex(source) {
    let kind : SemanticTokenKind? = if tok.kind.is_keyword() {
      Some(TokKeyword)
    } else if tok.kind.is_operator() {
      Some(TokOperator)
    } else if tok.kind == @syntax.SyntaxKind::LineComment
      || tok.kind == @syntax.SyntaxKind::BlockComment {
      Some(TokComment)
    } else {
      None
    }
    match kind {
      Some(k) => out.push(SemanticToken::{ kind: k, span: tok.span })
      None => ()
    }
  }
  out
}
