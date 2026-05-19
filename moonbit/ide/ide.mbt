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
/// Find the lexer token whose span contains `offset`.
///
/// Returns `None` if the offset is past the end of the source or only
/// lands on trivia (whitespace / newlines). Comments are kept because
/// hovering over a comment is a useful signal for the user.
pub fn token_at(source : String, offset : UInt) -> @parser.Token? {
  for tok in @parser.lex(source) {
    if tok.kind == @syntax.SyntaxKind::Eof {
      break
    }
    if tok.span.contains(offset) {
      if tok.kind == @syntax.SyntaxKind::Whitespace ||
        tok.kind == @syntax.SyntaxKind::Newline {
        continue
      }
      return Some(tok)
    }
  }
  None
}

///|
/// Compute the hover summary for the symbol under `offset`.
///
/// Returns a brief, kind-aware label so the LSP can already show
/// something useful while full type resolution is still landing.
pub fn hover_at(source : String, offset : UInt) -> HoverInfo? {
  match token_at(source, offset) {
    None => None
    Some(tok) => {
      let text = tok.span.source_text(source)
      let label = if tok.kind.is_keyword() {
        "keyword `" + text + "`"
      } else if tok.kind == @syntax.SyntaxKind::Ident {
        "identifier `" + text + "`"
      } else if tok.kind == @syntax.SyntaxKind::UpperIdent {
        "constructor or type `" + text + "`"
      } else if tok.kind.is_literal() {
        "literal `" + text + "`"
      } else if tok.kind.is_operator() {
        "operator `" + text + "`"
      } else if tok.kind == @syntax.SyntaxKind::LineComment ||
        tok.kind == @syntax.SyntaxKind::BlockComment {
        "comment"
      } else {
        tok.kind.display()
      }
      Some(HoverInfo::{ contents: label, span: tok.span })
    }
  }
}

///|
/// Resolve a goto-definition request.
///
/// Looks up the first declaration that matches the identifier under
/// `offset`. Returns `None` for non-identifier tokens or when no
/// matching definition is found in the same source. Full cross-module
/// resolution lands once the module system does.
pub fn goto_definition_at(source : String, offset : UInt) -> GotoTarget? {
  match token_at(source, offset) {
    None => None
    Some(tok) => {
      let kind = tok.kind
      if kind != @syntax.SyntaxKind::Ident &&
        kind != @syntax.SyntaxKind::UpperIdent {
        return None
      }
      let name = tok.span.source_text(source)
      // Walk the token stream for the first declaration that introduces
      // a matching name: `<name> :`, `<name> =`, `<name> x =`,
      // `data <Name>`, or `type <Name>`.
      let tokens = @parser.lex(source)
      let mut i = 0
      let n = tokens.length()
      while i < n {
        let t = tokens[i]
        if t.kind == @syntax.SyntaxKind::Eof {
          break
        }
        if (t.kind == @syntax.SyntaxKind::KwData ||
          t.kind == @syntax.SyntaxKind::KwType) &&
          i + 2 < n {
          let next = skip_trivia_from(tokens, i + 1)
          if next < n {
            let candidate = tokens[next]
            if candidate.kind == @syntax.SyntaxKind::UpperIdent &&
              candidate.span.source_text(source) == name {
              return Some(GotoTarget::{ span: candidate.span })
            }
          }
        }
        if t.kind == @syntax.SyntaxKind::Ident &&
          t.span.source_text(source) == name {
          let next = skip_trivia_from(tokens, i + 1)
          if next < n {
            let nk = tokens[next].kind
            if nk == @syntax.SyntaxKind::Colon ||
              nk == @syntax.SyntaxKind::Equals ||
              nk == @syntax.SyntaxKind::Ident ||
              nk == @syntax.SyntaxKind::Underscore {
              return Some(GotoTarget::{ span: t.span })
            }
          }
        }
        i = i + 1
      }
      None
    }
  }
}

///|
fn skip_trivia_from(tokens : Array[@parser.Token], from : Int) -> Int {
  let mut i = from
  let n = tokens.length()
  while i < n && tokens[i].kind.is_trivia() {
    i = i + 1
  }
  i
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
    } else if tok.kind == @syntax.SyntaxKind::LineComment ||
      tok.kind == @syntax.SyntaxKind::BlockComment {
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
