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
/// Returns three groups, in this order: language keywords, in-scope value
/// bindings (builtins + top-level definitions in `source`), and known
/// type constructors (data types + constructors). Each value carries a
/// `detail` line with its pretty-printed type so the editor can show the
/// signature in the completion popup. The client is responsible for
/// filtering by prefix.
pub fn completions_at(
  source : String,
  _offset : UInt
) -> Array[CompletionItem] {
  let items : Array[CompletionItem] = Array::new()
  for kw in keyword_completions() {
    items.push(CompletionItem::new(kw, Keyword))
  }
  let analyzer = analyze(source)
  let env = analyzer.env()
  // Bindings: builtins + top-level definitions. Classify as Function when
  // the scheme's underlying type is an arrow, Variable otherwise. Skip
  // constructors and operators here — they get dedicated handling below.
  env.bindings.each(fn(name, scheme) {
    if is_operator_name(name) {
      return
    }
    if analyzer.constructors().get(name) is Some(_) {
      return
    }
    let kind = if is_arrow_scheme(scheme) { Function } else { Variable }
    let detail = ": " + pretty_scheme(scheme)
    items.push(
      CompletionItem::{
        label: name,
        kind,
        detail: Some(detail),
        documentation: None,
        snippet: None,
      },
    )
  })
  // Constructors registered by `data` declarations and the builtin
  // prelude (Some / None / Ok / Err / Pair).
  analyzer.constructors().each(fn(name, info) {
    let scheme = constructor_scheme(info)
    let detail = "constructor of " + info.type_name + " : " + pretty_scheme(scheme)
    items.push(
      CompletionItem::{
        label: name,
        kind: Constructor,
        detail: Some(detail),
        documentation: None,
        snippet: None,
      },
    )
  })
  // Data type names (Option, Result, Pair, plus any `data` decls).
  analyzer.data_types().each(fn(name, info) {
    let detail = "data type" +
      (if info.type_params.length() == 0 {
        ""
      } else {
        " of arity " + info.type_params.length().to_string()
      })
    items.push(
      CompletionItem::{
        label: name,
        kind: Type,
        detail: Some(detail),
        documentation: None,
        snippet: None,
      },
    )
  })
  items
}

///|
/// The reserved-word completion seed. Exposed so tests and downstream
/// tooling can assert the exact list without duplicating the literal.
pub fn keyword_completions() -> Array[String] {
  [
    "let", "in", "where", "data", "type", "case", "of", "match", "if",
    "then", "else", "do", "forall", "infixl", "infixr", "infix", "deriving",
    "class", "instance", "import", "module",
  ]
}

///|
/// Build the function type a constructor takes when used as a value
/// (`Some : a -> Option a`, `None : Option a`, `Pair : a -> b -> Pair a b`).
fn constructor_scheme(
  info : @typechecker.ConstructorInfo
) -> @typechecker.Scheme {
  let body = match info.fields {
    Empty => info.result_ty
    Positional(field_tys) => arrow_fold_constructors(field_tys, info.result_ty)
    RecordFields(record_fields) => {
      let tys : Array[@typechecker.Ty] = Array::new()
      for r in record_fields {
        let (_, ty) = r
        tys.push(ty)
      }
      arrow_fold_constructors(tys, info.result_ty)
    }
  }
  @typechecker.Scheme::poly(info.scheme_vars, body)
}

///|
fn arrow_fold_constructors(
  fields : Array[@typechecker.Ty],
  result_ty : @typechecker.Ty
) -> @typechecker.Ty {
  let mut ty = result_ty
  let mut i = fields.length() - 1
  while i >= 0 {
    ty = @typechecker.Ty::arrow(fields[i], ty)
    i = i - 1
  }
  ty
}

///|
fn is_arrow_scheme(scheme : @typechecker.Scheme) -> Bool {
  match scheme.ty {
    Arrow(_, _) => true
    Forall(_, inner) =>
      match inner {
        Arrow(_, _) => true
        _ => false
      }
    _ => false
  }
}

///|
fn is_operator_name(name : String) -> Bool {
  // The semantic analyser registers binary operators (`+`, `-`, `==`,
  // `&&`, etc.) directly in the type env. Skip them in identifier
  // completions — editors don't want them in the same dropdown.
  if name.length() == 0 {
    return false
  }
  let c = name.unsafe_charcode_at(0)
  let is_letter = (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A)
  let is_dollar = c == 0x24
  let is_underscore = c == 0x5F
  !(is_letter || is_dollar || is_underscore)
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
/// For identifiers, the contents are the binding's inferred type scheme
/// (`<name> : <type>`) looked up in the semantic environment after a full
/// parse + analyse pass. Tokens that don't carry a binding (keywords,
/// literals, operators, comments) fall back to a brief kind-aware label.
pub fn hover_at(source : String, offset : UInt) -> HoverInfo? {
  match token_at(source, offset) {
    None => None
    Some(tok) => {
      let text = tok.span.source_text(source)
      let typed = if tok.kind == @syntax.SyntaxKind::Ident ||
        tok.kind == @syntax.SyntaxKind::UpperIdent {
        lookup_type(source, text)
      } else {
        None
      }
      let label = match typed {
        Some(ty_str) => text + " : " + ty_str
        None => default_hover_label(tok, text)
      }
      Some(HoverInfo::{ contents: label, span: tok.span })
    }
  }
}

///|
fn default_hover_label(tok : @parser.Token, text : String) -> String {
  if tok.kind.is_keyword() {
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
}

///|
/// Parse `source`, run the semantic analyser, look `name` up in the type
/// environment, and pretty-print the scheme. Returns `None` if the name
/// isn't bound — falls back to the kind-aware label in that case.
fn lookup_type(source : String, name : String) -> String? {
  let analyzer = analyze(source)
  match analyzer.env().lookup(name) {
    Some(scheme) => Some(pretty_scheme(scheme))
    None => None
  }
}

///|
/// Render a `Scheme` for hover. Drops the `forall a b.` prefix because
/// the type variables already show up inline (and hover panes are short).
pub fn pretty_scheme(scheme : @typechecker.Scheme) -> String {
  pretty_type(scheme.ty)
}

///|
/// Render a `Ty` in the right-associative arrow form (`a -> b -> c`)
/// instead of `Ty::display`'s fully-parenthesised `(a -> (b -> c))`.
///
/// Type variables come out as letters (`a`, `b`, ...) when there are at
/// most 26 of them — much friendlier in a hover popup than `t12`.
pub fn pretty_type(ty : @typechecker.Ty) -> String {
  let env = TyVarNames::new()
  render_ty(ty, 0, env)
}

///|
priv struct TyVarNames {
  names : Map[UInt, String]
  mut next : Int
}

///|
fn TyVarNames::new() -> TyVarNames {
  TyVarNames::{ names: Map::new(), next: 0 }
}

///|
fn TyVarNames::name_for(self : TyVarNames, v : UInt) -> String {
  match self.names.get(v) {
    Some(s) => s
    None => {
      let s = letter_name(self.next)
      self.names[v] = s
      self.next = self.next + 1
      s
    }
  }
}

///|
fn letter_name(i : Int) -> String {
  if i < 26 {
    let letters = "abcdefghijklmnopqrstuvwxyz"
    letters.substring(start=i, end=i + 1)
  } else {
    "t" + i.to_string()
  }
}

///|
/// `prec` controls when we add parentheses:
///   0 = top level
///   1 = right-hand side of an arrow (no parens needed for `a -> b -> c`)
///   2 = left-hand side of an arrow (parenthesise nested arrows)
///   3 = type-application argument (parenthesise everything compound)
fn render_ty(
  ty : @typechecker.Ty,
  prec : Int,
  env : TyVarNames
) -> String {
  match ty {
    Var(v) => env.name_for(v)
    Con(name) => name
    Nat(n) => n.to_string()
    Error => "<error>"
    Arrow(from, to) => {
      let inner = render_ty(from, 2, env) +
        " -> " +
        render_ty(to, 1, env)
      if prec >= 2 {
        "(" + inner + ")"
      } else {
        inner
      }
    }
    App(f, a) => {
      let inner = render_ty(f, 2, env) + " " + render_ty(a, 3, env)
      if prec >= 3 {
        "(" + inner + ")"
      } else {
        inner
      }
    }
    Forall(_, body) => render_ty(body, prec, env)
  }
}

///|
fn analyze(source : String) -> @semantic.SemanticAnalyzer {
  let parser = @parser.Parser::new(source)
  let program = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(program)
  analyzer
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
