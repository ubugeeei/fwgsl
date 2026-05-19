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
    letters.unsafe_substring(start=i, end=i + 1)
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
/// A single edit that an editor should apply to a source buffer.
pub(all) struct TextEdit {
  /// Byte offset (inclusive) where the edit starts.
  start : UInt
  /// Byte offset (exclusive) where the edit ends.
  end : UInt
  /// Replacement text. The empty string means "delete the range".
  new_text : String
} derive(Show, Eq)

///|
pub(all) enum CodeActionKind {
  /// A `quickfix.*` action — tied to a diagnostic that the action resolves.
  QuickFix
  /// A `refactor.*` action — improves the code without fixing an error.
  Refactor
} derive(Eq, Show)

///|
pub fn CodeActionKind::lsp_str(self : CodeActionKind) -> String {
  match self {
    QuickFix => "quickfix"
    Refactor => "refactor"
  }
}

///|
/// A code action the editor can offer the user. A single action may carry
/// multiple text edits, but the typical "add type signature" case has
/// exactly one.
pub(all) struct CodeAction {
  title : String
  kind : CodeActionKind
  edits : Array[TextEdit]
} derive(Show)

///|
/// Compute code actions available at byte offset `offset` in `source`.
///
/// First action: for a `FunDecl` under the cursor that lacks an explicit
/// `<name> : <type>` signature, propose inserting one above the decl
/// using the inferred type from the semantic env.
pub fn code_actions_at(source : String, offset : UInt) -> Array[CodeAction] {
  let actions : Array[CodeAction] = Array::new()
  let parser = @parser.Parser::new(source)
  let program = parser.parse_program()
  let analyzer = @semantic.SemanticAnalyzer::new()
  analyzer.analyze(program)
  let signed : Map[String, Bool] = Map::new()
  for decl in program.decls {
    match decl {
      TypeSig(name, _, _) => signed[name] = true
      _ => ()
    }
  }
  let env = analyzer.env()
  for decl in program.decls {
    match decl {
      FunDecl(name, _, _, _, span) =>
        if span_contains_offset(span, offset) && signed.get(name) is None {
          match env.lookup(name) {
            Some(scheme) =>
              actions.push(
                build_add_signature_action(name, scheme, span, source),
              )
            None => ()
          }
        }
      _ => ()
    }
  }
  actions
}

///|
fn span_contains_offset(span : @span.Span, offset : UInt) -> Bool {
  offset >= span.start && offset <= span.end
}

///|
/// Build a `quickfix.add-type-signature` action that inserts
/// `<name> : <type>\n` immediately before `decl_span.start`. The
/// insertion preserves the leading indentation of the decl line.
fn build_add_signature_action(
  name : String,
  scheme : @typechecker.Scheme,
  decl_span : @span.Span,
  source : String
) -> CodeAction {
  let indent = leading_indent_for(decl_span.start, source)
  let line = indent + name + " : " + pretty_scheme(scheme) + "\n"
  let edit = TextEdit::{
    start: decl_span.start - indent.length().reinterpret_as_uint(),
    end: decl_span.start - indent.length().reinterpret_as_uint(),
    new_text: line,
  }
  let edits : Array[TextEdit] = Array::new()
  edits.push(edit)
  CodeAction::{
    title: "Add type signature for `" + name + "`",
    kind: QuickFix,
    edits,
  }
}

///|
/// Walk back from `offset` to the previous newline (or the start of the
/// document) and return the run of horizontal whitespace immediately
/// after it — the indentation of the line `offset` sits on.
fn leading_indent_for(offset : UInt, source : String) -> String {
  let off = offset.reinterpret_as_int()
  let mut line_start = off
  while line_start > 0 {
    if source.unsafe_charcode_at(line_start - 1) == 0x0A {
      break
    }
    line_start = line_start - 1
  }
  let mut i = line_start
  while i < off {
    let c = source.unsafe_charcode_at(i)
    if c != 0x20 && c != 0x09 {
      break
    }
    i = i + 1
  }
  source.unsafe_substring(start=line_start, end=i)
}

///|
/// Compute semantic tokens for the entire document.
///
/// Classifies each token with the LSP semantic-token kind editors use to
/// drive themed highlighting:
///
/// - `TokKeyword` — language keywords (`let`, `data`, ...).
/// - `TokOperator` — punctuator operators (`+`, `==`, `->`, ...).
/// - `TokDecorator` — the `@` plus the identifier that follows
///   (`@compute`, `@workgroup_size`).
/// - `TokType` — `UpperIdent` (data types and constructors).
/// - `TokFunction` — `Ident` bound to an arrow type in the semantic env.
/// - `TokVariable` — every other `Ident`.
/// - `TokNumber` — int / float literals.
/// - `TokString` — string / char literals.
/// - `TokComment` — line / block comments.
///
/// Falls back to omitting the token when none of the above apply, so the
/// stream stays compact.
pub fn semantic_tokens(source : String) -> Array[SemanticToken] {
  let out : Array[SemanticToken] = Array::new()
  let tokens = @parser.lex(source)
  let env = analyze(source).env()
  let mut prev_non_trivia : @syntax.SyntaxKind? = None
  let mut i = 0
  let n = tokens.length()
  while i < n {
    let tok = tokens[i]
    if tok.kind == @syntax.SyntaxKind::Eof {
      break
    }
    if tok.kind.is_trivia() &&
      tok.kind != @syntax.SyntaxKind::LineComment &&
      tok.kind != @syntax.SyntaxKind::BlockComment {
      i = i + 1
      continue
    }
    let classified = classify_semantic_token(
      tok,
      source,
      env,
      prev_non_trivia,
    )
    match classified {
      Some(k) => out.push(SemanticToken::{ kind: k, span: tok.span })
      None => ()
    }
    prev_non_trivia = Some(tok.kind)
    i = i + 1
  }
  out
}

///|
fn classify_semantic_token(
  tok : @parser.Token,
  source : String,
  env : @typechecker.TypeEnv,
  prev : @syntax.SyntaxKind?
) -> SemanticTokenKind? {
  let kind = tok.kind
  if kind == @syntax.SyntaxKind::At {
    return Some(TokDecorator)
  }
  if kind.is_keyword() {
    return Some(TokKeyword)
  }
  if kind.is_operator() {
    return Some(TokOperator)
  }
  if kind == @syntax.SyntaxKind::LineComment ||
    kind == @syntax.SyntaxKind::BlockComment {
    return Some(TokComment)
  }
  if kind == @syntax.SyntaxKind::IntLiteral || kind == @syntax.SyntaxKind::FloatLiteral {
    return Some(TokNumber)
  }
  if kind == @syntax.SyntaxKind::StringLiteral || kind == @syntax.SyntaxKind::CharLiteral {
    return Some(TokString)
  }
  if kind == @syntax.SyntaxKind::UpperIdent {
    return Some(TokType)
  }
  if kind == @syntax.SyntaxKind::Ident {
    // `@<name>` attribute lookups: the identifier immediately after an
    // `@` is part of the decorator.
    match prev {
      Some(At) => return Some(TokDecorator)
      _ => ()
    }
    let name = tok.span.source_text(source)
    match env.lookup(name) {
      Some(scheme) =>
        if is_arrow_scheme(scheme) {
          return Some(TokFunction)
        } else {
          return Some(TokVariable)
        }
      None => return Some(TokVariable)
    }
  }
  None
}
