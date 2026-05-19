///|
/// Recursive descent parser for fwgsl.
///
/// Produces a simple AST. Expression parsing uses Pratt (precedence climbing).

///|
pub struct Parser {
  tokens : Array[Token]
  mut pos : Int
  mut fuel : Int
  diagnostics : @diagnostics.DiagnosticSink
  source : String
}

///|
const MAX_FUEL = 10000

///|
/// Create a new parser from source text. Lexes and resolves layout.
pub fn Parser::new(source : String) -> Parser {
  let raw_tokens = lex(source)
  let tokens = resolve_layout(raw_tokens, source)
  Parser::{
    tokens,
    pos: 0,
    fuel: MAX_FUEL,
    diagnostics: @diagnostics.DiagnosticSink::new(),
    source,
  }
}

///|
/// Return the diagnostics accumulated during parsing.
pub fn Parser::diagnostics(self : Parser) -> @diagnostics.DiagnosticSink {
  self.diagnostics
}

// -- Navigation helpers ----------------------------------------------------

///|
fn Parser::current_token(self : Parser) -> Token {
  let idx = if self.pos < self.tokens.length() {
    self.pos
  } else {
    self.tokens.length() - 1
  }
  self.tokens[idx]
}

///|
fn Parser::peek(self : Parser) -> @syntax.SyntaxKind {
  self.current_token().kind
}

///|
fn Parser::at(self : Parser, kind : @syntax.SyntaxKind) -> Bool {
  self.peek() == kind
}

///|
fn Parser::at_end(self : Parser) -> Bool {
  self.peek() == @syntax.SyntaxKind::Eof
}

///|
fn Parser::bump(self : Parser) -> Token {
  let tok = self.current_token()
  if not(self.at_end()) {
    self.pos = self.pos + 1
  }
  tok
}

///|
fn Parser::expect(self : Parser, kind : @syntax.SyntaxKind) -> Token {
  if self.at(kind) {
    self.bump()
  } else {
    let tok = self.current_token()
    let msg = "expected " + kind.display() + ", found " + tok.kind.display()
    self.diagnostics.push(
      @diagnostics.Diagnostic::error(msg).with_label(
        @diagnostics.Label::primary(tok.span, "expected " + kind.display()),
      ),
    )
    tok
  }
}

///|
fn Parser::eat(self : Parser, kind : @syntax.SyntaxKind) -> Bool {
  if self.at(kind) {
    let _ = self.bump()
    true
  } else {
    false
  }
}

///|
fn Parser::skip_trivia(self : Parser) -> Unit {
  while self.peek().is_trivia() {
    let _ = self.bump()
  }
}

///|
fn Parser::peek_non_trivia(self : Parser) -> @syntax.SyntaxKind {
  let mut i = self.pos
  let len = self.tokens.length()
  while i < len {
    let kind = self.tokens[i].kind
    if not(kind.is_trivia()) {
      return kind
    }
    i = i + 1
  }
  @syntax.SyntaxKind::Eof
}

///|
fn Parser::span_from(self : Parser, start : UInt) -> @span.Span {
  let endv = if self.pos > 0 {
    self.tokens[self.pos - 1].span.end
  } else {
    start
  }
  @span.Span::new(start, endv)
}

///|
fn Parser::current_span(self : Parser) -> @span.Span {
  self.current_token().span
}

///|
fn Parser::text_of(self : Parser, tok : Token) -> String {
  tok.span.source_text(self.source)
}

///|
fn Parser::consume_fuel(self : Parser) -> Bool {
  if self.fuel == 0 {
    return false
  }
  self.fuel = self.fuel - 1
  true
}

// -- Layout helpers --------------------------------------------------------

///|
fn Parser::eat_layout_semi(self : Parser) -> Bool {
  self.skip_trivia()
  self.eat(@syntax.SyntaxKind::LayoutSemicolon)
}

///|
fn Parser::at_layout_end(self : Parser) -> Bool {
  let k = self.peek_non_trivia()
  k == @syntax.SyntaxKind::LayoutBraceClose || k == @syntax.SyntaxKind::Eof
}

///|
fn Parser::eat_layout_close(self : Parser) -> Bool {
  self.skip_trivia()
  self.eat(@syntax.SyntaxKind::LayoutBraceClose)
}

// =========================================================================
// Top level: Program
// =========================================================================

///|
pub fn Parser::parse_program(self : Parser) -> @ast.Program {
  let decls : Array[@ast.Decl] = Array::new()
  while true {
    self.skip_trivia()
    while self.eat(@syntax.SyntaxKind::LayoutSemicolon) ||
      self.eat(@syntax.SyntaxKind::LayoutBraceClose) ||
      self.eat(@syntax.SyntaxKind::LayoutBraceOpen) {
      self.skip_trivia()
    }
    if self.at_end() {
      break
    }
    if not(self.consume_fuel()) {
      break
    }
    match self.parse_decl() {
      Some(decl) => decls.push(decl)
      None => {
        // Error recovery: skip one token
        let _ = self.bump()
      }
    }
  }
  @ast.Program::new(decls)
}

// =========================================================================
// Declarations
// =========================================================================

///|
fn Parser::parse_decl(self : Parser) -> @ast.Decl? {
  self.skip_trivia()
  match self.peek_non_trivia() {
    At => Some(self.parse_attributed_decl())
    KwData => Some(self.parse_data_decl())
    KwType => Some(self.parse_type_alias())
    Ident => {
      self.skip_trivia()
      let name_tok = self.bump()
      let name = self.text_of(name_tok)
      let start = name_tok.span.start
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::Colon) {
        Some(self.parse_type_sig(name, start))
      } else {
        Some(self.parse_fun_decl(name, start))
      }
    }
    _ => None
  }
}

///|
fn Parser::parse_attributed_decl(self : Parser) -> @ast.Decl {
  let start = self.current_span().start
  let attributes : Array[@ast.Attribute] = Array::new()
  while self.peek_non_trivia() == @syntax.SyntaxKind::At {
    self.skip_trivia()
    attributes.push(self.parse_attribute())
    self.skip_trivia()
  }
  let _ = self.eat_layout_semi()
  self.skip_trivia()
  if self.at(@syntax.SyntaxKind::Ident) {
    let saved = self.pos
    let name_tok = self.bump()
    self.skip_trivia()
    if self.at(@syntax.SyntaxKind::Colon) {
      let name = self.text_of(name_tok)
      let _ = self.parse_type_sig(name, name_tok.span.start)
      let _ = self.eat_layout_semi()
      self.skip_trivia()
    } else {
      self.pos = saved
    }
  }
  self.skip_trivia()
  let name_tok = self.expect(@syntax.SyntaxKind::Ident)
  let name = self.text_of(name_tok)
  self.skip_trivia()
  let params : Array[@ast.Pat] = Array::new()
  while not(self.at(@syntax.SyntaxKind::Equals)) &&
    not(self.at_end()) &&
    self.consume_fuel() {
    let k = self.peek_non_trivia()
    if k == @syntax.SyntaxKind::LayoutSemicolon ||
      k == @syntax.SyntaxKind::LayoutBraceClose {
      break
    }
    let p = self.parse_pat_atom()
    params.push(p)
    self.skip_trivia()
  }
  let _ = self.expect(@syntax.SyntaxKind::Equals)
  self.skip_trivia()
  let body = self.parse_expr()
  let span = self.span_from(start)
  @ast.Decl::EntryPoint(attributes, name, params, body, span)
}

///|
fn Parser::parse_type_sig(
  self : Parser,
  name : String,
  start : UInt
) -> @ast.Decl {
  let _ = self.expect(@syntax.SyntaxKind::Colon)
  self.skip_trivia()
  let ty = self.parse_type()
  let span = self.span_from(start)
  @ast.Decl::TypeSig(name, ty, span)
}

///|
fn Parser::parse_fun_decl(
  self : Parser,
  name : String,
  start : UInt
) -> @ast.Decl {
  let params : Array[@ast.Pat] = Array::new()
  self.skip_trivia()
  while not(self.at(@syntax.SyntaxKind::Equals)) &&
    not(self.at(@syntax.SyntaxKind::Pipe)) &&
    not(self.at_end()) &&
    self.consume_fuel() {
    let k = self.peek_non_trivia()
    if k == @syntax.SyntaxKind::LayoutSemicolon ||
      k == @syntax.SyntaxKind::LayoutBraceClose {
      break
    }
    let p = self.parse_pat_atom()
    params.push(p)
    self.skip_trivia()
  }
  if self.at(@syntax.SyntaxKind::Pipe) {
    let body = self.parse_guard_clauses(start)
    let where_binds : Array[(String, @ast.Expr)] = Array::new()
    self.skip_trivia()
    if self.eat(@syntax.SyntaxKind::KwWhere) {
      self.parse_where_binds_into(where_binds)
    }
    let span = self.span_from(start)
    return @ast.Decl::FunDecl(name, params, body, where_binds, span)
  }
  let _ = self.expect(@syntax.SyntaxKind::Equals)
  self.skip_trivia()
  let body = self.parse_expr()
  let where_binds : Array[(String, @ast.Expr)] = Array::new()
  self.skip_trivia()
  if self.eat(@syntax.SyntaxKind::KwWhere) {
    self.parse_where_binds_into(where_binds)
  }
  let span = self.span_from(start)
  @ast.Decl::FunDecl(name, params, body, where_binds, span)
}

///|
/// Parse guard clauses and desugar into nested if-then-else.
fn Parser::parse_guard_clauses(self : Parser, start : UInt) -> @ast.Expr {
  let guards : Array[(@ast.Expr, @ast.Expr)] = Array::new()
  while self.at(@syntax.SyntaxKind::Pipe) && self.consume_fuel() {
    let _ = self.bump() // consume `|`
    self.skip_trivia()
    let cond = self.parse_expr()
    self.skip_trivia()
    let _ = self.expect(@syntax.SyntaxKind::Equals)
    self.skip_trivia()
    let body = self.parse_expr()
    guards.push((cond, body))
    self.skip_trivia()
    let _ = self.eat_layout_semi()
    self.skip_trivia()
  }
  if guards.length() == 0 {
    let span = self.span_from(start)
    return @ast.Expr::Var("<error>", span)
  }
  let mut result : @ast.Expr? = None
  let mut i = guards.length() - 1
  while i >= 0 {
    let (cond, body) = guards[i]
    let span = cond.span().merge(body.span())
    let is_otherwise = match cond {
      Var(n, _) => n == "otherwise"
      _ => false
    }
    if is_otherwise {
      result = Some(body)
    } else {
      let else_branch = match result {
        Some(e) => e
        None => @ast.Expr::Lit(@ast.Lit::Int(0), span)
      }
      result = Some(@ast.Expr::If(cond, body, else_branch, span))
    }
    i = i - 1
  }
  match result {
    Some(e) => e
    None => @ast.Expr::Var("<error>", self.span_from(start))
  }
}

///|
fn Parser::parse_where_binds_into(
  self : Parser,
  binds : Array[(String, @ast.Expr)]
) -> Unit {
  self.skip_trivia()
  let _ = self.eat(@syntax.SyntaxKind::LayoutBraceOpen)
  while true {
    self.skip_trivia()
    if self.at_layout_end() || self.at_end() {
      break
    }
    if not(self.consume_fuel()) {
      break
    }
    if self.at(@syntax.SyntaxKind::Ident) {
      let name_tok = self.bump()
      let name = self.text_of(name_tok)
      self.skip_trivia()
      let _ = self.expect(@syntax.SyntaxKind::Equals)
      self.skip_trivia()
      let expr = self.parse_expr()
      binds.push((name, expr))
      let _ = self.eat_layout_semi()
    } else {
      break
    }
  }
  let _ = self.eat_layout_close()
}

///|
fn Parser::parse_data_decl(self : Parser) -> @ast.Decl {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::KwData)
  self.skip_trivia()
  let name_tok = self.expect(@syntax.SyntaxKind::UpperIdent)
  let name = self.text_of(name_tok)
  let type_params : Array[String] = Array::new()
  self.skip_trivia()
  while self.at(@syntax.SyntaxKind::Ident) {
    let p = self.bump()
    type_params.push(self.text_of(p))
    self.skip_trivia()
  }
  let _ = self.expect(@syntax.SyntaxKind::Equals)
  self.skip_trivia()
  let constructors : Array[@ast.ConDecl] = Array::new()
  constructors.push(self.parse_con_decl())
  while true {
    self.skip_trivia()
    if self.eat(@syntax.SyntaxKind::Pipe) {
      self.skip_trivia()
      constructors.push(self.parse_con_decl())
    } else {
      break
    }
  }
  let span = self.span_from(start)
  @ast.Decl::DataDecl(name, type_params, constructors, span)
}

///|
fn Parser::parse_con_decl(self : Parser) -> @ast.ConDecl {
  let start = self.current_span().start
  let name_tok = self.expect(@syntax.SyntaxKind::UpperIdent)
  let name = self.text_of(name_tok)
  self.skip_trivia()
  let fields = if self.at(@syntax.SyntaxKind::LBrace) {
    let _ = self.bump()
    let flds : Array[(String, @ast.Type)] = Array::new()
    while true {
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::RBrace) || self.at_end() {
        break
      }
      let field_name_tok = self.expect(@syntax.SyntaxKind::Ident)
      let field_name = self.text_of(field_name_tok)
      self.skip_trivia()
      let _ = self.expect(@syntax.SyntaxKind::Colon)
      self.skip_trivia()
      let ty = self.parse_type_atom()
      flds.push((field_name, ty))
      self.skip_trivia()
      if not(self.eat(@syntax.SyntaxKind::Comma)) {
        break
      }
    }
    let _ = self.expect(@syntax.SyntaxKind::RBrace)
    @ast.ConFields::RecordFields(flds)
  } else {
    let tys : Array[@ast.Type] = Array::new()
    while not(self.at_end()) && self.consume_fuel() {
      let k = self.peek_non_trivia()
      let stop = k == @syntax.SyntaxKind::Pipe ||
        k == @syntax.SyntaxKind::Eof ||
        k == @syntax.SyntaxKind::LayoutSemicolon ||
        k == @syntax.SyntaxKind::LayoutBraceClose ||
        k == @syntax.SyntaxKind::KwData ||
        k == @syntax.SyntaxKind::KwWhere ||
        k == @syntax.SyntaxKind::Newline
      if stop {
        break
      }
      let accept = k == @syntax.SyntaxKind::UpperIdent ||
        k == @syntax.SyntaxKind::Ident ||
        k == @syntax.SyntaxKind::LParen
      if not(accept) {
        break
      }
      self.skip_trivia()
      let ty = self.parse_type_atom()
      tys.push(ty)
    }
    if tys.length() == 0 {
      @ast.ConFields::Empty
    } else {
      @ast.ConFields::Positional(tys)
    }
  }
  let span = self.span_from(start)
  @ast.ConDecl::new(name, fields, span)
}

///|
fn Parser::parse_type_alias(self : Parser) -> @ast.Decl {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::KwType)
  self.skip_trivia()
  let name_tok = self.expect(@syntax.SyntaxKind::UpperIdent)
  let name = self.text_of(name_tok)
  let params : Array[String] = Array::new()
  self.skip_trivia()
  while self.at(@syntax.SyntaxKind::Ident) {
    let p = self.bump()
    params.push(self.text_of(p))
    self.skip_trivia()
  }
  let _ = self.expect(@syntax.SyntaxKind::Equals)
  self.skip_trivia()
  let ty = self.parse_type()
  let span = self.span_from(start)
  @ast.Decl::TypeAlias(name, params, ty, span)
}

///|
fn Parser::parse_attribute(self : Parser) -> @ast.Attribute {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::At)
  self.skip_trivia()
  let name_tok = if self.at(@syntax.SyntaxKind::Ident) {
    self.bump()
  } else {
    self.expect(@syntax.SyntaxKind::UpperIdent)
  }
  let name = self.text_of(name_tok)
  let args : Array[String] = Array::new()
  self.skip_trivia()
  if self.at(@syntax.SyntaxKind::LParen) {
    let _ = self.bump()
    while true {
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::RParen) || self.at_end() {
        break
      }
      let arg_tok = self.bump()
      args.push(self.text_of(arg_tok))
      self.skip_trivia()
      if not(self.eat(@syntax.SyntaxKind::Comma)) {
        break
      }
    }
    let _ = self.expect(@syntax.SyntaxKind::RParen)
  } else {
    while is_int_or_float_literal(self.peek_non_trivia()) {
      self.skip_trivia()
      let arg_tok = self.bump()
      args.push(self.text_of(arg_tok))
    }
  }
  let span = self.span_from(start)
  @ast.Attribute::new(name, args, span)
}

///|
fn is_int_or_float_literal(k : @syntax.SyntaxKind) -> Bool {
  k == @syntax.SyntaxKind::IntLiteral || k == @syntax.SyntaxKind::FloatLiteral
}

// =========================================================================
// Expressions (Pratt / precedence climbing)
// =========================================================================

///|
pub fn Parser::parse_expr(self : Parser) -> @ast.Expr {
  self.parse_expr_bp(0)
}

///|
fn Parser::parse_expr_bp(self : Parser, min_bp : Int) -> @ast.Expr {
  self.skip_trivia()
  let mut lhs = match self.peek_non_trivia() {
    Minus => {
      self.skip_trivia()
      let start = self.current_span().start
      let _ = self.bump()
      self.skip_trivia()
      let rhs = self.parse_expr_bp(13)
      let span = self.span_from(start)
      @ast.Expr::Neg(rhs, span)
    }
    Backslash => self.parse_lambda()
    KwIf => self.parse_if()
    KwMatch => self.parse_match()
    KwLet => self.parse_let()
    KwDo => self.parse_do()
    _ => self.parse_atom()
  }
  while true {
    if not(self.consume_fuel()) {
      break
    }
    self.skip_trivia()
    let op_kind = self.peek_non_trivia()

    // Function application via juxtaposition
    if is_atom_start(op_kind) && min_bp <= 11 {
      self.skip_trivia()
      let arg = self.parse_atom()
      let span = lhs.span().merge(arg.span())
      lhs = @ast.Expr::App(lhs, arg, span)
      continue
    }

    // Field access
    if op_kind == @syntax.SyntaxKind::Dot && min_bp <= 13 {
      self.skip_trivia()
      let _ = self.bump()
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::Ident) {
        let field_tok = self.bump()
        let field = self.text_of(field_tok)
        let span = lhs.span().merge(field_tok.span)
        lhs = @ast.Expr::FieldAccess(lhs, field, span)
        continue
      } else {
        let tok = self.current_token()
        self.diagnostics.push(
          @diagnostics.Diagnostic::error("expected field name after '.'")
            .with_label(@diagnostics.Label::primary(tok.span, "expected identifier"))
            .with_help("write a field access like `value.field`"),
        )
        break
      }
    }

    // Binary operators
    match infix_binding_power(op_kind) {
      Some((l_bp, r_bp)) => {
        if l_bp < min_bp {
          break
        }
        self.skip_trivia()
        let op_tok = self.bump()
        let op_text = self.text_of(op_tok)
        self.skip_trivia()
        let rhs = self.parse_expr_bp(r_bp)
        let span = lhs.span().merge(rhs.span())
        lhs = @ast.Expr::Infix(lhs, op_text, rhs, span)
        continue
      }
      None => ()
    }

    // Backtick infix: expr `func` expr
    if op_kind == @syntax.SyntaxKind::Backtick && min_bp <= 3 {
      self.skip_trivia()
      let _ = self.bump()
      self.skip_trivia()
      let func_tok = self.bump()
      let func = self.text_of(func_tok)
      self.skip_trivia()
      let _ = self.expect(@syntax.SyntaxKind::Backtick)
      self.skip_trivia()
      let rhs = self.parse_expr_bp(4)
      let span = lhs.span().merge(rhs.span())
      lhs = @ast.Expr::Infix(lhs, func, rhs, span)
      continue
    }

    // Pipeline operator: x |> f desugars to f x
    if op_kind == @syntax.SyntaxKind::PipeForward && min_bp <= 1 {
      self.skip_trivia()
      let _ = self.bump()
      self.skip_trivia()
      let func = self.parse_expr_bp(2)
      let span = lhs.span().merge(func.span())
      lhs = @ast.Expr::App(func, lhs, span)
      continue
    }
    break
  }
  lhs
}

///|
fn Parser::parse_atom(self : Parser) -> @ast.Expr {
  self.skip_trivia()
  match self.peek() {
    IntLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      let val = parse_int_literal(text)
      @ast.Expr::Lit(@ast.Lit::Int(val), tok.span)
    }
    FloatLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      let val = parse_float_literal(text)
      @ast.Expr::Lit(@ast.Lit::Float(val), tok.span)
    }
    StringLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      let inner = text.substring(start=1, end=text.length() - 1)
      @ast.Expr::Lit(@ast.Lit::StringLit(unescape_string(inner)), tok.span)
    }
    CharLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      let inner = text.substring(start=1, end=text.length() - 1)
      let ch = unescape_char(inner)
      @ast.Expr::Lit(@ast.Lit::Char(ch), tok.span)
    }
    Ident => {
      let tok = self.bump()
      let name = self.text_of(tok)
      @ast.Expr::Var(name, tok.span)
    }
    UpperIdent => {
      let tok = self.bump()
      let name = self.text_of(tok)
      @ast.Expr::Con(name, tok.span)
    }
    LParen => self.parse_paren_expr()
    LBracket => self.parse_vec_lit()
    KwIf => self.parse_if()
    KwMatch => self.parse_match()
    KwLet => self.parse_let()
    KwDo => self.parse_do()
    Backslash => self.parse_lambda()
    _ => {
      let tok = self.current_token()
      self.diagnostics.push(
        @diagnostics.Diagnostic::error("unexpected token: " + tok.kind.display())
          .with_label(@diagnostics.Label::primary(tok.span, "unexpected")),
      )
      let span = tok.span
      let _ = self.bump()
      @ast.Expr::Var("<error>", span)
    }
  }
}

///|
fn Parser::parse_paren_expr(self : Parser) -> @ast.Expr {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::LParen)
  self.skip_trivia()
  if self.at(@syntax.SyntaxKind::RParen) {
    let _ = self.bump()
    let span = self.span_from(start)
    return @ast.Expr::Tuple(Array::new(), span)
  }
  if is_operator_token(self.peek()) {
    let op_tok = self.bump()
    self.skip_trivia()
    if self.at(@syntax.SyntaxKind::RParen) {
      let _ = self.bump()
      let op = self.text_of(op_tok)
      let span = self.span_from(start)
      return @ast.Expr::OpSection(op, span)
    }
    let op = self.text_of(op_tok)
    while not(self.at(@syntax.SyntaxKind::RParen)) &&
      not(self.at_end()) &&
      self.consume_fuel() {
      let _ = self.bump()
    }
    let _ = self.eat(@syntax.SyntaxKind::RParen)
    let span = self.span_from(start)
    return @ast.Expr::OpSection(op, span)
  }
  let first = self.parse_expr()
  self.skip_trivia()
  if self.at(@syntax.SyntaxKind::Comma) {
    let elems : Array[@ast.Expr] = [first]
    while self.eat(@syntax.SyntaxKind::Comma) {
      self.skip_trivia()
      elems.push(self.parse_expr())
      self.skip_trivia()
    }
    let _ = self.expect(@syntax.SyntaxKind::RParen)
    let span = self.span_from(start)
    return @ast.Expr::Tuple(elems, span)
  }
  let _ = self.expect(@syntax.SyntaxKind::RParen)
  let span = self.span_from(start)
  @ast.Expr::Paren(first, span)
}

///|
fn Parser::parse_vec_lit(self : Parser) -> @ast.Expr {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::LBracket)
  self.skip_trivia()
  let elems : Array[@ast.Expr] = Array::new()
  if not(self.at(@syntax.SyntaxKind::RBracket)) && not(self.at_end()) {
    elems.push(self.parse_expr())
    self.skip_trivia()
    while self.eat(@syntax.SyntaxKind::Comma) {
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::RBracket) {
        break
      }
      elems.push(self.parse_expr())
      self.skip_trivia()
    }
  }
  let _ = self.expect(@syntax.SyntaxKind::RBracket)
  let span = self.span_from(start)
  @ast.Expr::VecLit(elems, span)
}

///|
fn Parser::parse_if(self : Parser) -> @ast.Expr {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::KwIf)
  self.skip_trivia()
  let cond = self.parse_expr()
  self.skip_trivia()
  let _ = self.expect(@syntax.SyntaxKind::KwThen)
  self.skip_trivia()
  let then_expr = self.parse_expr()
  self.skip_trivia()
  let _ = self.expect(@syntax.SyntaxKind::KwElse)
  self.skip_trivia()
  let else_expr = self.parse_expr()
  let span = self.span_from(start)
  @ast.Expr::If(cond, then_expr, else_expr, span)
}

///|
fn Parser::parse_match(self : Parser) -> @ast.Expr {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::KwMatch)
  self.skip_trivia()
  let scrutinee = self.parse_match_scrutinee()
  self.skip_trivia()
  let arms : Array[(@ast.Pat, @ast.Expr)] = Array::new()
  while true {
    self.skip_trivia()
    if self.at_end() {
      break
    }
    if not(self.consume_fuel()) {
      break
    }
    if not(self.at(@syntax.SyntaxKind::Pipe)) {
      break
    }
    let _ = self.bump()
    self.skip_trivia()
    let pat = self.parse_pat()
    self.skip_trivia()
    let _ = self.expect(@syntax.SyntaxKind::Arrow)
    self.skip_trivia()
    let body = self.parse_expr()
    arms.push((pat, body))
  }
  let span = self.span_from(start)
  @ast.Expr::Case(scrutinee, arms, span)
}

///|
fn Parser::parse_match_scrutinee(self : Parser) -> @ast.Expr {
  self.skip_trivia()
  let mut expr = self.parse_match_scrutinee_atom()
  while true {
    self.skip_trivia()
    let k = self.peek_non_trivia()
    if k == @syntax.SyntaxKind::Pipe ||
      k == @syntax.SyntaxKind::Eof ||
      k == @syntax.SyntaxKind::LayoutBraceOpen ||
      k == @syntax.SyntaxKind::LayoutSemicolon ||
      k == @syntax.SyntaxKind::LayoutBraceClose {
      break
    }
    if not(is_atom_start(k)) {
      break
    }
    if not(self.consume_fuel()) {
      break
    }
    self.skip_trivia()
    let arg = self.parse_match_scrutinee_atom()
    let span = expr.span().merge(arg.span())
    expr = @ast.Expr::App(expr, arg, span)
  }
  expr
}

///|
fn Parser::parse_match_scrutinee_atom(self : Parser) -> @ast.Expr {
  self.skip_trivia()
  match self.peek() {
    Ident => {
      let tok = self.bump()
      @ast.Expr::Var(self.text_of(tok), tok.span)
    }
    UpperIdent => {
      let tok = self.bump()
      @ast.Expr::Con(self.text_of(tok), tok.span)
    }
    IntLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      @ast.Expr::Lit(@ast.Lit::Int(parse_int_literal(text)), tok.span)
    }
    LParen => self.parse_paren_expr()
    _ => {
      let tok = self.current_token()
      @ast.Expr::Var("<error>", tok.span)
    }
  }
}

///|
fn Parser::parse_let(self : Parser) -> @ast.Expr {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::KwLet)
  self.skip_trivia()
  let _ = self.eat(@syntax.SyntaxKind::LayoutBraceOpen)
  let binds : Array[(String, @ast.Expr)] = Array::new()
  while true {
    self.skip_trivia()
    if self.at(@syntax.SyntaxKind::KwIn) || self.at_layout_end() || self.at_end() {
      break
    }
    if not(self.consume_fuel()) {
      break
    }
    if self.at(@syntax.SyntaxKind::Ident) {
      let name_tok = self.bump()
      let name = self.text_of(name_tok)
      self.skip_trivia()
      let _ = self.expect(@syntax.SyntaxKind::Equals)
      self.skip_trivia()
      let expr = self.parse_expr()
      binds.push((name, expr))
      let _ = self.eat_layout_semi()
    } else {
      break
    }
  }
  let _ = self.eat_layout_close()
  self.skip_trivia()
  let _ = self.expect(@syntax.SyntaxKind::KwIn)
  self.skip_trivia()
  let body = self.parse_expr()
  let span = self.span_from(start)
  @ast.Expr::Let(binds, body, span)
}

///|
fn Parser::parse_do(self : Parser) -> @ast.Expr {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::KwDo)
  self.skip_trivia()
  let _ = self.eat(@syntax.SyntaxKind::LayoutBraceOpen)
  let stmts : Array[@ast.DoStmt] = Array::new()
  while true {
    self.skip_trivia()
    if self.at_layout_end() || self.at_end() {
      break
    }
    if not(self.consume_fuel()) {
      break
    }
    let stmt_start = self.current_span().start
    if self.at(@syntax.SyntaxKind::KwLet) {
      let _ = self.bump()
      self.skip_trivia()
      let _ = self.eat(@syntax.SyntaxKind::LayoutBraceOpen)
      self.skip_trivia()
      let name_tok = self.expect(@syntax.SyntaxKind::Ident)
      let name = self.text_of(name_tok)
      self.skip_trivia()
      let _ = self.expect(@syntax.SyntaxKind::Equals)
      self.skip_trivia()
      let expr = self.parse_expr()
      let span = self.span_from(stmt_start)
      stmts.push(@ast.DoStmt::LetStmt(name, expr, span))
      let _ = self.eat_layout_semi()
      let _ = self.eat_layout_close()
      continue
    }
    if self.at(@syntax.SyntaxKind::Ident) {
      let saved_pos = self.pos
      let name_tok = self.bump()
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::LeftArrow) {
        let _ = self.bump()
        self.skip_trivia()
        let name = self.text_of(name_tok)
        let expr = self.parse_expr()
        let span = self.span_from(stmt_start)
        stmts.push(@ast.DoStmt::Bind(name, expr, span))
        let _ = self.eat_layout_semi()
        continue
      } else {
        self.pos = saved_pos
      }
    }
    let expr = self.parse_expr()
    let span = self.span_from(stmt_start)
    stmts.push(@ast.DoStmt::ExprStmt(expr, span))
    let _ = self.eat_layout_semi()
  }
  let _ = self.eat_layout_close()
  let span = self.span_from(start)
  @ast.Expr::Do(stmts, span)
}

///|
fn Parser::parse_lambda(self : Parser) -> @ast.Expr {
  let start = self.current_span().start
  let _ = self.expect(@syntax.SyntaxKind::Backslash)
  self.skip_trivia()
  let params : Array[@ast.Pat] = Array::new()
  while not(self.at(@syntax.SyntaxKind::Arrow)) &&
    not(self.at_end()) &&
    self.consume_fuel() {
    let p = self.parse_pat_atom()
    params.push(p)
    self.skip_trivia()
  }
  let _ = self.expect(@syntax.SyntaxKind::Arrow)
  self.skip_trivia()
  let body = self.parse_expr()
  let span = self.span_from(start)
  @ast.Expr::Lambda(params, body, span)
}

// =========================================================================
// Patterns
// =========================================================================

///|
fn Parser::parse_pat(self : Parser) -> @ast.Pat {
  self.skip_trivia()
  let start = self.current_span().start
  match self.peek() {
    UpperIdent => {
      let name_tok = self.bump()
      let name = self.text_of(name_tok)
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::LBrace) {
        return self.parse_record_pat(name, start)
      }
      let sub_pats : Array[@ast.Pat] = Array::new()
      while is_pat_atom_start(self.peek_non_trivia()) && self.consume_fuel() {
        self.skip_trivia()
        sub_pats.push(self.parse_pat_atom())
      }
      let span = self.span_from(start)
      @ast.Pat::Con(name, sub_pats, span)
    }
    _ => self.parse_pat_atom()
  }
}

///|
fn Parser::parse_pat_atom(self : Parser) -> @ast.Pat {
  self.skip_trivia()
  let start = self.current_span().start
  match self.peek() {
    Underscore => {
      let tok = self.bump()
      @ast.Pat::Wild(tok.span)
    }
    Ident => {
      let tok = self.bump()
      let name = self.text_of(tok)
      if self.at(@syntax.SyntaxKind::At) {
        let _ = self.bump()
        self.skip_trivia()
        let inner = self.parse_pat_atom()
        let span = self.span_from(start)
        @ast.Pat::As(name, inner, span)
      } else {
        @ast.Pat::Var(name, tok.span)
      }
    }
    UpperIdent => {
      let tok = self.bump()
      @ast.Pat::Con(self.text_of(tok), Array::new(), tok.span)
    }
    IntLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      @ast.Pat::Lit(@ast.Lit::Int(parse_int_literal(text)), tok.span)
    }
    FloatLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      @ast.Pat::Lit(@ast.Lit::Float(parse_float_literal(text)), tok.span)
    }
    StringLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      let inner = text.substring(start=1, end=text.length() - 1)
      @ast.Pat::Lit(@ast.Lit::StringLit(unescape_string(inner)), tok.span)
    }
    CharLiteral => {
      let tok = self.bump()
      let text = self.text_of(tok)
      let inner = text.substring(start=1, end=text.length() - 1)
      @ast.Pat::Lit(@ast.Lit::Char(unescape_char(inner)), tok.span)
    }
    LParen => {
      let _ = self.bump()
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::RParen) {
        let _ = self.bump()
        let span = self.span_from(start)
        return @ast.Pat::Tuple(Array::new(), span)
      }
      let first = self.parse_pat()
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::Comma) {
        let elems : Array[@ast.Pat] = [first]
        while self.eat(@syntax.SyntaxKind::Comma) {
          self.skip_trivia()
          elems.push(self.parse_pat())
          self.skip_trivia()
        }
        let _ = self.expect(@syntax.SyntaxKind::RParen)
        let span = self.span_from(start)
        @ast.Pat::Tuple(elems, span)
      } else {
        let _ = self.expect(@syntax.SyntaxKind::RParen)
        let span = self.span_from(start)
        @ast.Pat::Paren(first, span)
      }
    }
    _ => {
      let tok = self.current_token()
      self.diagnostics.push(
        @diagnostics.Diagnostic::error(
          "unexpected token in pattern: " + tok.kind.display(),
        ).with_label(@diagnostics.Label::primary(tok.span, "unexpected")),
      )
      let span = tok.span
      let _ = self.bump()
      @ast.Pat::Wild(span)
    }
  }
}

///|
fn Parser::parse_record_pat(
  self : Parser,
  con_name : String,
  start : UInt
) -> @ast.Pat {
  let _ = self.expect(@syntax.SyntaxKind::LBrace)
  let fields : Array[(String, @ast.Pat?)] = Array::new()
  while true {
    self.skip_trivia()
    if self.at(@syntax.SyntaxKind::RBrace) || self.at_end() {
      break
    }
    let field_tok = self.expect(@syntax.SyntaxKind::Ident)
    let field_name = self.text_of(field_tok)
    self.skip_trivia()
    if self.at(@syntax.SyntaxKind::Equals) {
      let _ = self.bump()
      self.skip_trivia()
      let pat = self.parse_pat()
      fields.push((field_name, Some(pat)))
    } else {
      fields.push((field_name, None))
    }
    self.skip_trivia()
    if not(self.eat(@syntax.SyntaxKind::Comma)) {
      break
    }
  }
  let _ = self.expect(@syntax.SyntaxKind::RBrace)
  let span = self.span_from(start)
  @ast.Pat::Record(con_name, fields, span)
}

// =========================================================================
// Types
// =========================================================================

///|
pub fn Parser::parse_type(self : Parser) -> @ast.Type {
  self.skip_trivia()
  let lhs = self.parse_type_app()
  self.skip_trivia()
  if self.at(@syntax.SyntaxKind::Arrow) {
    let _ = self.bump()
    self.skip_trivia()
    let rhs = self.parse_type()
    let span = lhs.span().merge(rhs.span())
    @ast.Type::Arrow(lhs, rhs, span)
  } else {
    lhs
  }
}

///|
fn Parser::parse_type_app(self : Parser) -> @ast.Type {
  self.skip_trivia()
  let mut ty = self.parse_type_atom()
  while true {
    self.skip_trivia()
    let next = self.peek_non_trivia()
    let accept = next == @syntax.SyntaxKind::UpperIdent ||
      next == @syntax.SyntaxKind::Ident ||
      next == @syntax.SyntaxKind::IntLiteral ||
      next == @syntax.SyntaxKind::LParen
    if accept {
      let arg = self.parse_type_atom()
      let span = ty.span().merge(arg.span())
      ty = @ast.Type::App(ty, arg, span)
    } else {
      break
    }
  }
  ty
}

///|
fn Parser::parse_type_atom(self : Parser) -> @ast.Type {
  self.skip_trivia()
  let start = self.current_span().start
  match self.peek() {
    UpperIdent => {
      let tok = self.bump()
      @ast.Type::Con(self.text_of(tok), tok.span)
    }
    Ident => {
      let tok = self.bump()
      @ast.Type::Var(self.text_of(tok), tok.span)
    }
    IntLiteral => {
      let tok = self.bump()
      let v = parse_int_literal(self.text_of(tok))
      let value = if v < 0L { 0UL } else { v.reinterpret_as_uint64() }
      @ast.Type::Nat(value, tok.span)
    }
    LParen => {
      let _ = self.bump()
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::RParen) {
        let _ = self.bump()
        let span = self.span_from(start)
        return @ast.Type::Unit(span)
      }
      let first = self.parse_type()
      self.skip_trivia()
      if self.at(@syntax.SyntaxKind::Comma) {
        let elems : Array[@ast.Type] = [first]
        while self.eat(@syntax.SyntaxKind::Comma) {
          self.skip_trivia()
          elems.push(self.parse_type())
          self.skip_trivia()
        }
        let _ = self.expect(@syntax.SyntaxKind::RParen)
        let span = self.span_from(start)
        @ast.Type::Tuple(elems, span)
      } else {
        let _ = self.expect(@syntax.SyntaxKind::RParen)
        let span = self.span_from(start)
        @ast.Type::Paren(first, span)
      }
    }
    _ => {
      let tok = self.current_token()
      self.diagnostics.push(
        @diagnostics.Diagnostic::error(
          "unexpected token in type: " + tok.kind.display(),
        )
          .with_label(@diagnostics.Label::primary(tok.span, "unexpected"))
          .with_help("check the type expression syntax around this token"),
      )
      let span = tok.span
      let _ = self.bump()
      @ast.Type::Con("<error>", span)
    }
  }
}

// =========================================================================
// Helpers
// =========================================================================

///|
fn infix_binding_power(kind : @syntax.SyntaxKind) -> (Int, Int)? {
  match kind {
    Dollar => Some((1, 0))
    OrOr => Some((1, 2))
    AndAnd => Some((3, 4))
    EqualEqual | NotEqual | Less | Greater | LessEqual | GreaterEqual =>
      Some((5, 6))
    Plus | Minus => Some((7, 8))
    Star | Slash | Percent => Some((9, 10))
    _ => None
  }
}

///|
fn is_atom_start(kind : @syntax.SyntaxKind) -> Bool {
  match kind {
    IntLiteral
    | FloatLiteral
    | StringLiteral
    | CharLiteral
    | Ident
    | UpperIdent
    | LParen
    | LBracket
    | Backslash => true
    _ => false
  }
}

///|
fn is_pat_atom_start(kind : @syntax.SyntaxKind) -> Bool {
  match kind {
    Underscore
    | Ident
    | UpperIdent
    | IntLiteral
    | FloatLiteral
    | StringLiteral
    | CharLiteral
    | LParen => true
    _ => false
  }
}

///|
fn is_operator_token(kind : @syntax.SyntaxKind) -> Bool {
  match kind {
    Plus
    | Minus
    | Star
    | Slash
    | Percent
    | Less
    | Greater
    | LessEqual
    | GreaterEqual
    | EqualEqual
    | NotEqual
    | AndAnd
    | OrOr
    | Dollar
    | Bang
    | Dot => true
    _ => false
  }
}

///|
fn parse_int_literal(text : String) -> Int64 {
  if text.length() >= 2 &&
    text.unsafe_charcode_at(0) == 0x30 &&
    (text.unsafe_charcode_at(1) == 0x78 || text.unsafe_charcode_at(1) == 0x58) {
    parse_int_in_radix(text.substring(start=2, end=text.length()), 16)
  } else if text.length() >= 2 &&
    text.unsafe_charcode_at(0) == 0x30 &&
    (text.unsafe_charcode_at(1) == 0x6F || text.unsafe_charcode_at(1) == 0x4F) {
    parse_int_in_radix(text.substring(start=2, end=text.length()), 8)
  } else if text.length() >= 2 &&
    text.unsafe_charcode_at(0) == 0x30 &&
    (text.unsafe_charcode_at(1) == 0x62 || text.unsafe_charcode_at(1) == 0x42) {
    parse_int_in_radix(text.substring(start=2, end=text.length()), 2)
  } else {
    parse_int_in_radix(text, 10)
  }
}

///|
fn parse_int_in_radix(text : String, radix : Int64) -> Int64 {
  let mut acc : Int64 = 0L
  let mut i = 0
  let len = text.length()
  let r = radix
  while i < len {
    let c = text.unsafe_charcode_at(i)
    let digit = if c >= 0x30 && c <= 0x39 {
      c - 0x30
    } else if c >= 0x61 && c <= 0x66 {
      c - 0x61 + 10
    } else if c >= 0x41 && c <= 0x46 {
      c - 0x41 + 10
    } else {
      -1
    }
    if digit < 0 || digit.to_int64() >= r {
      return acc
    }
    acc = acc * r + digit.to_int64()
    i = i + 1
  }
  acc
}

///|
/// Parse a decimal float literal. Hand-rolled to avoid pinning a particular
/// `@strconv` API surface across MoonBit toolchain versions.
fn parse_float_literal(text : String) -> Double {
  let len = text.length()
  let mut i = 0
  let mut sign : Double = 1.0
  if i < len && text.unsafe_charcode_at(i) == 0x2D {
    sign = -1.0
    i = i + 1
  } else if i < len && text.unsafe_charcode_at(i) == 0x2B {
    i = i + 1
  }
  let mut int_part : Double = 0.0
  while i < len {
    let c = text.unsafe_charcode_at(i)
    if c < 0x30 || c > 0x39 {
      break
    }
    int_part = int_part * 10.0 + (c - 0x30).to_double()
    i = i + 1
  }
  let mut frac : Double = 0.0
  let mut scale : Double = 1.0
  if i < len && text.unsafe_charcode_at(i) == 0x2E {
    i = i + 1
    while i < len {
      let c = text.unsafe_charcode_at(i)
      if c < 0x30 || c > 0x39 {
        break
      }
      frac = frac * 10.0 + (c - 0x30).to_double()
      scale = scale * 10.0
      i = i + 1
    }
  }
  let mut mantissa : Double = int_part + frac / scale
  let mut exp_sign : Int = 1
  let mut exp_val : Int = 0
  if i < len && (text.unsafe_charcode_at(i) == 0x65 || text.unsafe_charcode_at(i) == 0x45) {
    i = i + 1
    if i < len && text.unsafe_charcode_at(i) == 0x2D {
      exp_sign = -1
      i = i + 1
    } else if i < len && text.unsafe_charcode_at(i) == 0x2B {
      i = i + 1
    }
    while i < len {
      let c = text.unsafe_charcode_at(i)
      if c < 0x30 || c > 0x39 {
        break
      }
      exp_val = exp_val * 10 + (c - 0x30)
      i = i + 1
    }
  }
  let mut e = exp_val * exp_sign
  while e > 0 {
    mantissa = mantissa * 10.0
    e = e - 1
  }
  while e < 0 {
    mantissa = mantissa / 10.0
    e = e + 1
  }
  sign * mantissa
}


///|
/// Process backslash escapes inside a string literal body (already stripped
/// of the surrounding quotes).
fn unescape_string(s : String) -> String {
  let len = s.length()
  let mut out = ""
  let mut i = 0
  while i < len {
    let c = s.unsafe_charcode_at(i)
    if c == 0x5C && i + 1 < len {
      let nxt = s.unsafe_charcode_at(i + 1)
      match nxt {
        0x6E => out = out + "\n"
        0x74 => out = out + "\t"
        0x72 => out = out + "\r"
        0x5C => out = out + "\\"
        0x22 => out = out + "\""
        0x30 => out = out + "\u{0000}"
        _ => out = out + "\\" + s.substring(start=i + 1, end=i + 2)
      }
      i = i + 2
    } else {
      out = out + s.substring(start=i, end=i + 1)
      i = i + 1
    }
  }
  out
}

///|
/// Decode a single-character `'x'` literal body (already stripped of quotes).
fn unescape_char(s : String) -> Char {
  if s.length() == 0 {
    return ' '
  }
  let first = s.unsafe_charcode_at(0)
  if first == 0x5C && s.length() >= 2 {
    let nxt = s.unsafe_charcode_at(1)
    match nxt {
      0x6E => '\n'
      0x74 => '\t'
      0x72 => '\r'
      0x5C => '\\'
      0x27 => '\''
      0x30 => '\u{0000}'
      _ => single_char_of(s, 1)
    }
  } else {
    single_char_of(s, 0)
  }
}

///|
/// Best-effort "first Char" of a substring. Used to materialise the body of a
/// char literal without depending on a particular `Int -> Char` API.
fn single_char_of(s : String, offset : Int) -> Char {
  let one = s.substring(start=offset, end=offset + 1)
  let mut out : Char = ' '
  for c in one {
    out = c
    break
  }
  out
}
