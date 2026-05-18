///|
fn kinds(source : String) -> Array[@syntax.SyntaxKind] {
  @parser.lex(source).map(fn(t) { t.kind })
}

///|
test "lex empty" {
  let tokens = @parser.lex("")
  assert_eq(tokens.length(), 1)
  assert_eq(tokens[0].kind, @syntax.SyntaxKind::Eof)
}

///|
test "lex whitespace and newline" {
  assert_eq(
    kinds("  \n\t"),
    [
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Newline,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Eof,
    ],
  )
}

///|
test "lex line comment" {
  let tokens = @parser.lex("-- hello\nx")
  assert_eq(tokens[0].kind, @syntax.SyntaxKind::LineComment)
  assert_eq(tokens[1].kind, @syntax.SyntaxKind::Newline)
  assert_eq(tokens[2].kind, @syntax.SyntaxKind::Ident)
}

///|
test "lex block comment nested" {
  let tokens = @parser.lex("{- outer {- inner -} still outer -}")
  assert_eq(tokens[0].kind, @syntax.SyntaxKind::BlockComment)
  assert_eq(tokens[1].kind, @syntax.SyntaxKind::Eof)
}

///|
test "lex integers" {
  assert_eq(kinds("42"), [@syntax.SyntaxKind::IntLiteral, @syntax.SyntaxKind::Eof])
  assert_eq(kinds("0xFF"), [@syntax.SyntaxKind::IntLiteral, @syntax.SyntaxKind::Eof])
  assert_eq(kinds("0o77"), [@syntax.SyntaxKind::IntLiteral, @syntax.SyntaxKind::Eof])
  assert_eq(kinds("0b101"), [@syntax.SyntaxKind::IntLiteral, @syntax.SyntaxKind::Eof])
}

///|
test "lex float" {
  assert_eq(kinds("3.14"), [@syntax.SyntaxKind::FloatLiteral, @syntax.SyntaxKind::Eof])
  assert_eq(kinds("1.0e10"), [@syntax.SyntaxKind::FloatLiteral, @syntax.SyntaxKind::Eof])
  assert_eq(kinds("2.5e-3"), [@syntax.SyntaxKind::FloatLiteral, @syntax.SyntaxKind::Eof])
}

///|
test "lex string" {
  let tokens = @parser.lex("\"hello world\"")
  assert_eq(tokens[0].kind, @syntax.SyntaxKind::StringLiteral)
}

///|
test "lex char" {
  let tokens = @parser.lex("'a'")
  assert_eq(tokens[0].kind, @syntax.SyntaxKind::CharLiteral)
}

///|
test "lex keywords" {
  assert_eq(
    kinds("let in case of match where data"),
    [
      @syntax.SyntaxKind::KwLet,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::KwIn,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::KwCase,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::KwOf,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::KwMatch,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::KwWhere,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::KwData,
      @syntax.SyntaxKind::Eof,
    ],
  )
}

///|
test "lex identifiers" {
  assert_eq(
    kinds("foo Bar _x"),
    [
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::UpperIdent,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Eof,
    ],
  )
}

///|
test "lex underscore alone" {
  assert_eq(
    kinds("_ x"),
    [
      @syntax.SyntaxKind::Underscore,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Eof,
    ],
  )
}

///|
test "lex multi-char operators" {
  assert_eq(
    kinds("-> => :: .. <= >= == /= && || <-"),
    [
      @syntax.SyntaxKind::Arrow,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::FatArrow,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::ColonColon,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::DotDot,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::LessEqual,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::GreaterEqual,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::EqualEqual,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::NotEqual,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::AndAnd,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::OrOr,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::LeftArrow,
      @syntax.SyntaxKind::Eof,
    ],
  )
}

///|
test "lex primed identifier" {
  let tokens = @parser.lex("x'")
  assert_eq(tokens[0].kind, @syntax.SyntaxKind::Ident)
  assert_eq(tokens[0].span.source_text("x'"), "x'")
}

///|
test "lex small program" {
  let source = "add x y = x + y"
  let tokens = @parser.lex(source)
  let kinds_arr : Array[@syntax.SyntaxKind] = tokens.map(fn(t) { t.kind })
  assert_eq(
    kinds_arr,
    [
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Equals,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Plus,
      @syntax.SyntaxKind::Whitespace,
      @syntax.SyntaxKind::Ident,
      @syntax.SyntaxKind::Eof,
    ],
  )
}
