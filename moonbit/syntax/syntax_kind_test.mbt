///|
test "trivia classification" {
  assert_true(@syntax.SyntaxKind::Whitespace.is_trivia())
  assert_true(@syntax.SyntaxKind::Newline.is_trivia())
  assert_true(@syntax.SyntaxKind::LineComment.is_trivia())
  assert_true(@syntax.SyntaxKind::BlockComment.is_trivia())
  assert_false(@syntax.SyntaxKind::Ident.is_trivia())
}

///|
test "keyword classification" {
  assert_true(@syntax.SyntaxKind::KwModule.is_keyword())
  assert_true(@syntax.SyntaxKind::KwDeriving.is_keyword())
  assert_true(@syntax.SyntaxKind::KwLet.is_keyword())
  assert_false(@syntax.SyntaxKind::Ident.is_keyword())
  assert_false(@syntax.SyntaxKind::Plus.is_keyword())
}

///|
test "literal classification" {
  assert_true(@syntax.SyntaxKind::IntLiteral.is_literal())
  assert_true(@syntax.SyntaxKind::FloatLiteral.is_literal())
  assert_true(@syntax.SyntaxKind::StringLiteral.is_literal())
  assert_true(@syntax.SyntaxKind::CharLiteral.is_literal())
  assert_false(@syntax.SyntaxKind::Ident.is_literal())
}

///|
test "operator classification" {
  assert_true(@syntax.SyntaxKind::Plus.is_operator())
  assert_true(@syntax.SyntaxKind::Minus.is_operator())
  assert_true(@syntax.SyntaxKind::Bang.is_operator())
  assert_true(@syntax.SyntaxKind::Dollar.is_operator())
  assert_true(@syntax.SyntaxKind::Backtick.is_operator())
  assert_false(@syntax.SyntaxKind::Ident.is_operator())
  // Compose was deliberately placed past LAST_OPERATOR in the Rust crate.
  assert_false(@syntax.SyntaxKind::Compose.is_operator())
}

///|
test "keyword from str" {
  assert_eq(@syntax.keyword_from_str("module"), Some(@syntax.SyntaxKind::KwModule))
  assert_eq(@syntax.keyword_from_str("where"), Some(@syntax.SyntaxKind::KwWhere))
  assert_eq(@syntax.keyword_from_str("let"), Some(@syntax.SyntaxKind::KwLet))
  assert_eq(@syntax.keyword_from_str("forall"), Some(@syntax.SyntaxKind::KwForall))
  assert_eq(@syntax.keyword_from_str("deriving"), Some(@syntax.SyntaxKind::KwDeriving))
  assert_eq(@syntax.keyword_from_str("notakeyword"), None)
  assert_eq(@syntax.keyword_from_str(""), None)
}

///|
test "display formatting" {
  assert_eq(@syntax.SyntaxKind::KwLet.display(), "'let'")
  assert_eq(@syntax.SyntaxKind::Ident.display(), "identifier")
  assert_eq(@syntax.SyntaxKind::Arrow.display(), "'->'")
  assert_eq(@syntax.SyntaxKind::SourceFile.display(), "source file")
}
