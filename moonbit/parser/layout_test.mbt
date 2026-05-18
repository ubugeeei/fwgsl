///|
fn layout_kinds(source : String) -> Array[@syntax.SyntaxKind] {
  let tokens = @parser.lex(source)
  let resolved = @parser.resolve_layout(tokens, source)
  resolved.map(fn(t) { t.kind })
}

///|
test "layout where" {
  let source = "f x = x where\n  g = 1\n  h = 2\n"
  let kinds = layout_kinds(source)
  // After `where`, we expect LayoutBraceOpen
  assert_true(kinds.contains(@syntax.SyntaxKind::LayoutBraceOpen))
  // Between the bindings at the same indent, we expect LayoutSemicolon
  assert_true(kinds.contains(@syntax.SyntaxKind::LayoutSemicolon))
}

///|
test "layout let in" {
  let source = "let\n  x = 1\n  y = 2\nin x"
  let kinds = layout_kinds(source)
  assert_true(kinds.contains(@syntax.SyntaxKind::LayoutBraceOpen))
  assert_true(kinds.contains(@syntax.SyntaxKind::LayoutSemicolon))
  assert_true(kinds.contains(@syntax.SyntaxKind::LayoutBraceClose))
}

///|
test "layout closes at eof" {
  let source = "do\n  x <- action"
  let kinds = layout_kinds(source)
  let mut open_count = 0
  let mut close_count = 0
  for k in kinds {
    if k == @syntax.SyntaxKind::LayoutBraceOpen {
      open_count = open_count + 1
    } else if k == @syntax.SyntaxKind::LayoutBraceClose {
      close_count = close_count + 1
    }
  }
  assert_eq(open_count, close_count)
}

///|
test "explicit braces not overridden" {
  let source = "let { x = 1; y = 2 } in x"
  let kinds = layout_kinds(source)
  // The top-level layout context always inserts one LayoutBraceOpen.
  // The `let` keyword should NOT insert an additional one because an
  // explicit `{` follows it.
  let mut open_count = 0
  for k in kinds {
    if k == @syntax.SyntaxKind::LayoutBraceOpen {
      open_count = open_count + 1
    }
  }
  assert_eq(open_count, 1)
}
