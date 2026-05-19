///|
test "completions returns keyword list" {
  let items = @ide.completions_at("", 0)
  assert_true(items.length() >= 10)
  let mut saw_let = false
  let mut saw_match = false
  for it in items {
    if it.label == "let" {
      saw_let = true
    }
    if it.label == "match" {
      saw_match = true
    }
  }
  assert_true(saw_let)
  assert_true(saw_match)
  // Every item is of kind Keyword for the placeholder.
  for it in items {
    assert_eq(it.kind, @ide.CompletionKind::Keyword)
  }
}

///|
test "hover identifies kinds" {
  // Hovering on `let` -> keyword.
  match @ide.hover_at("let x = 1 in x", 1) {
    Some(h) => assert_true(h.contents.contains("keyword `let`"))
    None => fail("expected hover on `let`")
  }
  // Hovering on identifier.
  match @ide.hover_at("main = 1", 0) {
    Some(h) => assert_true(h.contents.contains("identifier `main`"))
    None => fail("expected hover on identifier")
  }
  // Hovering past the end of the document is None.
  match @ide.hover_at("main = 1", 999) {
    None => ()
    Some(_) => fail("expected None hover past EOF")
  }
}

///|
test "goto definition jumps to the function name" {
  let source = "scale x = x + 1\nuse x = scale x"
  // Hover on the second `scale` (after `=`).
  let offset_of_use : UInt = 24
  match @ide.goto_definition_at(source, offset_of_use) {
    Some(target) => assert_eq(target.span.start, 0U)
    None => fail("expected goto target")
  }
}

///|
test "goto definition is None for keyword tokens" {
  match @ide.goto_definition_at("let x = 1 in x", 0) {
    None => ()
    Some(_) => fail("expected None for keyword")
  }
}

///|
test "token_at returns None for trivia" {
  // Offset 1 falls on whitespace.
  match @ide.token_at("a b", 1) {
    None => ()
    Some(_) => fail("expected None for whitespace")
  }
}

///|
test "semantic_tokens classifies keyword + operator + comment" {
  let source = "-- a comment\nlet x = 1 in x + 1"
  let tokens = @ide.semantic_tokens(source)
  let mut kw = 0
  let mut op = 0
  let mut com = 0
  for t in tokens {
    match t.kind {
      TokKeyword => kw = kw + 1
      TokOperator => op = op + 1
      TokComment => com = com + 1
      _ => ()
    }
  }
  assert_true(kw >= 2) // `let` and `in`
  assert_true(op >= 1) // `+`
  assert_true(com >= 1) // line comment
}

///|
test "CompletionItem::new wires the label and kind" {
  let it = @ide.CompletionItem::new("foo", @ide.CompletionKind::Function)
  assert_eq(it.label, "foo")
  assert_eq(it.kind, @ide.CompletionKind::Function)
  match it.detail {
    None => ()
    Some(_) => fail("expected default None detail")
  }
}
