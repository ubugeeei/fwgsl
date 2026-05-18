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
test "hover and goto are None until name resolution lands" {
  match @ide.hover_at("main = 1", 0) {
    None => ()
    Some(_) => fail("expected None hover")
  }
  match @ide.goto_definition_at("main = 1", 0) {
    None => ()
    Some(_) => fail("expected None goto target")
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
