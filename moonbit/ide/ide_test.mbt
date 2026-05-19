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
  // Hovering past the end of the document is None.
  match @ide.hover_at("main = 1", 999) {
    None => ()
    Some(_) => fail("expected None hover past EOF")
  }
}

///|
test "hover falls back to kind label for unbound identifiers" {
  // `xyz` is referenced inside the body of `f` but never bound, so the
  // semantic env can't type it — we fall through to the kind-aware
  // label. (`xyz` alone as a top-level token would end up bound by the
  // parser's error-recovery FunDecl, which would defeat the test.)
  let source = "f x = xyz"
  // Offset 6 lands on the `x` of `xyz`.
  match @ide.hover_at(source, 6) {
    Some(h) => assert_true(h.contents.contains("identifier `xyz`"))
    None => fail("expected hover on `xyz`")
  }
}

///|
test "hover renders the inferred type for a top-level binding" {
  let source = "add : I32 -> I32 -> I32\nadd x y = x + y"
  // Hover on the `add` reference inside the definition (right side).
  match @ide.hover_at(source, 24) {
    Some(h) => {
      assert_true(h.contents.contains("add : "))
      assert_true(h.contents.contains("I32"))
      // The pretty-printed arrow type should be right-associative, with
      // no fully-parenthesised inner arrow.
      assert_true(h.contents.contains("->"))
      // Sanity: the falling-back label is not used here.
      assert_false(h.contents.contains("identifier `add`"))
    }
    None => fail("expected hover on `add`")
  }
}

///|
test "hover renders polymorphic builtins with letter type variables" {
  // `id` is a builtin with scheme `a -> a`. Hovering on the bare token
  // should render that type, not `t0 -> t0`.
  match @ide.hover_at("id", 0) {
    Some(h) => {
      assert_true(h.contents.contains("id : "))
      assert_true(h.contents.contains("a -> a"))
    }
    None => fail("expected hover on `id`")
  }
}

///|
test "hover renders a constructor type" {
  // `Some` is a builtin constructor with scheme `a -> Option a`.
  match @ide.hover_at("Some", 0) {
    Some(h) => {
      assert_true(h.contents.contains("Some : "))
      assert_true(h.contents.contains("Option"))
    }
    None => fail("expected hover on `Some`")
  }
}

///|
test "pretty_type right-associates arrows" {
  let t = @typechecker.Ty::arrow(
    @typechecker.Ty::i32(),
    @typechecker.Ty::arrow(@typechecker.Ty::i32(), @typechecker.Ty::i32()),
  )
  assert_eq(@ide.pretty_type(t), "I32 -> I32 -> I32")
}

///|
test "pretty_type parenthesises arrow on the left of an arrow" {
  let inner = @typechecker.Ty::arrow(
    @typechecker.Ty::i32(),
    @typechecker.Ty::i32(),
  )
  let t = @typechecker.Ty::arrow(inner, @typechecker.Ty::i32())
  assert_eq(@ide.pretty_type(t), "(I32 -> I32) -> I32")
}

///|
test "pretty_type parenthesises type applications inside an application" {
  let opt_a = @typechecker.Ty::app(
    @typechecker.Ty::Con("Option"),
    @typechecker.Ty::Var(0),
  )
  let opt_opt = @typechecker.Ty::app(@typechecker.Ty::Con("Option"), opt_a)
  assert_eq(@ide.pretty_type(opt_opt), "Option (Option a)")
}

///|
test "pretty_scheme drops the forall prefix" {
  let scheme = @typechecker.Scheme::poly(
    [0U, 1U],
    @typechecker.Ty::arrow(
      @typechecker.Ty::Var(0),
      @typechecker.Ty::arrow(@typechecker.Ty::Var(1), @typechecker.Ty::Var(0)),
    ),
  )
  assert_eq(@ide.pretty_scheme(scheme), "a -> b -> a")
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
