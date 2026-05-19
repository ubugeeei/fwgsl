///|
test "completions include every reserved keyword" {
  let items = @ide.completions_at("", 0)
  for kw in @ide.keyword_completions() {
    let mut found = false
    for it in items {
      if it.label == kw && it.kind == @ide.CompletionKind::Keyword {
        found = true
        break
      }
    }
    if !found {
      fail("missing keyword `" + kw + "`")
    }
  }
}

///|
test "completions include in-scope bindings with their inferred type" {
  let source = "add : I32 -> I32 -> I32\nadd x y = x + y"
  let items = @ide.completions_at(source, 0)
  let mut found_add = false
  for it in items {
    if it.label == "add" {
      assert_eq(it.kind, @ide.CompletionKind::Function)
      match it.detail {
        Some(d) => {
          assert_true(d.contains("I32"))
          assert_true(d.contains("->"))
        }
        None => fail("expected detail on `add`")
      }
      found_add = true
    }
  }
  assert_true(found_add)
}

///|
test "completions include polymorphic builtins" {
  let items = @ide.completions_at("", 0)
  let mut found_id = false
  let mut found_fmap = false
  for it in items {
    if it.label == "id" {
      found_id = true
      assert_eq(it.kind, @ide.CompletionKind::Function)
      match it.detail {
        Some(d) => assert_true(d.contains("a -> a"))
        None => fail("expected detail on `id`")
      }
    }
    if it.label == "fmap" {
      found_fmap = true
    }
  }
  assert_true(found_id)
  assert_true(found_fmap)
}

///|
test "completions include builtin constructors and data types" {
  let items = @ide.completions_at("", 0)
  let mut found_some = false
  let mut found_option = false
  let mut found_pair = false
  for it in items {
    if it.label == "Some" {
      found_some = true
      assert_eq(it.kind, @ide.CompletionKind::Constructor)
      match it.detail {
        Some(d) => {
          assert_true(d.contains("Option"))
          assert_true(d.contains("constructor of Option"))
        }
        None => fail("expected detail on `Some`")
      }
    }
    if it.label == "Option" {
      found_option = true
      assert_eq(it.kind, @ide.CompletionKind::Type)
    }
    if it.label == "Pair" {
      found_pair = true
    }
  }
  assert_true(found_some)
  assert_true(found_option)
  assert_true(found_pair)
}

///|
test "completions exclude binary operators from the identifier list" {
  // `+`, `-`, `==`, `&&` etc. live in the env but are not appropriate as
  // identifier completions — editors filter them by prefix anyway.
  let items = @ide.completions_at("", 0)
  for it in items {
    let lbl = it.label
    assert_false(lbl == "+")
    assert_false(lbl == "==")
    assert_false(lbl == "&&")
    assert_false(lbl == "/=")
  }
}

///|
test "completions surface user-declared constructors" {
  let source = "data Shape = Circle I32 | Square I32"
  let items = @ide.completions_at(source, 0)
  let mut found_shape = false
  let mut found_circle = false
  let mut found_square = false
  for it in items {
    if it.label == "Shape" {
      found_shape = true
      assert_eq(it.kind, @ide.CompletionKind::Type)
    }
    if it.label == "Circle" {
      found_circle = true
      assert_eq(it.kind, @ide.CompletionKind::Constructor)
      match it.detail {
        Some(d) => assert_true(d.contains("Shape"))
        None => fail("expected detail on `Circle`")
      }
    }
    if it.label == "Square" {
      found_square = true
    }
  }
  assert_true(found_shape)
  assert_true(found_circle)
  assert_true(found_square)
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

test "semantic_tokens classifies number literals" {
  let source = "f x = 42 + 3.14"
  let tokens = @ide.semantic_tokens(source)
  let mut nums = 0
  for t in tokens {
    if t.kind == @ide.SemanticTokenKind::TokNumber {
      nums = nums + 1
    }
  }
  assert_eq(nums, 2) // `42` and `3.14`
}

///|
test "semantic_tokens classifies string literals" {
  let source = "f x = \"hello\""
  let tokens = @ide.semantic_tokens(source)
  let mut strs = 0
  for t in tokens {
    if t.kind == @ide.SemanticTokenKind::TokString {
      strs = strs + 1
    }
  }
  assert_eq(strs, 1)
}

///|
test "semantic_tokens classifies UpperIdent as TokType" {
  let source = "data Color = Red"
  let tokens = @ide.semantic_tokens(source)
  let mut types = 0
  for t in tokens {
    if t.kind == @ide.SemanticTokenKind::TokType {
      types = types + 1
    }
  }
  // `Color` and `Red` are both UpperIdent.
  assert_true(types >= 2)
}

///|
test "semantic_tokens classifies arrow-typed bindings as TokFunction" {
  // `add` is bound to `I32 -> I32 -> I32` -> Function. `x` and `y` are
  // body-local Idents not in the env -> Variable.
  let source = "add : I32 -> I32 -> I32\nadd x y = x + y"
  let tokens = @ide.semantic_tokens(source)
  let mut found_fn_add = false
  for t in tokens {
    if t.kind == @ide.SemanticTokenKind::TokFunction &&
      t.span.source_text(source) == "add" {
      found_fn_add = true
    }
  }
  assert_true(found_fn_add)
}

///|
test "semantic_tokens classifies decorator attributes" {
  let source = "@compute\nmain x = x"
  let tokens = @ide.semantic_tokens(source)
  let mut decos = 0
  for t in tokens {
    if t.kind == @ide.SemanticTokenKind::TokDecorator {
      decos = decos + 1
    }
  }
  // `@` and `compute` should both be Decorator-classified.
  assert_true(decos >= 2)
}

///|
test "semantic_tokens skips raw whitespace and newlines" {
  let source = "let\nx = 1"
  let tokens = @ide.semantic_tokens(source)
  for t in tokens {
    // The classifier never emits a token for trivia.
    let text = t.span.source_text(source)
    assert_false(text == "\n")
    assert_false(text == " ")
  }
}

///|
test "code_actions_at offers `Add type signature` for an unannotated decl" {
  let source = "add x y = x + y"
  // Offset 0 lands on the `add` token.
  let actions = @ide.code_actions_at(source, 0)
  assert_eq(actions.length(), 1)
  let a = actions[0]
  assert_eq(a.kind, @ide.CodeActionKind::QuickFix)
  assert_true(a.title.contains("Add type signature"))
  assert_true(a.title.contains("`add`"))
  assert_eq(a.edits.length(), 1)
  let e = a.edits[0]
  assert_eq(e.start, 0U)
  assert_eq(e.end, 0U)
  // The new text is `add : <type>\n`.
  assert_true(e.new_text.contains("add : "))
  assert_true(e.new_text.contains("->"))
  // Ends with a newline so the original line stays intact.
  let last = e.new_text.unsafe_charcode_at(e.new_text.length() - 1)
  assert_eq(last, 0x0A)
}

///|
test "code_actions_at skips decls that already have a signature" {
  let source = "add : I32 -> I32 -> I32\nadd x y = x + y"
  let actions = @ide.code_actions_at(source, 24)
  assert_eq(actions.length(), 0)
}

///|
test "code_actions_at returns nothing when no decl covers the offset" {
  let source = "add x y = x + y"
  // Offset past the end of the source.
  let actions = @ide.code_actions_at(source, 999U)
  assert_eq(actions.length(), 0)
}

///|
test "code_actions_at preserves the decl's leading indentation" {
  // The decl sits at column 2 (two spaces of indent). The inserted
  // signature should match that indent so the edited output stays
  // aligned.
  let source = "  foo x = x"
  let actions = @ide.code_actions_at(source, 2)
  assert_eq(actions.length(), 1)
  let e = actions[0].edits[0]
  // Insertion happens at the line start, before the indent.
  assert_eq(e.start, 0U)
  assert_eq(e.end, 0U)
  let nl = e.new_text.unsafe_charcode_at(0)
  let nl2 = e.new_text.unsafe_charcode_at(1)
  // First two chars are the preserved indent.
  assert_eq(nl, 0x20)
  assert_eq(nl2, 0x20)
}

///|
test "CodeActionKind::lsp_str returns the LSP-protocol strings" {
  assert_eq(@ide.CodeActionKind::QuickFix.lsp_str(), "quickfix")
  assert_eq(@ide.CodeActionKind::Refactor.lsp_str(), "refactor")
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
