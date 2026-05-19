///|
test "clean source produces no lint warnings" {
  let source = "main : I32 -> I32\nmain x = x + 1"
  let diags = @linter.lint(source)
  assert_eq(diags.length(), 0)
}

///|
test "unused let binding is flagged" {
  let source = "f x = let y = x + 1 in x"
  let diags = @linter.lint(source)
  assert_eq(diags.length(), 1)
  let d = diags[0]
  assert_eq(d.severity, @diagnostics.Severity::Warning)
  assert_true(d.message.contains("Unused let binding"))
  assert_true(d.message.contains("`y`"))
  assert_eq(d.code, Some("unused-let"))
  match d.help {
    Some(h) => assert_true(h.contains("rename") || h.contains("remove"))
    None => fail("expected help text on unused-let")
  }
}

///|
test "leading-underscore let names are not flagged" {
  // `_unused` is the conventional "intentionally unused" suffix.
  let source = "f x = let _unused = x + 1 in x"
  let diags = @linter.lint(source)
  assert_eq(diags.length(), 0)
}

///|
test "let binding referenced in its own RHS (recursive shape) is not flagged" {
  // Even though `body` doesn't reference `y`, another binding does — so
  // the `unused-let` rule shouldn't fire (the user clearly intended to
  // use it for the sibling's RHS).
  let source = "f x = let y = x in let z = y + 1 in z"
  let diags = @linter.lint(source)
  // Outer `y` is used by inner let's `y + 1`, so only `z` is referenced
  // and `y` is too — no warnings.
  assert_eq(diags.length(), 0)
}

///|
test "unused where binding is flagged" {
  let source = "f x = x + 1 where\n  unused = x * 2"
  let diags = @linter.lint(source)
  assert_eq(diags.length(), 1)
  let d = diags[0]
  assert_true(d.message.contains("Unused where binding"))
  assert_eq(d.code, Some("unused-where"))
}

///|
test "used where binding is not flagged" {
  let source = "f x = y * 2 where\n  y = x + 1"
  let diags = @linter.lint(source)
  assert_eq(diags.length(), 0)
}

///|
test "redundant arm after a wildcard is flagged" {
  let source = #|data Color = Red | Green | Blue
                #|
                #|toI32 : Color -> I32
                #|toI32 color = match color
                #|  | Red -> 0
                #|  | _ -> 1
                #|  | Green -> 2
  let diags = @linter.lint(source)
  let mut found = false
  for d in diags {
    if d.message.contains("Unreachable match arm") {
      assert_eq(d.code, Some("redundant-wildcard"))
      found = true
    }
  }
  assert_true(found)
}

///|
test "wildcard at the end is not flagged" {
  let source = #|data Color = Red | Green | Blue
                #|
                #|toI32 : Color -> I32
                #|toI32 color = match color
                #|  | Red -> 0
                #|  | Green -> 1
                #|  | _ -> 99
  let diags = @linter.lint(source)
  for d in diags {
    assert_false(d.message.contains("Unreachable"))
  }
}

///|
test "diagnostics come back sorted by source position" {
  let source = "f x = let y = 1 in let z = 2 in x"
  let diags = @linter.lint(source)
  let mut prev_offset : UInt = 0U
  for d in diags {
    if d.labels.length() > 0 {
      let off = d.labels[0].span.start
      assert_true(off >= prev_offset)
      prev_offset = off
    }
  }
}

///|
test "LintRuleId::code returns kebab-case names" {
  assert_eq(@linter.LintRuleId::UnusedLet.code(), "unused-let")
  assert_eq(@linter.LintRuleId::UnusedWhere.code(), "unused-where")
  assert_eq(
    @linter.LintRuleId::ShadowedBinding.code(),
    "shadowed-binding",
  )
  assert_eq(
    @linter.LintRuleId::RedundantWildcard.code(),
    "redundant-wildcard",
  )
}

///|
test "lint_program: works on a parsed AST without re-parsing" {
  let parser = @parser.Parser::new("f x = let y = x in x")
  let program = parser.parse_program()
  let diags = @linter.lint_program(program)
  assert_eq(diags.length(), 1)
  assert_true(diags[0].message.contains("Unused let binding"))
}

///|
test "shadowed-binding: let shadows a function parameter" {
  let source = "f x = let x = 1 in x"
  let diags = @linter.lint(source)
  let mut found = false
  for d in diags {
    match d.code {
      Some(c) =>
        if c == "shadowed-binding" {
          assert_true(d.message.contains("Shadowed binding"))
          assert_true(d.message.contains("`x`"))
          found = true
        }
      None => ()
    }
  }
  assert_true(found)
}

///|
test "shadowed-binding: inner let shadows outer let" {
  let source = "f a = let x = a in let x = x + 1 in x"
  let diags = @linter.lint(source)
  let mut count = 0
  for d in diags {
    match d.code {
      Some(c) =>
        if c == "shadowed-binding" {
          count = count + 1
        }
      None => ()
    }
  }
  assert_eq(count, 1)
}

///|
test "shadowed-binding: distinct names produce no warning" {
  let source = "f x = let y = x in let z = y in z"
  let diags = @linter.lint(source)
  for d in diags {
    match d.code {
      Some(c) => assert_false(c == "shadowed-binding")
      None => ()
    }
  }
}

///|
test "shadowed-binding: leading-underscore names opt out" {
  let source = "f _x = let _x = 1 in _x"
  let diags = @linter.lint(source)
  for d in diags {
    match d.code {
      Some(c) => assert_false(c == "shadowed-binding")
      None => ()
    }
  }
}
