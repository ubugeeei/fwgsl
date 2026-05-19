///|
test "diagnostic error builder" {
  let diag = @diagnostics.Diagnostic::error("type mismatch")
    .with_code("E001")
    .with_label(@diagnostics.Label::primary(@span.Span::new(0, 5), "expected Int"))
    .with_help("try adding a type annotation")
  assert_eq(diag.severity, @diagnostics.Severity::Error)
  assert_eq(diag.message, "type mismatch")
  assert_eq(diag.code, Some("E001"))
  assert_eq(diag.labels.length(), 1)
  assert_eq(diag.help, Some("try adding a type annotation"))
}

///|
test "diagnostic warning" {
  let diag = @diagnostics.Diagnostic::warning("unused variable")
  assert_eq(diag.severity, @diagnostics.Severity::Warning)
  assert_eq(diag.message, "unused variable")
}

///|
test "diagnostic sink" {
  let sink = @diagnostics.DiagnosticSink::new()
  assert_false(sink.has_errors())
  sink.push(@diagnostics.Diagnostic::warning("unused import"))
  assert_false(sink.has_errors())
  sink.push(@diagnostics.Diagnostic::error("syntax error"))
  assert_true(sink.has_errors())
  assert_eq(sink.len(), 2)
}

///|
test "render includes prefix and code" {
  let diag = @diagnostics.Diagnostic::error("unexpected token")
    .with_code("E100")
    .with_label(@diagnostics.Label::primary(@span.Span::new(0, 3), "here"))
    .with_help("did you mean `let`?")
  let text = @diagnostics.render(diag, "test.fwgsl", "lat x = 42")
  assert_true(text.contains("error: "))
  assert_true(text.contains("[E100]"))
  assert_true(text.contains("unexpected token"))
  assert_true(text.contains("test.fwgsl:1:1"))
  assert_true(text.contains("here"))
  assert_true(text.contains("help: did you mean `let`?"))
}

///|
test "label constructors equal" {
  let span = @span.Span::new(5, 10)
  let l1 = @diagnostics.Label::new(span, "message")
  let l2 = @diagnostics.Label::primary(span, "message")
  assert_eq(l1.span, l2.span)
  assert_eq(l1.message, l2.message)
}

///|
test "offset to line col across newlines" {
  let diag = @diagnostics.Diagnostic::error("oops")
    .with_label(@diagnostics.Label::primary(@span.Span::new(8, 10), "here"))
  let text = @diagnostics.render(diag, "x.fwgsl", "line1\nline2\nline3")
  assert_true(text.contains("x.fwgsl:2:3"))
}

///|
test "render embeds the source line excerpt" {
  let diag = @diagnostics.Diagnostic::error("Unbound variable: y")
    .with_label(@diagnostics.Label::primary(@span.Span::new(6, 7), "not in scope"))
  let text = @diagnostics.render(diag, "x.fwgsl", "f x = y")
  // The line that contains `y` is included in the rendered output.
  assert_true(text.contains("f x = y"))
  // A caret points at the offending token.
  assert_true(text.contains("^"))
  // The label message follows the caret.
  assert_true(text.contains("not in scope"))
}

///|
test "render widens the caret to cover multi-char spans" {
  let diag = @diagnostics.Diagnostic::error("type mismatch")
    .with_label(@diagnostics.Label::primary(@span.Span::new(0, 5), "expected Int"))
  let text = @diagnostics.render(diag, "x.fwgsl", "false")
  // The caret should span 5 chars, not just one.
  assert_true(text.contains("^^^^^"))
}

///|
test "render works on the very last line of the source" {
  // No trailing newline; the snippet extractor still grabs the full line.
  let diag = @diagnostics.Diagnostic::error("oops")
    .with_label(@diagnostics.Label::primary(@span.Span::new(12, 13), "x"))
  let text = @diagnostics.render(diag, "x.fwgsl", "line1\nline2\nfoo")
  assert_true(text.contains("foo"))
}
