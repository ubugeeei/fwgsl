///|
test "check_source returns parser diagnostics" {
  let sink = check_source("data Color = Red | Green | Blue")
  // No syntax errors in well-formed source.
  assert_false(sink.has_errors())
}

///|
test "compile_source returns program and diagnostics" {
  let (prog, sink) = compile_source("data Color = Red | Green | Blue")
  assert_eq(prog.decls.length(), 1)
  assert_false(sink.has_errors())
}

///|
test "format_diagnostic includes location" {
  let diag = @diagnostics.Diagnostic::error("oops").with_label(
    @diagnostics.Label::primary(@span.Span::new(0, 3), "here"),
  )
  let s = format_diagnostic(diag, "x.fwgsl", "abc")
  assert_true(s.contains("x.fwgsl:1:1"))
  assert_true(s.contains("error: oops"))
}
