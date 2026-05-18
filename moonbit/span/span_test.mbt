///|
test "span new" {
  let s = @span.Span::new(10, 20)
  assert_eq(s.start, 10)
  assert_eq(s.end, 20)
}

///|
test "span merge" {
  let a = @span.Span::new(5, 15)
  let b = @span.Span::new(10, 25)
  let merged = a.merge(b)
  assert_eq(merged.start, 5)
  assert_eq(merged.end, 25)
}

///|
test "span contains" {
  let s = @span.Span::new(10, 20)
  assert_true(s.contains(10))
  assert_true(s.contains(15))
  assert_true(s.contains(19))
  assert_false(s.contains(20))
  assert_false(s.contains(9))
}

///|
test "span source_text" {
  let source = "hello world"
  let s = @span.Span::new(6, 11)
  assert_eq(s.source_text(source), "world")
}

///|
test "span zero" {
  assert_eq(@span.span_zero, @span.Span::new(0, 0))
}

///|
test "atom from str" {
  let atom = @span.Atom::new("hello")
  assert_eq(atom.as_str(), "hello")
}

///|
test "atom order" {
  let a = @span.Atom::new("apple")
  let b = @span.Atom::new("banana")
  assert_true(a < b)
}

///|
test "spanned" {
  let s = @span.Spanned::new(42, @span.Span::new(0, 5))
  assert_eq(s.node, 42)
  assert_eq(s.span, @span.Span::new(0, 5))
}
