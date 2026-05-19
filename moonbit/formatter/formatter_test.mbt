///|
test "format collapses multi-space whitespace" {
  let src = "f    x   =   x   +    1"
  let out = @formatter.format(src)
  assert_eq(out, "f x = x + 1\n")
}

///|
test "format preserves comments" {
  let src = "-- hello\nf x = x"
  let out = @formatter.format(src)
  assert_true(out.contains("-- hello"))
  assert_true(out.contains("f x = x"))
}

///|
test "format collapses runs of blank lines" {
  let src = "a = 1\n\n\n\n\nb = 2"
  let out = @formatter.format(src)
  assert_eq(out, "a = 1\n\nb = 2\n")
}

///|
test "format trims trailing whitespace per line" {
  let src = "f = 1   \n"
  let out = @formatter.format(src)
  assert_eq(out, "f = 1\n")
}

///|
test "format ends with exactly one newline" {
  let src = "f = 1"
  let out = @formatter.format(src)
  assert_eq(out, "f = 1\n")
}

///|
test "format leaves no-whitespace shape alone" {
  // `vec3(1, 2, 3)` and similar tight shapes are preserved when the
  // input had no whitespace between tokens.
  let src = "f = vec3(1,2,3)"
  let out = @formatter.format(src)
  assert_eq(out, "f = vec3(1,2,3)\n")
}
