///|
test "type span" {
  let s = @span.Span::new(0, 5)
  assert_eq(@ast.Type::Con("Int", s).span(), s)
  let inner = @ast.Type::Var("a", s)
  let outer = @ast.Type::Paren(inner, @span.Span::new(0, 10))
  assert_eq(outer.span(), @span.Span::new(0, 10))
}

///|
test "expr span" {
  let s = @span.Span::new(3, 7)
  assert_eq(@ast.Expr::Var("x", s).span(), s)
  let lam = @ast.Expr::Lambda(
    Array::new(),
    @ast.Expr::Var("x", s),
    @span.Span::new(0, 10),
  )
  assert_eq(lam.span(), @span.Span::new(0, 10))
}

///|
test "pat span" {
  let s = @span.Span::new(2, 4)
  assert_eq(@ast.Pat::Wild(s).span(), s)
  let p = @ast.Pat::Var("x", @span.Span::new(2, 3))
  let outer = @ast.Pat::As("y", p, @span.Span::new(0, 10))
  assert_eq(outer.span(), @span.Span::new(0, 10))
}

///|
test "decl span" {
  let s = @span.Span::new(0, 20)
  let d = @ast.Decl::TypeSig("foo", @ast.Type::Unit(s), s)
  assert_eq(d.span(), s)
}

///|
test "program new" {
  let p = @ast.Program::empty()
  assert_eq(p.decls.length(), 0)
}

///|
test "constructor decl" {
  let s = @span.Span::new(0, 5)
  let c = @ast.ConDecl::new("Red", @ast.ConFields::Empty, s)
  assert_eq(c.name, "Red")
  assert_eq(c.span, s)
}

///|
test "attribute" {
  let s = @span.Span::new(0, 8)
  let attr = @ast.Attribute::new("compute", Array::new(), s)
  assert_eq(attr.name, "compute")
}
