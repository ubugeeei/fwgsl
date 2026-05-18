///|
test "green token round trip" {
  let s = @span.Span::new(0, 5)
  let t = @cst.GreenToken::new(@syntax.SyntaxKind::Ident, s)
  assert_eq(t.kind, @syntax.SyntaxKind::Ident)
  assert_eq(t.span, s)
}

///|
test "green node holds children" {
  let span1 = @span.Span::new(0, 3)
  let span2 = @span.Span::new(4, 7)
  let outer = @span.Span::new(0, 7)
  let t1 = @cst.GreenToken::new(@syntax.SyntaxKind::Ident, span1)
  let t2 = @cst.GreenToken::new(@syntax.SyntaxKind::Plus, span2)
  let node = @cst.GreenNode::new(
    @syntax.SyntaxKind::ExprInfix,
    outer,
    [@cst.GreenChild::Token(t1), @cst.GreenChild::Token(t2)],
  )
  assert_eq(node.children.length(), 2)
  assert_eq(node.children[0].span(), span1)
  assert_eq(node.children[1].span(), span2)
}
