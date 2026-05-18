///|
test "hir expr ty" {
  let expr = @hir.HirExpr::Lit(
    @hir.HirLit::Int(42L),
    @typechecker.Ty::i32(),
    @span.Span::new(0, 2),
  )
  assert_eq(expr.ty(), @typechecker.Ty::i32())
}

///|
test "binop from str" {
  assert_eq(@hir.BinOp::parse("+"), Some(@hir.BinOp::Add))
  assert_eq(@hir.BinOp::parse("-"), Some(@hir.BinOp::Sub))
  assert_eq(@hir.BinOp::parse("*"), Some(@hir.BinOp::Mul))
  assert_eq(@hir.BinOp::parse("/"), Some(@hir.BinOp::Div))
  assert_eq(@hir.BinOp::parse("%"), Some(@hir.BinOp::Mod))
  assert_eq(@hir.BinOp::parse("=="), Some(@hir.BinOp::Eq))
  assert_eq(@hir.BinOp::parse("/="), Some(@hir.BinOp::Ne))
  assert_eq(@hir.BinOp::parse("<"), Some(@hir.BinOp::Lt))
  assert_eq(@hir.BinOp::parse(">"), Some(@hir.BinOp::Gt))
  assert_eq(@hir.BinOp::parse("<="), Some(@hir.BinOp::Le))
  assert_eq(@hir.BinOp::parse(">="), Some(@hir.BinOp::Ge))
  assert_eq(@hir.BinOp::parse("&&"), Some(@hir.BinOp::And))
  assert_eq(@hir.BinOp::parse("||"), Some(@hir.BinOp::Or))
  assert_eq(@hir.BinOp::parse("???"), None)
}

///|
test "binop to wgsl str" {
  assert_eq(@hir.BinOp::Add.to_wgsl_str(), "+")
  assert_eq(@hir.BinOp::Ne.to_wgsl_str(), "!=")
}

///|
test "hir if expr" {
  let expr = @hir.HirExpr::If(
    @hir.HirExpr::Lit(
      @hir.HirLit::Bool(true),
      @typechecker.Ty::bool(),
      @span.Span::new(0, 4),
    ),
    @hir.HirExpr::Lit(@hir.HirLit::Int(1L), @typechecker.Ty::i32(), @span.Span::new(10, 11)),
    @hir.HirExpr::Lit(@hir.HirLit::Int(0L), @typechecker.Ty::i32(), @span.Span::new(17, 18)),
    @typechecker.Ty::i32(),
    @span.Span::new(0, 18),
  )
  assert_eq(expr.ty(), @typechecker.Ty::i32())
}
