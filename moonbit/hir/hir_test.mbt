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

///|
test "all HirExpr variants expose ty()" {
  let s = @span.Span::new(0, 0)
  let lit = @hir.HirExpr::Lit(@hir.HirLit::Int(1L), @typechecker.Ty::i32(), s)
  assert_eq(lit.ty(), @typechecker.Ty::i32())
  let var = @hir.HirExpr::Var("x", @typechecker.Ty::f32(), s)
  assert_eq(var.ty(), @typechecker.Ty::f32())
  let app = @hir.HirExpr::App(lit, var, @typechecker.Ty::bool(), s)
  assert_eq(app.ty(), @typechecker.Ty::bool())
  let let_e = @hir.HirExpr::Let(
    [("y", lit)],
    var,
    @typechecker.Ty::u32(),
    s,
  )
  assert_eq(let_e.ty(), @typechecker.Ty::u32())
  let case_e = @hir.HirExpr::Case(
    lit,
    Array::new(),
    @typechecker.Ty::unit(),
    s,
  )
  assert_eq(case_e.ty(), @typechecker.Ty::unit())
  let binop = @hir.HirExpr::BinOpE(
    @hir.BinOp::Add,
    lit,
    var,
    @typechecker.Ty::i32(),
    s,
  )
  assert_eq(binop.ty(), @typechecker.Ty::i32())
  let cc = @hir.HirExpr::ConstructorCall(
    "Some",
    0U,
    [lit],
    @typechecker.Ty::Con("Maybe"),
    s,
  )
  assert_eq(cc.ty(), @typechecker.Ty::Con("Maybe"))
  let fa = @hir.HirExpr::FieldAccess(var, "x", @typechecker.Ty::f32(), s)
  assert_eq(fa.ty(), @typechecker.Ty::f32())
}

///|
test "BinOp::parse round-trips through to_wgsl_str" {
  for s in ["+", "-", "*", "/", "%", "==", "<", ">", "<=", ">="] {
    match @hir.BinOp::parse(s) {
      Some(op) => {
        let wgsl = op.to_wgsl_str()
        // WGSL keeps the same spelling for these.
        assert_eq(wgsl, s)
      }
      None => fail("expected to parse: " + s)
    }
  }
  // /= maps to WGSL !=
  match @hir.BinOp::parse("/=") {
    Some(op) => assert_eq(op.to_wgsl_str(), "!=")
    None => fail("expected to parse /=")
  }
}

///|
test "HirPattern variants" {
  let s = @hir.HirPattern::Wild
  match s {
    Wild => ()
    _ => fail("expected Wild")
  }
  let v = @hir.HirPattern::Var("x", @typechecker.Ty::i32())
  match v {
    Var(name, _) => assert_eq(name, "x")
    _ => fail("expected Var")
  }
  let con = @hir.HirPattern::Constructor("Red", 0U, Array::new())
  match con {
    Constructor(name, tag, subs) => {
      assert_eq(name, "Red")
      assert_eq(tag, 0U)
      assert_eq(subs.length(), 0)
    }
    _ => fail("expected Constructor")
  }
}
