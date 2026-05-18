///|
fn s() -> @span.Span {
  @span.Span::new(0, 0)
}

///|
test "builtin data types registered" {
  let sa = @semantic.SemanticAnalyzer::new()
  let dts = sa.data_types()
  assert_true(dts.get("Option") is Some(_))
  assert_true(dts.get("Result") is Some(_))
  assert_true(dts.get("Pair") is Some(_))
}

///|
test "builtin constructors registered" {
  let sa = @semantic.SemanticAnalyzer::new()
  let ctors = sa.constructors()
  assert_true(ctors.get("Some") is Some(_))
  assert_true(ctors.get("None") is Some(_))
  assert_true(ctors.get("Ok") is Some(_))
  assert_true(ctors.get("Err") is Some(_))
  assert_true(ctors.get("Pair") is Some(_))
}

///|
test "builtin operators registered" {
  let sa = @semantic.SemanticAnalyzer::new()
  let env = sa.env()
  for op in ["+", "-", "*", "/", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||"] {
    match env.lookup(op) {
      Some(_) => ()
      None => fail("operator " + op + " not in env")
    }
  }
}

///|
test "builtin value prelude registered" {
  let sa = @semantic.SemanticAnalyzer::new()
  let env = sa.env()
  for name in [
      "id",
      "const",
      "flip",
      "compose",
      "pure",
      "bind",
      "fmap",
      "map",
      "$map",
      "filter",
      "$filter",
      "foldl",
      "fold",
      "$fold",
      "foldr",
      "$foldr",
      "flat",
      "$flat",
      "flatMap",
      "$flatMap",
      "zip",
      "$zip",
      "all",
      "$all",
      "any",
      "$any",
      "fst",
      "$fst",
      "snd",
      "$snd",
      "swap",
      "$swap",
      "unwrapOr",
      "$unwrapOr",
      "$sin",
      "$cos",
      "$abs",
      "$fract",
      "$floor",
      "$sign",
      "$sqrt",
      "negate",
      "$normalize",
      "$max",
      "$min",
      "$step",
      "$mod",
      "$pow",
      "$reflect",
      "$atan",
      "$clamp",
      "$mix",
      "$smoothstep",
      "$length",
      "$dot",
      "$vec2",
      "$vec3",
      "$vec4",
      "$splat2",
      "$splat3",
      "$splat4",
      "$vecX",
      "$vecY",
      "$vecZ",
      "$vecW",
    ] {
    match env.lookup(name) {
      Some(_) => ()
      None => fail("prelude entry " + name + " not in env")
    }
  }
}

///|
test "register data decl from ast" {
  let sa = @semantic.SemanticAnalyzer::new()
  let prog = @ast.Program::new(
    [
      @ast.Decl::DataDecl(
        "Color",
        Array::new(),
        [
          @ast.ConDecl::new("Red", @ast.ConFields::Empty, s()),
          @ast.ConDecl::new("Green", @ast.ConFields::Empty, s()),
          @ast.ConDecl::new("Blue", @ast.ConFields::Empty, s()),
        ],
        s(),
      ),
    ],
  )
  sa.analyze(prog)
  match sa.data_types().get("Color") {
    Some(dt) => assert_eq(dt.constructors.length(), 3)
    None => fail("Color not registered")
  }
  assert_true(sa.constructors().get("Red") is Some(_))
}

///|
test "simple function inference" {
  let sa = @semantic.SemanticAnalyzer::new()
  // f x = x + 1
  let prog = @ast.Program::new(
    [
      @ast.Decl::FunDecl(
        "f",
        [@ast.Pat::Var("x", s())],
        @ast.Expr::Infix(
          @ast.Expr::Var("x", s()),
          "+",
          @ast.Expr::Lit(@ast.Lit::Int(1L), s()),
          s(),
        ),
        Array::new(),
        s(),
      ),
    ],
  )
  sa.analyze(prog)
  assert_false(sa.has_errors())
  match sa.env().lookup("f") {
    Some(scheme) => {
      let ty = sa.engine().finalize(scheme.ty)
      assert_eq(ty.display(), "(I32 -> I32)")
    }
    None => fail("f should be in env")
  }
}

///|
test "unbound variable reports error" {
  let sa = @semantic.SemanticAnalyzer::new()
  let prog = @ast.Program::new(
    [
      @ast.Decl::FunDecl(
        "f",
        [@ast.Pat::Var("x", s())],
        @ast.Expr::Var("y", s()),
        Array::new(),
        s(),
      ),
    ],
  )
  sa.analyze(prog)
  assert_true(sa.has_errors())
}

///|
test "type signature unified with body" {
  let sa = @semantic.SemanticAnalyzer::new()
  // add : I32 -> I32 -> I32
  // add x y = x + y
  let prog = @ast.Program::new(
    [
      @ast.Decl::TypeSig(
        "add",
        @ast.Type::Arrow(
          @ast.Type::Con("I32", s()),
          @ast.Type::Arrow(
            @ast.Type::Con("I32", s()),
            @ast.Type::Con("I32", s()),
            s(),
          ),
          s(),
        ),
        s(),
      ),
      @ast.Decl::FunDecl(
        "add",
        [@ast.Pat::Var("x", s()), @ast.Pat::Var("y", s())],
        @ast.Expr::Infix(
          @ast.Expr::Var("x", s()),
          "+",
          @ast.Expr::Var("y", s()),
          s(),
        ),
        Array::new(),
        s(),
      ),
    ],
  )
  sa.analyze(prog)
  assert_false(sa.has_errors())
}

///|
test "is_swizzle classification" {
  assert_true(@semantic.is_swizzle("x"))
  assert_true(@semantic.is_swizzle("xyz"))
  assert_true(@semantic.is_swizzle("rgba"))
  assert_false(@semantic.is_swizzle(""))
  assert_false(@semantic.is_swizzle("xyzwa"))
  assert_false(@semantic.is_swizzle("xq"))
}

///|
test "validate_swizzle bounds" {
  assert_true(@semantic.validate_swizzle("xy", 3))
  assert_true(@semantic.validate_swizzle("w", 4))
  assert_false(@semantic.validate_swizzle("z", 2))
  assert_false(@semantic.validate_swizzle("w", 3))
}

///|
test "extract_vec_type" {
  let ty = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Nat(3UL)),
    @typechecker.Ty::f32(),
  )
  match @semantic.extract_vec_type(ty) {
    Some(triple) => {
      let (n, scalar) = triple
      assert_eq(n, 3)
      assert_eq(scalar, @typechecker.Ty::f32())
    }
    None => fail("expected Some")
  }
  match @semantic.extract_vec_type(@typechecker.Ty::i32()) {
    None => ()
    Some(_) => fail("expected None for non-vec")
  }
}
