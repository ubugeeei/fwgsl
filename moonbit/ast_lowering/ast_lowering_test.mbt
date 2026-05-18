///|
fn s() -> @span.Span {
  @span.Span::new(0, 0)
}

///|
test "lower empty program" {
  let sa = @semantic.SemanticAnalyzer::new()
  let prog = @ast.Program::empty()
  sa.analyze(prog)
  let lowering = @ast_lowering.AstLowering::new(sa)
  let hir = lowering.lower_program(prog)
  assert_eq(hir.functions.length(), 0)
  assert_eq(hir.data_types.length(), 0)
  assert_eq(hir.entry_points.length(), 0)
}

///|
test "lower add function via type sig" {
  let sa = @semantic.SemanticAnalyzer::new()
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
  let lowering = @ast_lowering.AstLowering::new(sa)
  let hir = lowering.lower_program(prog)
  assert_eq(hir.functions.length(), 1)
  let f = hir.functions[0]
  assert_eq(f.name, "add")
  assert_eq(f.params.length(), 2)
  let body_is_binop = match f.body {
    BinOpE(Add, _, _, _, _) => true
    _ => false
  }
  assert_true(body_is_binop)
}

///|
test "lower data declaration" {
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
  let lowering = @ast_lowering.AstLowering::new(sa)
  let hir = lowering.lower_program(prog)
  assert_eq(hir.data_types.length(), 1)
  let dt = hir.data_types[0]
  assert_eq(dt.name, "Color")
  assert_eq(dt.constructors.length(), 3)
  assert_eq(dt.constructors[0].name, "Red")
  assert_eq(dt.constructors[0].tag, 0)
  assert_eq(dt.constructors[2].tag, 2)
}

///|
test "lower where clause becomes let" {
  let sa = @semantic.SemanticAnalyzer::new()
  let prog = @ast.Program::new(
    [
      @ast.Decl::FunDecl(
        "f",
        [@ast.Pat::Var("x", s())],
        @ast.Expr::Infix(
          @ast.Expr::Var("y", s()),
          "+",
          @ast.Expr::Lit(@ast.Lit::Int(1L), s()),
          s(),
        ),
        [("y", @ast.Expr::Var("x", s()))],
        s(),
      ),
    ],
  )
  sa.analyze(prog)
  let lowering = @ast_lowering.AstLowering::new(sa)
  let hir = lowering.lower_program(prog)
  assert_eq(hir.functions.length(), 1)
  let f = hir.functions[0]
  let body_is_let = match f.body {
    Let(binds, _, _, _) => binds.length() == 1 && binds[0].0 == "y"
    _ => false
  }
  assert_true(body_is_let)
}

///|
test "constructor used as value" {
  let sa = @semantic.SemanticAnalyzer::new()
  let prog = @ast.Program::new(
    [
      @ast.Decl::DataDecl(
        "Color",
        Array::new(),
        [@ast.ConDecl::new("Red", @ast.ConFields::Empty, s())],
        s(),
      ),
      @ast.Decl::FunDecl(
        "red",
        Array::new(),
        @ast.Expr::Con("Red", s()),
        Array::new(),
        s(),
      ),
    ],
  )
  sa.analyze(prog)
  let lowering = @ast_lowering.AstLowering::new(sa)
  let hir = lowering.lower_program(prog)
  assert_eq(hir.functions.length(), 1)
  let body_is_ctor_call = match hir.functions[0].body {
    ConstructorCall("Red", 0, _, _, _) => true
    _ => false
  }
  assert_true(body_is_ctor_call)
}
