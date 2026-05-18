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
test "register data decl from ast" {
  let sa = @semantic.SemanticAnalyzer::new()
  let s = @span.Span::new(0, 10)
  let prog = @ast.Program::new(
    [
      @ast.Decl::DataDecl(
        "Color",
        Array::new(),
        [
          @ast.ConDecl::new("Red", @ast.ConFields::Empty, s),
          @ast.ConDecl::new("Green", @ast.ConFields::Empty, s),
          @ast.ConDecl::new("Blue", @ast.ConFields::Empty, s),
        ],
        s,
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
