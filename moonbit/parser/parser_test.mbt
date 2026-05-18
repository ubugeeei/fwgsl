///|
fn parse(source : String) -> @ast.Program {
  let p = @parser.Parser::new(source)
  p.parse_program()
}

///|
test "parse type sig" {
  let prog = parse("add : I32 -> I32 -> I32")
  assert_eq(prog.decls.length(), 1)
  match prog.decls[0] {
    TypeSig(name, _, _) => assert_eq(name, "add")
    _ => fail("expected TypeSig")
  }
}

///|
test "parse dependent array type sig" {
  let prog = parse("grid : Array 2 (Array 4 F32)")
  assert_eq(prog.decls.length(), 1)
  match prog.decls[0] {
    TypeSig(_, ty, _) =>
      match ty {
        App(outer, inner, _) => {
          let outer_is_app = match outer {
            App(_, _, _) => true
            _ => false
          }
          let inner_is_paren = match inner {
            Paren(_, _) => true
            _ => false
          }
          assert_true(outer_is_app)
          assert_true(inner_is_paren)
        }
        _ => fail("expected applied array type")
      }
    _ => fail("expected TypeSig")
  }
}

///|
test "parse fun decl" {
  let prog = parse("add x y = x + y")
  assert_eq(prog.decls.length(), 1)
  match prog.decls[0] {
    FunDecl(name, params, _, _, _) => {
      assert_eq(name, "add")
      assert_eq(params.length(), 2)
    }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse data decl" {
  let prog = parse("data Color = Red | Green | Blue")
  assert_eq(prog.decls.length(), 1)
  match prog.decls[0] {
    DataDecl(name, _, constructors, _) => {
      assert_eq(name, "Color")
      assert_eq(constructors.length(), 3)
      assert_eq(constructors[0].name, "Red")
      assert_eq(constructors[1].name, "Green")
      assert_eq(constructors[2].name, "Blue")
    }
    _ => fail("expected DataDecl")
  }
}

///|
test "parse case expr" {
  let source = "f c = match c\n  | Red -> 0\n  | Green -> 1\n  | Blue -> 2"
  let prog = parse(source)
  assert_eq(prog.decls.length(), 1)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) =>
      match body {
        Case(_, arms, _) => assert_eq(arms.length(), 3)
        _ => fail("expected Case")
      }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse let expr" {
  let source = "f x = let y = x + 1 in y"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) =>
      match body {
        Let(binds, _, _) => {
          assert_eq(binds.length(), 1)
          assert_eq(binds[0].0, "y")
        }
        _ => fail("expected Let")
      }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse where clause" {
  let source = "f x = y + 1 where y = x"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, where_binds, _) => {
      let is_infix = match body {
        Infix(_, _, _, _) => true
        _ => false
      }
      assert_true(is_infix)
      assert_eq(where_binds.length(), 1)
      assert_eq(where_binds[0].0, "y")
    }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse lambda" {
  let source = "f = \\x y -> x + y"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) =>
      match body {
        Lambda(params, _, _) => assert_eq(params.length(), 2)
        _ => fail("expected Lambda")
      }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse if expr" {
  let source = "f x = if x == 0 then 1 else 2"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) => {
      let is_if = match body {
        If(_, _, _, _) => true
        _ => false
      }
      assert_true(is_if)
    }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse full program" {
  let source = #|data Color = Red | Green | Blue
                #|
                #|show : Color -> I32
                #|show c = match c
                #|  | Red   -> 0
                #|  | Green -> 1
                #|  | Blue  -> 2
                #|
                #|add : I32 -> I32 -> I32
                #|add x y = x + y
                #|
                #|main : I32 -> I32
                #|main x =
                #|  let y = add x 1
                #|  in show Red
  let prog = parse(source)
  assert_true(prog.decls.length() >= 4)
}

///|
test "parse entry point" {
  let source = "@vertex\nmain x = x + 1"
  let prog = parse(source)
  assert_eq(prog.decls.length(), 1)
  match prog.decls[0] {
    EntryPoint(attributes, name, _, _, _) => {
      assert_eq(attributes.length(), 1)
      assert_eq(attributes[0].name, "vertex")
      assert_eq(name, "main")
    }
    _ => fail("expected EntryPoint")
  }
}

///|
test "parse operator section" {
  let source = "f = (+)"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) => {
      let is_section = match body {
        OpSection(_, _) => true
        _ => false
      }
      assert_true(is_section)
    }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse tuple" {
  let source = "f = (1, 2, 3)"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) =>
      match body {
        Tuple(elems, _) => assert_eq(elems.length(), 3)
        _ => fail("expected Tuple")
      }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse unit" {
  let source = "f = ()"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) => {
      let is_unit_tuple = match body {
        Tuple(elems, _) => elems.length() == 0
        _ => false
      }
      assert_true(is_unit_tuple)
    }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse precedence" {
  let source = "f = 1 + 2 * 3"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) =>
      match body {
        Infix(_, op, rhs, _) => {
          assert_eq(op, "+")
          let rhs_is_mul = match rhs {
            Infix(_, op2, _, _) => op2 == "*"
            _ => false
          }
          assert_true(rhs_is_mul)
        }
        _ => fail("expected Infix(+)")
      }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse dollar right assoc" {
  let source = "r = f $ g $ x"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) =>
      match body {
        Infix(_, op, rhs, _) => {
          assert_eq(op, "$")
          let rhs_is_dollar = match rhs {
            Infix(_, op2, _, _) => op2 == "$"
            _ => false
          }
          assert_true(rhs_is_dollar)
        }
        _ => fail("expected Infix($)")
      }
    _ => fail("expected FunDecl")
  }
}

///|
test "parse function application" {
  let source = "f = add x y"
  let prog = parse(source)
  match prog.decls[0] {
    FunDecl(_, _, body, _, _) =>
      match body {
        App(lhs, rhs, _) => {
          let rhs_is_y = match rhs {
            Var(name, _) => name == "y"
            _ => false
          }
          let lhs_is_app = match lhs {
            App(_, _, _) => true
            _ => false
          }
          assert_true(rhs_is_y)
          assert_true(lhs_is_app)
        }
        _ => fail("expected App")
      }
    _ => fail("expected FunDecl")
  }
}
