///|
fn s() -> @span.Span {
  @span.Span::new(0, 0)
}

///|
test "ty display con" {
  assert_eq(@typechecker.Ty::i32().display(), "I32")
  assert_eq(@typechecker.Ty::f32().display(), "F32")
  assert_eq(@typechecker.Ty::bool().display(), "Bool")
}

///|
test "ty display arrow" {
  let ty = @typechecker.Ty::arrow(@typechecker.Ty::i32(), @typechecker.Ty::bool())
  assert_eq(ty.display(), "(I32 -> Bool)")
}

///|
test "ty display app" {
  let ty = @typechecker.Ty::app(
    @typechecker.Ty::Con("Maybe"),
    @typechecker.Ty::i32(),
  )
  assert_eq(ty.display(), "(Maybe I32)")
}

///|
test "ty display forall" {
  let ty = @typechecker.Ty::Forall(["a"], @typechecker.Ty::Var(0))
  assert_eq(ty.display(), "forall a. t0")
}

///|
test "ty display nat" {
  assert_eq(@typechecker.Ty::Nat(3UL).display(), "3")
}

///|
test "ty display var" {
  assert_eq(@typechecker.Ty::Var(42).display(), "t42")
}

///|
test "ty display error" {
  assert_eq(@typechecker.Ty::Error.display(), "<error>")
}

///|
test "contains var" {
  let ty = @typechecker.Ty::arrow(@typechecker.Ty::Var(0), @typechecker.Ty::i32())
  assert_true(ty.contains_var(0))
  assert_false(ty.contains_var(1))
}

///|
test "free vars" {
  let ty = @typechecker.Ty::arrow(
    @typechecker.Ty::Var(0),
    @typechecker.Ty::arrow(@typechecker.Ty::Var(1), @typechecker.Ty::Var(0)),
  )
  let fv = ty.free_vars()
  assert_eq(fv.length(), 2)
  assert_eq(fv[0], 0)
  assert_eq(fv[1], 1)
}

///|
test "substitution basic" {
  let subst = @typechecker.Substitution::new()
  subst.insert(0, @typechecker.Ty::i32())
  let ty = @typechecker.Ty::arrow(@typechecker.Ty::Var(0), @typechecker.Ty::Var(1))
  let result = ty.apply_subst(subst)
  let expected = @typechecker.Ty::arrow(@typechecker.Ty::i32(), @typechecker.Ty::Var(1))
  assert_eq(result, expected)
}

///|
test "substitution compose" {
  let s1 = @typechecker.Substitution::new()
  s1.insert(0, @typechecker.Ty::Var(1))
  let s2 = @typechecker.Substitution::new()
  s2.insert(1, @typechecker.Ty::i32())
  let composed = s1.compose(s2)
  let result = @typechecker.Ty::Var(0).apply_subst(composed)
  assert_eq(result, @typechecker.Ty::i32())
}

///|
test "unify same con" {
  let engine = @typechecker.InferEngine::new()
  engine.unify(@typechecker.Ty::i32(), @typechecker.Ty::i32(), s())
  assert_false(engine.diagnostics().has_errors())
}

///|
test "unify var with con" {
  let engine = @typechecker.InferEngine::new()
  let var = engine.fresh_var()
  engine.unify(var, @typechecker.Ty::i32(), s())
  assert_false(engine.diagnostics().has_errors())
  assert_eq(engine.finalize(var), @typechecker.Ty::i32())
}

///|
test "unify arrows" {
  let engine = @typechecker.InferEngine::new()
  let a = engine.fresh_var()
  let b = engine.fresh_var()
  let arrow1 = @typechecker.Ty::arrow(a, b)
  let arrow2 = @typechecker.Ty::arrow(@typechecker.Ty::i32(), @typechecker.Ty::bool())
  engine.unify(arrow1, arrow2, s())
  assert_false(engine.diagnostics().has_errors())
  assert_eq(engine.finalize(a), @typechecker.Ty::i32())
  assert_eq(engine.finalize(b), @typechecker.Ty::bool())
}

///|
test "unify mismatch" {
  let engine = @typechecker.InferEngine::new()
  engine.unify(@typechecker.Ty::i32(), @typechecker.Ty::bool(), s())
  assert_true(engine.diagnostics().has_errors())
}

///|
test "unify occurs check" {
  let engine = @typechecker.InferEngine::new()
  let a = engine.fresh_var()
  let circular = @typechecker.Ty::arrow(a, @typechecker.Ty::i32())
  engine.unify(a, circular, s())
  assert_true(engine.diagnostics().has_errors())
}

///|
test "generalize" {
  let engine = @typechecker.InferEngine::new()
  let env = @typechecker.TypeEnv::new()
  let ty = @typechecker.Ty::arrow(@typechecker.Ty::Var(0), @typechecker.Ty::Var(0))
  let scheme = engine.generalize(env, ty)
  assert_eq(scheme.vars.length(), 1)
  assert_eq(scheme.vars[0], 0)
}

///|
test "generalize with env constraint" {
  let engine = @typechecker.InferEngine::new()
  let env = @typechecker.TypeEnv::new()
  env.insert("x", @typechecker.Scheme::mono(@typechecker.Ty::Var(0)))
  let ty = @typechecker.Ty::arrow(@typechecker.Ty::Var(0), @typechecker.Ty::Var(1))
  let scheme = engine.generalize(env, ty)
  assert_eq(scheme.vars.length(), 1)
  assert_eq(scheme.vars[0], 1)
}

///|
test "ty to wgsl scalars" {
  assert_eq(@typechecker.ty_to_wgsl(@typechecker.Ty::i32()), Ok(@typechecker.WgslType::I32))
  assert_eq(@typechecker.ty_to_wgsl(@typechecker.Ty::f32()), Ok(@typechecker.WgslType::F32))
  assert_eq(@typechecker.ty_to_wgsl(@typechecker.Ty::u32()), Ok(@typechecker.WgslType::U32))
  assert_eq(@typechecker.ty_to_wgsl(@typechecker.Ty::bool()), Ok(@typechecker.WgslType::Bool))
  assert_eq(@typechecker.ty_to_wgsl(@typechecker.Ty::unit()), Ok(@typechecker.WgslType::Unit))
}

///|
test "ty to wgsl vec" {
  let ty = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Nat(3UL)),
    @typechecker.Ty::f32(),
  )
  assert_eq(
    @typechecker.ty_to_wgsl(ty),
    Ok(@typechecker.WgslType::Vec(3, @typechecker.WgslType::F32)),
  )
}

///|
test "ty to wgsl mat" {
  let ty = @typechecker.matrix_ty(4UL, 4UL, @typechecker.Ty::f32())
  assert_eq(
    @typechecker.ty_to_wgsl(ty),
    Ok(@typechecker.WgslType::Mat(4, 4, @typechecker.WgslType::F32)),
  )
}

///|
test "ty to wgsl tensor" {
  let ty = @typechecker.tensor_ty(@typechecker.Ty::Nat(4UL), @typechecker.Ty::f32())
  assert_eq(
    @typechecker.ty_to_wgsl(ty),
    Ok(@typechecker.WgslType::WgslArray(@typechecker.WgslType::F32, Some(4))),
  )
}

///|
test "ty aliases normalize before wgsl lowering" {
  assert_eq(
    @typechecker.ty_to_wgsl(
      @typechecker.Ty::app(@typechecker.Ty::Con("Scalar"), @typechecker.Ty::f32()),
    ),
    Ok(@typechecker.WgslType::F32),
  )
  let vec3 = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Vector"), @typechecker.Ty::Nat(3UL)),
    @typechecker.Ty::f32(),
  )
  assert_eq(
    @typechecker.ty_to_wgsl(vec3),
    Ok(@typechecker.WgslType::Vec(3, @typechecker.WgslType::F32)),
  )
}

///|
test "ty to wgsl function error" {
  let ty = @typechecker.Ty::arrow(@typechecker.Ty::i32(), @typechecker.Ty::i32())
  match @typechecker.ty_to_wgsl(ty) {
    Err(_) => ()
    Ok(_) => fail("expected error for function type")
  }
}

///|
test "wgsl type display" {
  assert_eq(@typechecker.WgslType::I32.display(), "i32")
  assert_eq(@typechecker.WgslType::F32.display(), "f32")
  assert_eq(
    @typechecker.WgslType::Vec(3, @typechecker.WgslType::F32).display(),
    "vec3<f32>",
  )
  assert_eq(
    @typechecker.WgslType::Mat(4, 4, @typechecker.WgslType::F32).display(),
    "mat4x4<f32>",
  )
  assert_eq(
    @typechecker.WgslType::WgslArray(@typechecker.WgslType::F32, Some(16)).display(),
    "array<f32, 16>",
  )
  assert_eq(@typechecker.WgslType::Unit.display(), "void")
  assert_eq(@typechecker.WgslType::Struct("MyStruct").display(), "MyStruct")
}

///|
test "scheme mono" {
  let scheme = @typechecker.Scheme::mono(@typechecker.Ty::i32())
  assert_eq(scheme.vars.length(), 0)
  assert_eq(scheme.ty, @typechecker.Ty::i32())
}

///|
test "type env basic" {
  let env = @typechecker.TypeEnv::new()
  env.insert("x", @typechecker.Scheme::mono(@typechecker.Ty::i32()))
  match env.lookup("x") {
    Some(_) => ()
    None => fail("expected x to be in env")
  }
  match env.lookup("y") {
    None => ()
    Some(_) => fail("expected y to be absent")
  }
}
