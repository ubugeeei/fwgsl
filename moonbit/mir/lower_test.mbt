///|
fn s() -> @span.Span {
  @span.Span::new(0, 0)
}

///|
test "ty to mir type scalars" {
  assert_eq(@mir.ty_to_mir_type(@typechecker.Ty::i32()), Ok(@mir.MirType::I32))
  assert_eq(@mir.ty_to_mir_type(@typechecker.Ty::f32()), Ok(@mir.MirType::F32))
  assert_eq(@mir.ty_to_mir_type(@typechecker.Ty::u32()), Ok(@mir.MirType::U32))
  assert_eq(@mir.ty_to_mir_type(@typechecker.Ty::bool()), Ok(@mir.MirType::Bool))
  assert_eq(@mir.ty_to_mir_type(@typechecker.Ty::unit()), Ok(@mir.MirType::Unit))
}

///|
test "ty to mir type tensor" {
  let ty = @typechecker.tensor_ty(@typechecker.Ty::Nat(4UL), @typechecker.Ty::f32())
  assert_eq(@mir.ty_to_mir_type(ty), Ok(@mir.MirType::MirArray(@mir.MirType::F32, 4)))
}

///|
test "ty to mir type nested tensor" {
  let inner = @typechecker.tensor_ty(@typechecker.Ty::Nat(4UL), @typechecker.Ty::f32())
  let ty = @typechecker.tensor_ty(@typechecker.Ty::Nat(2UL), inner)
  assert_eq(
    @mir.ty_to_mir_type(ty),
    Ok(@mir.MirType::MirArray(@mir.MirType::MirArray(@mir.MirType::F32, 4), 2)),
  )
}

///|
test "ty aliases normalize before mir lowering" {
  let ty = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Array"), @typechecker.Ty::Nat(8UL)),
    @typechecker.Ty::f32(),
  )
  assert_eq(@mir.ty_to_mir_type(ty), Ok(@mir.MirType::MirArray(@mir.MirType::F32, 8)))
}

///|
test "lower simple function" {
  let hir = @hir.HirProgram::{
    functions: [
      @hir.HirFunction::{
        name: "add",
        params: [("x", @typechecker.Ty::i32()), ("y", @typechecker.Ty::i32())],
        return_ty: @typechecker.Ty::i32(),
        body: @hir.HirExpr::BinOpE(
          @hir.BinOp::Add,
          @hir.HirExpr::Var("x", @typechecker.Ty::i32(), s()),
          @hir.HirExpr::Var("y", @typechecker.Ty::i32(), s()),
          @typechecker.Ty::i32(),
          s(),
        ),
        span: s(),
      },
    ],
    data_types: Array::new(),
    entry_points: Array::new(),
  }
  match @mir.lower_hir_to_mir(hir) {
    Ok(prog) => {
      assert_eq(prog.functions.length(), 1)
      let f = prog.functions[0]
      assert_eq(f.name, "add")
      assert_eq(f.params.length(), 2)
      assert_eq(f.return_ty, @mir.MirType::I32)
      assert_true(f.return_expr is Some(_))
    }
    Err(errs) => fail("lowering failed: " + errs.length().to_string() + " errors")
  }
}

///|
test "lower let expression" {
  let bind_lit = @hir.HirExpr::Lit(
    @hir.HirLit::Int(42L),
    @typechecker.Ty::i32(),
    s(),
  )
  let body_expr = @hir.HirExpr::BinOpE(
    @hir.BinOp::Add,
    @hir.HirExpr::Var("x", @typechecker.Ty::i32(), s()),
    @hir.HirExpr::Lit(@hir.HirLit::Int(1L), @typechecker.Ty::i32(), s()),
    @typechecker.Ty::i32(),
    s(),
  )
  let hir = @hir.HirProgram::{
    functions: [
      @hir.HirFunction::{
        name: "f",
        params: Array::new(),
        return_ty: @typechecker.Ty::i32(),
        body: @hir.HirExpr::Let(
          [("x", bind_lit)],
          body_expr,
          @typechecker.Ty::i32(),
          s(),
        ),
        span: s(),
      },
    ],
    data_types: Array::new(),
    entry_points: Array::new(),
  }
  match @mir.lower_hir_to_mir(hir) {
    Ok(prog) => {
      assert_eq(prog.functions.length(), 1)
      assert_true(prog.functions[0].body.length() > 0)
    }
    Err(_) => fail("lowering failed")
  }
}
