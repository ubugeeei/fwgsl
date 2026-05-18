///|
test "emit simple function" {
  let prog = @mir.MirProgram::{
    structs: Array::new(),
    functions: [
      @mir.MirFunction::{
        name: "add",
        params: [
          @mir.MirParam::{ name: "x", ty: @mir.MirType::I32 },
          @mir.MirParam::{ name: "y", ty: @mir.MirType::I32 },
        ],
        return_ty: @mir.MirType::I32,
        body: Array::new(),
        return_expr: Some(
          @mir.MirExpr::BinOpE(
            @mir.MirBinOp::Add,
            @mir.MirExpr::VarExpr("x", @mir.MirType::I32),
            @mir.MirExpr::VarExpr("y", @mir.MirType::I32),
            @mir.MirType::I32,
          ),
        ),
      },
    ],
    entry_points: Array::new(),
  }
  let wgsl = @wgsl_codegen.emit_wgsl(prog)
  assert_true(wgsl.contains("fn add(x: i32, y: i32) -> i32"))
  assert_true(wgsl.contains("return (x + y)"))
}

///|
test "emit void function" {
  let prog = @mir.MirProgram::{
    structs: Array::new(),
    functions: [
      @mir.MirFunction::{
        name: "do_nothing",
        params: Array::new(),
        return_ty: @mir.MirType::Unit,
        body: Array::new(),
        return_expr: None,
      },
    ],
    entry_points: Array::new(),
  }
  let wgsl = @wgsl_codegen.emit_wgsl(prog)
  assert_true(wgsl.contains("fn do_nothing() {"))
  assert_false(wgsl.contains("->"))
}

///|
test "emit struct" {
  let prog = @mir.MirProgram::{
    structs: [
      @mir.MirStruct::{
        name: "Particle",
        fields: [
          @mir.MirField::{ name: "tag", ty: @mir.MirType::U32 },
          @mir.MirField::{
            name: "position",
            ty: @mir.MirType::MirVec(3, @mir.MirType::F32),
          },
          @mir.MirField::{ name: "life", ty: @mir.MirType::F32 },
        ],
      },
    ],
    functions: Array::new(),
    entry_points: Array::new(),
  }
  let wgsl = @wgsl_codegen.emit_wgsl(prog)
  assert_true(wgsl.contains("struct Particle {"))
  assert_true(wgsl.contains("tag: u32,"))
  assert_true(wgsl.contains("position: vec3<f32>,"))
  assert_true(wgsl.contains("life: f32,"))
}

///|
test "emit compute entry point" {
  let prog = @mir.MirProgram::{
    structs: Array::new(),
    functions: Array::new(),
    entry_points: [
      @mir.MirEntryPoint::{
        name: "main",
        stage: @mir.ShaderStage::Compute,
        workgroup_size: Some((64, 1, 1)),
        params: Array::new(),
        builtins: [
          (
            "gid",
            @mir.BuiltinBinding::GlobalInvocationId,
            @mir.MirType::MirVec(3, @mir.MirType::U32),
          ),
        ],
        return_ty: @mir.MirType::Unit,
        body: [
          @mir.MirStmt::Let(
            "idx",
            @mir.MirType::U32,
            @mir.MirExpr::FieldAccess(
              @mir.MirExpr::VarExpr(
                "gid",
                @mir.MirType::MirVec(3, @mir.MirType::U32),
              ),
              "x",
              @mir.MirType::U32,
            ),
          ),
        ],
        return_expr: None,
      },
    ],
  }
  let wgsl = @wgsl_codegen.emit_wgsl(prog)
  assert_true(wgsl.contains("@compute @workgroup_size(64, 1, 1)"))
  assert_true(wgsl.contains("@builtin(global_invocation_id) gid: vec3<u32>"))
  assert_true(wgsl.contains("let idx: u32 = gid.x;"))
}

///|
test "emit literals" {
  let prog = @mir.MirProgram::{
    structs: Array::new(),
    functions: [
      @mir.MirFunction::{
        name: "lits",
        params: Array::new(),
        return_ty: @mir.MirType::Unit,
        body: [
          @mir.MirStmt::Let(
            "a",
            @mir.MirType::I32,
            @mir.MirExpr::Lit(@mir.MirLit::I32Lit(-5)),
          ),
          @mir.MirStmt::Let(
            "b",
            @mir.MirType::U32,
            @mir.MirExpr::Lit(@mir.MirLit::U32Lit(10)),
          ),
          @mir.MirStmt::Let(
            "c",
            @mir.MirType::F32,
            @mir.MirExpr::Lit(@mir.MirLit::F32Lit(2.0)),
          ),
          @mir.MirStmt::Let(
            "d",
            @mir.MirType::Bool,
            @mir.MirExpr::Lit(@mir.MirLit::BoolLit(true)),
          ),
        ],
        return_expr: None,
      },
    ],
    entry_points: Array::new(),
  }
  let wgsl = @wgsl_codegen.emit_wgsl(prog)
  assert_true(wgsl.contains("let a: i32 = (-5i);"))
  assert_true(wgsl.contains("let b: u32 = 10u;"))
  assert_true(wgsl.contains("let c: f32 = 2.0"))
  assert_true(wgsl.contains("let d: bool = true;"))
}
