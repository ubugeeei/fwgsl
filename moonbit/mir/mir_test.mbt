///|
test "mir type display scalars" {
  assert_eq(@mir.MirType::I32.display(), "i32")
  assert_eq(@mir.MirType::U32.display(), "u32")
  assert_eq(@mir.MirType::F32.display(), "f32")
  assert_eq(@mir.MirType::Bool.display(), "bool")
  assert_eq(@mir.MirType::Unit.display(), "void")
}

///|
test "mir type display vec mat array" {
  assert_eq(@mir.MirType::MirVec(3, @mir.MirType::F32).display(), "vec3<f32>")
  assert_eq(@mir.MirType::MirMat(4, 4, @mir.MirType::F32).display(), "mat4x4<f32>")
  assert_eq(
    @mir.MirType::MirArray(@mir.MirType::F32, 16).display(),
    "array<f32, 16>",
  )
  assert_eq(@mir.MirType::MirStructTy("MyStruct").display(), "MyStruct")
}

///|
test "shader stage display" {
  assert_eq(@mir.ShaderStage::Compute.display(), "compute")
  assert_eq(@mir.ShaderStage::Vertex.display(), "vertex")
  assert_eq(@mir.ShaderStage::Fragment.display(), "fragment")
}

///|
test "builtin binding display" {
  assert_eq(
    @mir.BuiltinBinding::GlobalInvocationId.display(),
    "global_invocation_id",
  )
  assert_eq(@mir.BuiltinBinding::Position.display(), "position")
}

///|
test "mir bin op display" {
  assert_eq(@mir.MirBinOp::Add.display(), "+")
  assert_eq(@mir.MirBinOp::Neq.display(), "!=")
  assert_eq(@mir.MirBinOp::Shl.display(), "<<")
}

///|
test "mir unary op display" {
  assert_eq(@mir.MirUnaryOp::Neg.display(), "-")
  assert_eq(@mir.MirUnaryOp::Not.display(), "!")
  assert_eq(@mir.MirUnaryOp::BitNot.display(), "~")
}
