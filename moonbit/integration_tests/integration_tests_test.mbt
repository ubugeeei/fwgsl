///|
test "hello example emits a `main` function in WGSL" {
  let source = #|add : I32 -> I32 -> I32
                #|add x y = x + y
                #|
                #|double : I32 -> I32
                #|double x = x * 2
                #|
                #|main : I32 -> I32
                #|main x =
                #|  let y = double x
                #|  in y + 1
  let wgsl = @integration_tests.compile_or_fail(source)
  assert_true(wgsl.contains("fn main("))
  assert_true(wgsl.contains("fn add("))
  assert_true(wgsl.contains("fn double("))
}

///|
test "adt match example desugars to if-chain" {
  let source = #|data Color = Red | Green | Blue
                #|
                #|toI32 : Color -> I32
                #|toI32 color = match color
                #|  | Red -> 0
                #|  | Green -> 1
                #|  | Blue -> 2
  let wgsl = @integration_tests.compile_or_fail(source)
  assert_true(wgsl.contains("fn toI32("))
  // Match on enum tags lowers to an if-chain.
  assert_true(wgsl.contains("if ("))
}

///|
test "where example becomes let binding" {
  // `where` lives on the same line as the function body so layout
  // resolution attaches it; the column-zero variant is a separate
  // case tracked by the parser/layout tests.
  let source = #|scaleAndBias : I32 -> I32
                #|scaleAndBias x = y * 2 where
                #|  y = x + 1
  let wgsl = @integration_tests.compile_or_fail(source)
  assert_true(wgsl.contains("fn scaleAndBias"))
  assert_true(wgsl.contains("let y"))
}

///|
test "compute entry point gets the @compute attribute" {
  let source = "@compute @workgroup_size(64, 1, 1)\nmain idx =\n  let doubled = idx * 2\n  in doubled"
  let wgsl = @integration_tests.compile_or_fail(source)
  assert_true(wgsl.contains("@compute @workgroup_size(64, 1, 1)"))
  assert_true(wgsl.contains("@builtin(global_invocation_id)"))
  assert_true(wgsl.contains("fn main("))
}

///|
test "generic data parses without errors" {
  let source = #|data Box a = Box a
                #|data Phantom a = Phantom
                #|
                #|unbox : Box a -> a
                #|unbox value = match value
                #|  | Box x -> x
  assert_false(@integration_tests.pipeline_has_errors(source))
}

///|
test "vertex entry point emits @vertex" {
  let source = "@vertex\nvs_main vid =\n  let pos = vid + 1\n  in pos"
  let wgsl = @integration_tests.compile_or_fail(source)
  assert_true(wgsl.contains("@vertex"))
}

///|
test "fragment entry point emits @fragment" {
  let source = "@fragment\nfs_main x =\n  let c = x + 1\n  in c"
  let wgsl = @integration_tests.compile_or_fail(source)
  assert_true(wgsl.contains("@fragment"))
}

///|
test "unbound variable produces a diagnostic" {
  let r = @integration_tests.run_pipeline("f x = y")
  assert_true(r.has_errors)
  match r.wgsl {
    None => ()
    Some(_) => fail("expected no WGSL on semantic error")
  }
  let mut found_msg = false
  for d in r.diagnostics {
    if d.message.contains("Unbound variable") {
      found_msg = true
      break
    }
  }
  assert_true(found_msg)
}

///|
test "if expression compiles to an if-statement" {
  let source = "f x = if x == 0 then 1 else 2"
  let wgsl = @integration_tests.compile_or_fail(source)
  assert_true(wgsl.contains("if ("))
  assert_true(wgsl.contains("} else {"))
}

///|
test "operator precedence respected through codegen" {
  let source = "f x = 1 + 2 * 3"
  let wgsl = @integration_tests.compile_or_fail(source)
  // 2 * 3 must be the inner application; the outer expression is 1 + (...).
  assert_true(wgsl.contains("(1i + (2i * 3i))"))
}
