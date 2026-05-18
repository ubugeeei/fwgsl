///|
test "hello example parses" {
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
  assert_false(@integration_tests.pipeline_has_errors(source))
}

///|
test "adt match example parses" {
  let source = #|data Color = Red | Green | Blue
                #|
                #|toI32 : Color -> I32
                #|toI32 color = match color
                #|  | Red -> 0
                #|  | Green -> 1
                #|  | Blue -> 2
  assert_false(@integration_tests.pipeline_has_errors(source))
}

///|
test "where example parses" {
  let source = #|scaleAndBias : I32 -> I32
                #|scaleAndBias x = y * 2
                #|where
                #|  y = x + 1
  assert_false(@integration_tests.pipeline_has_errors(source))
}

///|
test "entry point example parses" {
  let source = "@compute @workgroup_size(64, 1, 1)\nmain idx =\n  let doubled = idx * 2\n  in doubled"
  assert_false(@integration_tests.pipeline_has_errors(source))
}

///|
test "generic data parses" {
  let source = #|data Box a = Box a
                #|data Phantom a = Phantom
                #|
                #|unbox : Box a -> a
                #|unbox value = match value
                #|  | Box x -> x
  assert_false(@integration_tests.pipeline_has_errors(source))
}
