///|
test "wasm_check on well-formed source returns empty array" {
  let json = @wasm.wasm_check("data Color = Red")
  assert_eq(json, "[]")
}

///|
test "wasm_check returns a JSON array" {
  let json = @wasm.wasm_check("f x = y")
  // Starts with '[' and ends with ']'.
  assert_eq(json.unsafe_charcode_at(0), 0x5B)
  assert_eq(json.unsafe_charcode_at(json.length() - 1), 0x5D)
}

///|
test "wasm_compile returns JSON object with diagnostics + wgsl fields" {
  let json = @wasm.wasm_compile("data X = X")
  assert_true(json.contains("\"diagnostics\":"))
  assert_true(json.contains("\"wgsl\":"))
}

///|
test "wasm_compile preserves diagnostics on semantic error" {
  let json = @wasm.wasm_compile("f x = y")
  assert_true(json.contains("\"severity\":\"error\""))
  assert_true(json.contains("Unbound variable"))
  // No WGSL when there's an error.
  assert_true(json.contains("\"wgsl\":null"))
}

///|
test "wasm_compile emits WGSL on a well-formed program" {
  let json = @wasm.wasm_compile("main : I32 -> I32\nmain x = x + 1")
  // `wgsl` is now a real string, not just `null`.
  assert_true(json.contains("\"wgsl\":\""))
  assert_true(json.contains("fn main"))
}

///|
test "wasm_check surfaces lint warnings" {
  let json = @wasm.wasm_check("f x = let y = x + 1 in x")
  // Unused-let warning surfaces.
  assert_true(json.contains("\"severity\":\"warning\""))
  assert_true(json.contains("Unused let binding"))
}

///|
test "wasm_compile surfaces lint warnings alongside the WGSL" {
  let json = @wasm.wasm_compile(
    "main : I32 -> I32\nmain x = let y = x + 1 in x",
  )
  assert_true(json.contains("\"severity\":\"warning\""))
  // Even with a lint warning the WGSL still comes through because the
  // pipeline succeeded at the compile stage.
  assert_true(json.contains("\"wgsl\":\""))
}
