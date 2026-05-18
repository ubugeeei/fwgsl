///|
test "wasm_check on well-formed source returns empty array" {
  let json = @wasm.wasm_check("data Color = Red")
  assert_eq(json, "[]")
}

///|
test "wasm_check returns a JSON array" {
  let json = @wasm.wasm_check("f x = y")
  // Starts with '[' and ends with ']'.
  assert_eq(json.charcode_at(0), 0x5B)
  assert_eq(json.charcode_at(json.length() - 1), 0x5D)
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
}
