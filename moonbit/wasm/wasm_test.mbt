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

///|
test "compile is an alias for wasm_compile" {
  let a = @wasm.compile("main : I32 -> I32\nmain x = x + 1")
  let b = @wasm.wasm_compile("main : I32 -> I32\nmain x = x + 1")
  assert_eq(a, b)
}

///|
test "format canonicalises whitespace through the wasm export" {
  let out = @wasm.format("f    x   =  x  +  1")
  assert_eq(out, "f x = x + 1\n")
}

///|
test "editor_completions returns a JSON array with at least keywords" {
  let json = @wasm.editor_completions("", 1, 1)
  // Empty source still surfaces the keyword seed.
  assert_eq(json.unsafe_charcode_at(0), 0x5B) // '['
  assert_eq(json.unsafe_charcode_at(json.length() - 1), 0x5D) // ']'
  assert_true(json.contains("\"label\":\"let\""))
  assert_true(json.contains("\"label\":\"data\""))
  // Completions carry Monaco-style ranges.
  assert_true(json.contains("\"startLineNumber\":1"))
}

///|
test "editor_hover returns `null` for empty input" {
  let json = @wasm.editor_hover("", 1, 1)
  assert_eq(json, "null")
}

///|
test "editor_hover returns a JSON object with contents + range" {
  let source = "add : I32 -> I32 -> I32\nadd x y = x + y"
  // (line 2, column 1) lands on the `add` definition.
  let json = @wasm.editor_hover(source, 2, 1)
  assert_true(json.contains("\"contents\":"))
  assert_true(json.contains("\"range\":"))
  // Monaco range schema (1-based).
  assert_true(json.contains("\"startLineNumber\":"))
  assert_true(json.contains("\"endLineNumber\":"))
}

///|
test "editor_definition returns an array of `{ range }` locations" {
  let source = "scale x = x + 1\nuse x = scale x"
  // (line 2, column 9) lands on the `scale` call site.
  let json = @wasm.editor_definition(source, 2, 9)
  assert_true(json.contains("\"range\":"))
  assert_eq(json.unsafe_charcode_at(0), 0x5B)
}

///|
test "editor_definition returns `[]` when nothing matches" {
  // Column 0 on an empty file: no token, no definition.
  let json = @wasm.editor_definition("", 1, 1)
  assert_eq(json, "[]")
}

///|
test "editor_references is currently an alias for editor_definition" {
  let source = "scale x = x + 1\nuse x = scale x"
  let a = @wasm.editor_references(source, 2, 9, true)
  let b = @wasm.editor_definition(source, 2, 9)
  assert_eq(a, b)
}
