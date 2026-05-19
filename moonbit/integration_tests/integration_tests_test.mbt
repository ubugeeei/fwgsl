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
test "unbound variable diagnostic points at the offending reference" {
  // `y` lives at offset 6 (after `f x = `). The diagnostic span should
  // pick up exactly that token, not the start of the file or the whole
  // function body.
  let source = "f x = y"
  let r = @integration_tests.run_pipeline(source)
  let mut got_span : (UInt, UInt) = (999U, 999U)
  for d in r.diagnostics {
    if d.message.contains("Unbound variable") && d.labels.length() > 0 {
      let l = d.labels[0]
      got_span = (l.span.start, l.span.end)
      break
    }
  }
  assert_eq(got_span, (6U, 7U))
}

///|
test "unbound variable diagnostic carries scope help" {
  let r = @integration_tests.run_pipeline("f x = y")
  let mut found_help = false
  for d in r.diagnostics {
    if d.message.contains("Unbound variable") {
      match d.help {
        Some(h) => {
          if h.contains("parameter") || h.contains("let") || h.contains("where") {
            found_help = true
          }
        }
        None => ()
      }
    }
  }
  assert_true(found_help)
}

///|
test "type mismatch between signature and body produces a diagnostic" {
  // `f : I32 -> I32` but the body uses a comparison that returns Bool,
  // so the inferred body type clashes with the declared signature.
  let source = "f : I32 -> I32\nf x = x == 0"
  let r = @integration_tests.run_pipeline(source)
  assert_true(r.has_errors)
  match r.wgsl {
    None => ()
    Some(_) => fail("expected no WGSL on type-mismatch")
  }
  let mut found = false
  for d in r.diagnostics {
    if d.message.contains("Type mismatch") {
      found = true
      break
    }
  }
  assert_true(found)
}

///|
test "type mismatch diagnostic carries an annotation help" {
  let source = "f : I32 -> I32\nf x = x == 0"
  let r = @integration_tests.run_pipeline(source)
  let mut found_help = false
  for d in r.diagnostics {
    if d.message.contains("Type mismatch") {
      match d.help {
        Some(h) =>
          if h.contains("annotation") || h.contains("agree") {
            found_help = true
          }
        None => ()
      }
    }
  }
  assert_true(found_help)
}

///|
test "unknown constructor in a match arm produces a diagnostic" {
  // `Triangle` isn't a constructor of `Color`; the match arm references
  // an undefined name.
  let source = #|data Color = Red | Green | Blue
                #|
                #|toI32 : Color -> I32
                #|toI32 color = match color
                #|  | Triangle -> 0
                #|  | _ -> 1
  let r = @integration_tests.run_pipeline(source)
  assert_true(r.has_errors)
  let mut found = false
  for d in r.diagnostics {
    if d.message.contains("Unknown constructor") &&
      d.message.contains("Triangle") {
      found = true
      break
    }
  }
  assert_true(found)
}

///|
test "parser error: missing `=` in a function declaration" {
  // The recursive-descent parser surfaces a structured `expected ...,
  // found ...` diagnostic when a token doesn't match.
  let r = @integration_tests.run_pipeline("f x")
  assert_true(r.has_errors)
  let mut found = false
  for d in r.diagnostics {
    if d.message.contains("expected") {
      found = true
      break
    }
  }
  assert_true(found)
}

///|
test "multiple unrelated errors all surface at once" {
  let source = "f x = y\ng x = z"
  let r = @integration_tests.run_pipeline(source)
  let mut unbound_count = 0
  for d in r.diagnostics {
    if d.message.contains("Unbound variable") {
      unbound_count = unbound_count + 1
    }
  }
  // Both `y` and `z` are unbound; the analyser shouldn't bail out on
  // the first error.
  assert_true(unbound_count >= 2)
}

///|
test "every error diagnostic carries at least one label" {
  let source = "f x = y\ng : Bool\ng = 1"
  let r = @integration_tests.run_pipeline(source)
  assert_true(r.has_errors)
  let mut error_count = 0
  for d in r.diagnostics {
    if d.severity == @diagnostics.Severity::Error {
      error_count = error_count + 1
      assert_true(d.labels.length() >= 1)
    }
  }
  assert_true(error_count >= 1)
}

///|
test "successful pipeline reports zero error diagnostics" {
  let source = "main : I32 -> I32\nmain x = x + 1"
  let r = @integration_tests.run_pipeline(source)
  assert_false(r.has_errors)
  for d in r.diagnostics {
    assert_true(d.severity != @diagnostics.Severity::Error)
  }
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
