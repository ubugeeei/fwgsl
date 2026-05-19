///|
test "DocumentStore tracks open/update/close" {
  let store = @language_server.DocumentStore::new()
  store.open("file:///foo.fwgsl", "main = 1", 1)
  match store.get("file:///foo.fwgsl") {
    Some(doc) => {
      assert_eq(doc.text, "main = 1")
      assert_eq(doc.version, 1)
    }
    None => fail("expected document to be open")
  }
  store.update("file:///foo.fwgsl", "main = 2", 2)
  match store.get("file:///foo.fwgsl") {
    Some(doc) => {
      assert_eq(doc.text, "main = 2")
      assert_eq(doc.version, 2)
    }
    None => fail("expected document still open after update")
  }
  store.close("file:///foo.fwgsl")
  match store.get("file:///foo.fwgsl") {
    None => ()
    Some(_) => fail("expected document closed")
  }
}

///|
test "compute_diagnostics reports semantic errors" {
  let diags = @language_server.compute_diagnostics("f x = y")
  assert_true(diags.length() > 0)
  let first = diags[0]
  assert_eq(first.severity, @language_server.LspSeverity::ErrorSev)
  // The unbound `y` lives on line 1 (0-based 0).
  assert_eq(first.start_line, 0)
}

///|
test "compute_diagnostics on well-formed source is empty" {
  let diags = @language_server.compute_diagnostics(
    "data Color = Red | Green | Blue",
  )
  assert_eq(diags.length(), 0)
}

///|
test "diagnostic_to_json renders severity and range" {
  let d = @language_server.LspDiagnostic::{
    message: "oops",
    severity: @language_server.LspSeverity::ErrorSev,
    start_line: 0,
    start_character: 5,
    end_line: 0,
    end_character: 8,
    help: Some("try this"),
  }
  let json = @language_server.diagnostic_to_json(d)
  assert_true(json.contains("\"severity\":1"))
  assert_true(json.contains("\"line\":0"))
  assert_true(json.contains("\"character\":5"))
  assert_true(json.contains("\"character\":8"))
  assert_true(json.contains("\"message\":\"oops\""))
  assert_true(json.contains("\"help\":\"try this\""))
}

///|
test "diagnostics_json_for produces a JSON array" {
  let json = @language_server.diagnostics_json_for("f x = y")
  assert_true(json.length() >= 2)
  assert_eq(json.unsafe_charcode_at(0), 0x5B) // '['
  assert_eq(json.unsafe_charcode_at(json.length() - 1), 0x5D) // ']'
}

///|
test "semantic_tokens_for highlights keywords" {
  let tokens = @language_server.semantic_tokens_for("let x = 1 in x")
  let mut saw_keyword = false
  for t in tokens {
    if t.kind == @ide.SemanticTokenKind::TokKeyword {
      saw_keyword = true
      break
    }
  }
  assert_true(saw_keyword)
}
