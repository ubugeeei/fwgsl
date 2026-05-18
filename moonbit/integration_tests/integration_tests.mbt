///|
/// End-to-end compiler pipeline tests for fwgsl.
///
/// Mirrors `crates/fwgsl_integration_tests/tests/integration.rs`. The
/// Rust file drives the full parse -> semantic -> AST lowering ->
/// MIR -> WGSL pipeline on a curated set of example programs.
///
/// Until the AST -> HIR lowering is fully ported, the integration tests
/// land here as parse + semantic-registration checks against
/// `examples/*.fwgsl` — the strongest signal we can give right now that
/// the front-end still accepts the existing sample programs.

///|
/// Helper that runs parse + semantic registration and returns whether
/// the pipeline produced any error-severity diagnostics.
pub fn pipeline_has_errors(source : String) -> Bool {
  let (_, sink) = @cli.compile_source(source)
  sink.has_errors()
}
