///|
/// End-to-end compiler pipeline tests for fwgsl.
///
/// Mirrors `crates/fwgsl_integration_tests/tests/integration.rs`. Drives
/// the full parse -> semantic -> AST lowering -> MIR lowering ->
/// WGSL codegen pipeline on a curated set of example programs.

///|
/// Run the full pipeline. Returns the `CompileResult` carrying the WGSL
/// output (or `None` on error) along with the combined diagnostics.
pub fn run_pipeline(source : String) -> @cli.CompileResult {
  @cli.compile_source(source)
}

///|
/// Whether the pipeline produced any error-severity diagnostic.
pub fn pipeline_has_errors(source : String) -> Bool {
  run_pipeline(source).has_errors
}

///|
/// Run the pipeline and require that it emit WGSL. Returns the emitted
/// WGSL text, or fails with the joined diagnostic messages on error.
pub fn compile_or_fail(source : String) -> String {
  let r = run_pipeline(source)
  match r.wgsl {
    Some(wgsl) => wgsl
    None => {
      let mut msg = "pipeline failed:"
      for d in r.diagnostics {
        msg = msg + "\n  - " + d.message
      }
      abort(msg)
    }
  }
}
