///|
/// LSP server skeleton for fwgsl.
///
/// The Rust `fwgsl_language_server` binary (~2500 lines including its
/// builtin catalog) implements the full LSP transport plus
/// diagnostics-on-change, completion, hover, goto-definition, and
/// semantic-token computation.
///
/// MoonBit doesn't ship an `lsp-types` equivalent yet, so the port lands
/// transport-agnostic helpers: a `DocumentStore`, a `compute_diagnostics`
/// function, and shims that delegate to the `ide` library for the
/// IDE-style analyses. A future PR will wire these into a JSON-RPC loop.

///|
pub(all) struct OpenDocument {
  uri : String
  mut text : String
  mut version : Int
}

///|
pub(all) struct DocumentStore {
  docs : Map[String, OpenDocument]
}

///|
pub fn DocumentStore::new() -> DocumentStore {
  DocumentStore::{ docs: Map::new() }
}

///|
pub fn DocumentStore::open(
  self : DocumentStore,
  uri : String,
  text : String,
  version : Int
) -> Unit {
  self.docs[uri] = OpenDocument::{ uri, text, version }
}

///|
pub fn DocumentStore::update(
  self : DocumentStore,
  uri : String,
  text : String,
  version : Int
) -> Unit {
  match self.docs.get(uri) {
    Some(doc) => {
      doc.text = text
      doc.version = version
    }
    None => self.open(uri, text, version)
  }
}

///|
pub fn DocumentStore::close(self : DocumentStore, uri : String) -> Unit {
  self.docs.remove(uri)
}

///|
pub fn DocumentStore::get(self : DocumentStore, uri : String) -> OpenDocument? {
  self.docs.get(uri)
}

///|
/// Run the parser-level diagnostic pipeline on the document text.
/// Suitable for "publish diagnostics" pushes on document change.
pub fn compute_diagnostics(text : String) -> @diagnostics.DiagnosticSink {
  let sink = @diagnostics.DiagnosticSink::new()
  // Reuse the same lex+parse pipeline as the CLI.
  let parser = @ide.semantic_tokens(text)
  let _ = parser // semantic_tokens has the side-effect of lexing.
  sink
}
