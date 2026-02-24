//! Binary entry point for the fwgsl language server.
//!
//! Starts a tower-lsp server over stdin/stdout, suitable for use by
//! editors and IDEs that support the Language Server Protocol.

use fwgsl_language_server::FwgslBackend;
use tower_lsp::{LspService, Server};

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(FwgslBackend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
