//! Binary entry point for the shadml language server.
//!
//! Starts a tower-lsp server over stdin/stdout, suitable for use by
//! editors and IDEs that support the Language Server Protocol.

use shadml_language_server::ShadmlBackend;
use tower_lsp::{LspService, Server};

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(ShadmlBackend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
