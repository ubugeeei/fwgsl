//! Language Server Protocol implementation for fwgsl.
//!
//! Provides IDE features (diagnostics, completions, hover, semantic tokens,
//! go-to-definition) by integrating with the fwgsl compiler pipeline
//! (`fwgsl_parser`, `fwgsl_semantic`, `fwgsl_typechecker`).

use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use fwgsl_diagnostics::{Diagnostic as FwgslDiag, Severity as FwgslSeverity};
use fwgsl_parser::lexer::Token;
use fwgsl_parser::{lex, Parser};
use fwgsl_semantic::SemanticAnalyzer;
use fwgsl_span::Span;
use fwgsl_syntax::SyntaxKind;

// ============================================================================
// Semantic token legend
// ============================================================================

/// The set of semantic token types this server supports.
pub const TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::KEYWORD,     // 0
    SemanticTokenType::TYPE,        // 1
    SemanticTokenType::FUNCTION,    // 2
    SemanticTokenType::VARIABLE,    // 3
    SemanticTokenType::NUMBER,      // 4
    SemanticTokenType::STRING,      // 5
    SemanticTokenType::COMMENT,     // 6
    SemanticTokenType::OPERATOR,    // 7
    SemanticTokenType::ENUM,        // 8
    SemanticTokenType::ENUM_MEMBER, // 9
    SemanticTokenType::DECORATOR,   // 10
];

/// The set of semantic token modifiers this server supports.
pub const TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[];

/// Build the `SemanticTokensLegend` from our constants.
pub fn semantic_tokens_legend() -> SemanticTokensLegend {
    SemanticTokensLegend {
        token_types: TOKEN_TYPES.to_vec(),
        token_modifiers: TOKEN_MODIFIERS.to_vec(),
    }
}

// ============================================================================
// Semantic token indices (must match TOKEN_TYPES order)
// ============================================================================

const TOK_KEYWORD: u32 = 0;
const TOK_TYPE: u32 = 1;
#[allow(dead_code)]
const TOK_FUNCTION: u32 = 2;
const TOK_VARIABLE: u32 = 3;
const TOK_NUMBER: u32 = 4;
const TOK_STRING: u32 = 5;
const TOK_COMMENT: u32 = 6;
const TOK_OPERATOR: u32 = 7;
#[allow(dead_code)]
const TOK_ENUM: u32 = 8;
#[allow(dead_code)]
const TOK_ENUM_MEMBER: u32 = 9;
const TOK_DECORATOR: u32 = 10;

// ============================================================================
// Backend
// ============================================================================

/// The fwgsl language server backend.
pub struct FwgslBackend {
    /// The LSP client used to send notifications (e.g. diagnostics).
    client: Client,
    /// In-memory document store: URI -> document text.
    documents: DashMap<Url, String>,
}

impl FwgslBackend {
    /// Create a new backend with the given LSP client.
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
        }
    }

    // -- Diagnostics ---------------------------------------------------------

    /// Lex, parse, and semantically analyze the document, then publish
    /// diagnostics back to the client.
    async fn run_diagnostics(&self, uri: Url, text: &str) {
        let mut all_diagnostics: Vec<tower_lsp::lsp_types::Diagnostic> = Vec::new();

        // Phase 1: Parse
        let mut parser = Parser::new(text);
        let program = parser.parse_program();

        // Collect parser diagnostics
        for diag in parser.diagnostics().iter() {
            if let Some(lsp_diag) = fwgsl_diag_to_lsp(diag, text) {
                all_diagnostics.push(lsp_diag);
            }
        }

        // Phase 2: Semantic analysis
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program);

        for diag in analyzer.diagnostics().iter() {
            if let Some(lsp_diag) = fwgsl_diag_to_lsp(diag, text) {
                all_diagnostics.push(lsp_diag);
            }
        }

        self.client
            .publish_diagnostics(uri, all_diagnostics, None)
            .await;
    }
}

// ============================================================================
// LanguageServer trait implementation
// ============================================================================

#[tower_lsp::async_trait]
impl LanguageServer for FwgslBackend {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".into(), ":".into(), "@".into()]),
                    resolve_provider: Some(false),
                    ..Default::default()
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            legend: semantic_tokens_legend(),
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                            range: None,
                            ..Default::default()
                        },
                    ),
                ),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "fwgsl-lsp".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "fwgsl language server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    // -- Document synchronization -------------------------------------------

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let text = params.text_document.text.clone();
        self.documents.insert(uri.clone(), text.clone());
        self.run_diagnostics(uri, &text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        // We use TextDocumentSyncKind::FULL, so the first change contains
        // the full new text.
        if let Some(change) = params.content_changes.into_iter().next() {
            let text = change.text;
            self.documents.insert(uri.clone(), text.clone());
            self.run_diagnostics(uri, &text).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.remove(&uri);
        // Clear diagnostics for the closed document.
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    // -- Completion ---------------------------------------------------------

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        let items = build_completions(&text, pos);
        Ok(Some(CompletionResponse::Array(items)))
    }

    // -- Hover --------------------------------------------------------------

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        Ok(build_hover(&text, pos))
    }

    // -- Go to definition ---------------------------------------------------

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        Ok(build_goto_definition(uri, &text, pos))
    }

    // -- Semantic tokens (full) ---------------------------------------------

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        let tokens = build_semantic_tokens(&text);
        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: tokens,
        })))
    }
}

// ============================================================================
// Completions
// ============================================================================

/// Build a list of completion items based on context.
pub fn build_completions(source: &str, _pos: Position) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // Keyword completions
    let keywords = [
        ("match", "Pattern matching expression"),
        ("let", "Let binding"),
        ("in", "Body of a let expression"),
        ("if", "Conditional expression"),
        ("then", "Then branch of if expression"),
        ("else", "Else branch of if expression"),
        ("where", "Local definitions"),
        ("data", "Algebraic data type declaration"),
        ("type", "Type alias declaration"),
        ("class", "Type class declaration"),
        ("instance", "Type class instance"),
        ("module", "Module declaration"),
        ("import", "Import declaration"),
        ("do", "Do notation block"),
        ("forall", "Universal quantification"),
        ("case", "Case expression"),
        ("of", "Case arms"),
        ("infixl", "Left-associative fixity"),
        ("infixr", "Right-associative fixity"),
        ("infix", "Non-associative fixity"),
        ("deriving", "Automatic instance derivation"),
    ];

    for (kw, detail) in &keywords {
        items.push(CompletionItem {
            label: kw.to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some(detail.to_string()),
            insert_text: Some(kw.to_string()),
            ..Default::default()
        });
    }

    // Type completions (WGSL-compatible built-in types)
    let types = [
        ("I32", "32-bit signed integer"),
        ("U32", "32-bit unsigned integer"),
        ("F32", "32-bit floating point"),
        ("Bool", "Boolean type"),
        ("Vec", "Vector type (e.g. Vec 3 F32)"),
        ("Mat", "Matrix type (e.g. Mat 4 4 F32)"),
        ("Array", "Array type"),
        ("String", "String type"),
        ("Char", "Character type"),
    ];

    for (ty, detail) in &types {
        items.push(CompletionItem {
            label: ty.to_string(),
            kind: Some(CompletionItemKind::TYPE_PARAMETER),
            detail: Some(detail.to_string()),
            insert_text: Some(ty.to_string()),
            ..Default::default()
        });
    }

    // Built-in function completions (functional)
    let builtins = [
        ("map", "Apply a function to each element"),
        ("filter", "Filter elements by predicate"),
        ("foldl", "Left fold over a structure"),
        ("foldr", "Right fold over a structure"),
        ("id", "Identity function :: a -> a"),
        ("const", "Constant function :: a -> b -> a"),
        (
            "flip",
            "Flip function arguments :: (a -> b -> c) -> b -> a -> c",
        ),
        ("compose", "Function composition (.)"),
        ("pure", "Lift a value into an applicative"),
        ("bind", "Monadic bind (>>=)"),
        ("fmap", "Functor map"),
    ];

    // WGSL built-in functions ($ prefix)
    let wgsl_builtins = [
        ("$vec2", "Construct Vec 2 :: a -> a -> Vec 2 a"),
        ("$vec3", "Construct Vec 3 :: a -> a -> a -> Vec 3 a"),
        ("$vec4", "Construct Vec 4 :: a -> a -> a -> a -> Vec 4 a"),
        ("$sin", "Sine :: F32 -> F32"),
        ("$cos", "Cosine :: F32 -> F32"),
        ("$tan", "Tangent :: F32 -> F32"),
        ("$atan", "Arctangent :: F32 -> F32 -> F32"),
        ("$abs", "Absolute value :: F32 -> F32"),
        ("$fract", "Fractional part :: F32 -> F32"),
        ("$floor", "Floor :: F32 -> F32"),
        ("$ceil", "Ceiling :: F32 -> F32"),
        ("$round", "Round :: F32 -> F32"),
        ("$smoothstep", "Smooth step :: F32 -> F32 -> F32 -> F32"),
        ("$step", "Step :: F32 -> F32 -> F32"),
        ("$mix", "Linear interpolation :: a -> a -> F32 -> a"),
        ("$clamp", "Clamp :: F32 -> F32 -> F32 -> F32"),
        ("$dot", "Dot product :: Vec n a -> Vec n a -> a"),
        ("$cross", "Cross product :: Vec 3 a -> Vec 3 a -> Vec 3 a"),
        ("$length", "Vector length :: Vec n a -> a"),
        ("$normalize", "Normalize :: Vec n a -> Vec n a"),
        ("$reflect", "Reflect :: Vec n a -> Vec n a -> Vec n a"),
        ("$sqrt", "Square root :: F32 -> F32"),
        ("$pow", "Power :: F32 -> F32 -> F32"),
        ("$exp", "Exponential :: F32 -> F32"),
        ("$log", "Natural logarithm :: F32 -> F32"),
        ("$min", "Minimum :: F32 -> F32 -> F32"),
        ("$max", "Maximum :: F32 -> F32 -> F32"),
        ("$sign", "Sign :: F32 -> F32"),
        ("$mod", "Modulo :: F32 -> F32 -> F32"),
    ];

    for (name, detail) in &builtins {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: Some(CompletionItemKind::FUNCTION),
            detail: Some(detail.to_string()),
            insert_text: Some(name.to_string()),
            ..Default::default()
        });
    }

    for (name, detail) in &wgsl_builtins {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: Some(CompletionItemKind::FUNCTION),
            detail: Some(detail.to_string()),
            insert_text: Some(name.to_string()),
            ..Default::default()
        });
    }

    // Add identifiers from the current document as variable completions.
    let tokens = lex(source);
    let mut seen = std::collections::HashSet::new();
    for tok in &tokens {
        match tok.kind {
            SyntaxKind::Ident => {
                let text = tok.text(source);
                if seen.insert(text.to_string()) {
                    items.push(CompletionItem {
                        label: text.to_string(),
                        kind: Some(CompletionItemKind::VARIABLE),
                        detail: Some("local identifier".to_string()),
                        ..Default::default()
                    });
                }
            }
            SyntaxKind::UpperIdent => {
                let text = tok.text(source);
                if seen.insert(text.to_string()) {
                    items.push(CompletionItem {
                        label: text.to_string(),
                        kind: Some(CompletionItemKind::CONSTRUCTOR),
                        detail: Some("type/constructor".to_string()),
                        ..Default::default()
                    });
                }
            }
            _ => {}
        }
    }

    items
}

// ============================================================================
// Hover
// ============================================================================

/// Build hover information for the identifier at the given cursor position.
pub fn build_hover(source: &str, pos: Position) -> Option<Hover> {
    let offset = position_to_offset(source, pos)? as u32;
    let tokens = lex(source);

    // Find the token under the cursor.
    let tok = tokens
        .iter()
        .find(|t| t.span.start <= offset && offset < t.span.end)?;

    match tok.kind {
        SyntaxKind::Ident | SyntaxKind::UpperIdent => {
            let name = tok.text(source);

            // Parse and run semantic analysis to get type information.
            let mut parser = Parser::new(source);
            let program = parser.parse_program();
            let mut analyzer = SemanticAnalyzer::new();
            analyzer.analyze(&program);

            let type_info = if let Some(scheme) = analyzer.env.lookup(name) {
                let ty = analyzer.engine.finalize(&scheme.ty);
                format!("{}", ty)
            } else if let Some(con_info) = analyzer.constructors.get(name) {
                format!("constructor of `{}`", con_info.type_name)
            } else {
                return None;
            };

            let hover_text = format!("```fwgsl\n{} :: {}\n```", name, type_info);
            let range = span_to_range(source, tok.span);

            Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: hover_text,
                }),
                range: Some(range),
            })
        }
        SyntaxKind::KwLet
        | SyntaxKind::KwIn
        | SyntaxKind::KwMatch
        | SyntaxKind::KwIf
        | SyntaxKind::KwThen
        | SyntaxKind::KwElse
        | SyntaxKind::KwWhere
        | SyntaxKind::KwDo
        | SyntaxKind::KwModule
        | SyntaxKind::KwImport
        | SyntaxKind::KwData
        | SyntaxKind::KwType
        | SyntaxKind::KwClass
        | SyntaxKind::KwInstance
        | SyntaxKind::KwForall
        | SyntaxKind::KwCase
        | SyntaxKind::KwOf
        | SyntaxKind::KwInfixl
        | SyntaxKind::KwInfixr
        | SyntaxKind::KwInfix
        | SyntaxKind::KwDeriving => {
            let keyword = tok.text(source);
            let description = keyword_description(keyword);
            let range = span_to_range(source, tok.span);
            Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: format!("**`{}`** - {}", keyword, description),
                }),
                range: Some(range),
            })
        }
        _ => None,
    }
}

/// Returns a human-readable description for a keyword.
fn keyword_description(kw: &str) -> &'static str {
    match kw {
        "let" => "Introduces local bindings",
        "in" => "Marks the body of a let expression",
        "match" => "Pattern matching on a value",
        "case" => "Case expression (alias for match ... of)",
        "of" => "Introduces case arms",
        "if" => "Conditional expression",
        "then" => "Then branch of an if expression",
        "else" => "Else branch of an if expression",
        "where" => "Local definitions attached to a declaration",
        "do" => "Do-notation block for monadic sequencing",
        "module" => "Module declaration",
        "import" => "Import another module",
        "data" => "Algebraic data type declaration",
        "type" => "Type alias declaration",
        "class" => "Type class declaration",
        "instance" => "Type class instance declaration",
        "forall" => "Universal quantification in types",
        "infixl" => "Left-associative operator fixity declaration",
        "infixr" => "Right-associative operator fixity declaration",
        "infix" => "Non-associative operator fixity declaration",
        "deriving" => "Automatically derive type class instances",
        _ => "keyword",
    }
}

// ============================================================================
// Go-to-definition
// ============================================================================

/// Build a go-to-definition response for local bindings.
pub fn build_goto_definition(
    uri: &Url,
    source: &str,
    pos: Position,
) -> Option<GotoDefinitionResponse> {
    let offset = position_to_offset(source, pos)? as u32;
    let tokens = lex(source);

    // Find the identifier token under the cursor.
    let tok = tokens.iter().find(|t| {
        (t.kind == SyntaxKind::Ident || t.kind == SyntaxKind::UpperIdent)
            && t.span.start <= offset
            && offset < t.span.end
    })?;

    let name = tok.text(source);

    // Scan tokens for a definition site: look for `name =` or `name ::`
    // patterns that represent function definitions or type signatures.
    let definition_span = find_definition_span(&tokens, source, name)?;
    let range = span_to_range(source, definition_span);

    Some(GotoDefinitionResponse::Scalar(Location {
        uri: uri.clone(),
        range,
    }))
}

/// Scan the token stream for a definition site of the given name.
/// Looks for patterns like `<name> ::` (type signature) or `<name> <params...> =` (function def).
fn find_definition_span(tokens: &[Token], source: &str, name: &str) -> Option<Span> {
    let non_trivia: Vec<&Token> = tokens
        .iter()
        .filter(|t| {
            !t.kind.is_trivia()
                && t.kind != SyntaxKind::LayoutBraceOpen
                && t.kind != SyntaxKind::LayoutSemicolon
                && t.kind != SyntaxKind::LayoutBraceClose
        })
        .collect();

    for (i, tok) in non_trivia.iter().enumerate() {
        if (tok.kind == SyntaxKind::Ident || tok.kind == SyntaxKind::UpperIdent)
            && tok.text(source) == name
        {
            // Check if next non-trivia token is `::` (type signature)
            if i + 1 < non_trivia.len() && non_trivia[i + 1].kind == SyntaxKind::ColonColon {
                return Some(tok.span);
            }
            // Check for function definition pattern: name followed by params
            // then `=`, within the next tokens (allowing for parameters).
            for next_tok in non_trivia.iter().skip(i + 1).take(19) {
                if next_tok.kind == SyntaxKind::Equals {
                    return Some(tok.span);
                }
                // Stop if we hit something that cannot be a parameter
                if next_tok.kind == SyntaxKind::Eof || next_tok.kind.is_keyword() {
                    break;
                }
            }
        }
    }

    None
}

// ============================================================================
// Semantic tokens
// ============================================================================

/// Build the full set of semantic tokens from lexer output.
pub fn build_semantic_tokens(source: &str) -> Vec<SemanticToken> {
    let tokens = lex(source);
    let classified = classify_tokens(&tokens, source);
    encode_semantic_tokens(&classified, source)
}

/// A classified token ready for semantic-token encoding.
#[derive(Debug, Clone)]
pub struct ClassifiedToken {
    pub span: Span,
    pub token_type: u32,
}

/// Walk the token stream and assign semantic token types.
pub fn classify_tokens(tokens: &[Token], _source: &str) -> Vec<ClassifiedToken> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        let tok = &tokens[i];
        let token_type = match tok.kind {
            // Keywords
            SyntaxKind::KwLet
            | SyntaxKind::KwIn
            | SyntaxKind::KwMatch
            | SyntaxKind::KwIf
            | SyntaxKind::KwThen
            | SyntaxKind::KwElse
            | SyntaxKind::KwWhere
            | SyntaxKind::KwDo
            | SyntaxKind::KwModule
            | SyntaxKind::KwImport
            | SyntaxKind::KwData
            | SyntaxKind::KwType
            | SyntaxKind::KwClass
            | SyntaxKind::KwInstance
            | SyntaxKind::KwForall
            | SyntaxKind::KwCase
            | SyntaxKind::KwOf
            | SyntaxKind::KwInfixl
            | SyntaxKind::KwInfixr
            | SyntaxKind::KwInfix
            | SyntaxKind::KwDeriving => Some(TOK_KEYWORD),

            // Numbers
            SyntaxKind::IntLiteral | SyntaxKind::FloatLiteral => Some(TOK_NUMBER),

            // Strings
            SyntaxKind::StringLiteral | SyntaxKind::CharLiteral => Some(TOK_STRING),

            // Comments
            SyntaxKind::LineComment | SyntaxKind::BlockComment => Some(TOK_COMMENT),

            // Upper identifiers -> type
            SyntaxKind::UpperIdent => Some(TOK_TYPE),

            // Lower identifiers -> variable
            SyntaxKind::Ident => Some(TOK_VARIABLE),

            // `@` starts a decorator/attribute; if followed by an identifier,
            // classify both as decorator.
            SyntaxKind::At => {
                // Peek ahead for identifier, skipping whitespace
                let mut j = i + 1;
                while j < tokens.len() && tokens[j].kind == SyntaxKind::Whitespace {
                    j += 1;
                }
                if j < tokens.len()
                    && (tokens[j].kind == SyntaxKind::Ident
                        || tokens[j].kind == SyntaxKind::UpperIdent)
                {
                    // Emit `@` as decorator
                    result.push(ClassifiedToken {
                        span: tok.span,
                        token_type: TOK_DECORATOR,
                    });
                    // Emit the identifier as decorator too
                    result.push(ClassifiedToken {
                        span: tokens[j].span,
                        token_type: TOK_DECORATOR,
                    });
                    // Skip to after the identifier
                    i = j + 1;
                    continue;
                }
                Some(TOK_OPERATOR)
            }

            // Operators
            _ if tok.kind.is_operator() => Some(TOK_OPERATOR),

            // Punctuation that acts as an operator
            SyntaxKind::Arrow
            | SyntaxKind::FatArrow
            | SyntaxKind::Pipe
            | SyntaxKind::Equals
            | SyntaxKind::ColonColon
            | SyntaxKind::Backslash
            | SyntaxKind::LeftArrow
            | SyntaxKind::Compose => Some(TOK_OPERATOR),

            // Skip trivia and layout tokens
            SyntaxKind::Whitespace
            | SyntaxKind::Newline
            | SyntaxKind::LayoutBraceOpen
            | SyntaxKind::LayoutSemicolon
            | SyntaxKind::LayoutBraceClose
            | SyntaxKind::Eof => None,

            // Everything else: no classification
            _ => None,
        };

        if let Some(tt) = token_type {
            // Only emit tokens that have a non-zero length in the source.
            if tok.span.end > tok.span.start {
                result.push(ClassifiedToken {
                    span: tok.span,
                    token_type: tt,
                });
            }
        }

        i += 1;
    }

    result
}

/// Encode classified tokens into the LSP delta-encoded format.
pub fn encode_semantic_tokens(classified: &[ClassifiedToken], source: &str) -> Vec<SemanticToken> {
    let line_starts = compute_line_starts(source);
    let mut result = Vec::with_capacity(classified.len());
    let mut prev_line: u32 = 0;
    let mut prev_start: u32 = 0;

    for ct in classified {
        let (line, col) = offset_to_line_col(&line_starts, ct.span.start);
        let length = ct.span.end - ct.span.start;

        let delta_line = line - prev_line;
        let delta_start = if delta_line == 0 {
            col - prev_start
        } else {
            col
        };

        result.push(SemanticToken {
            delta_line,
            delta_start,
            length,
            token_type: ct.token_type,
            token_modifiers_bitset: 0,
        });

        prev_line = line;
        prev_start = col;
    }

    result
}

// ============================================================================
// Diagnostic conversion
// ============================================================================

/// Convert a fwgsl diagnostic to an LSP diagnostic.
fn fwgsl_diag_to_lsp(diag: &FwgslDiag, source: &str) -> Option<tower_lsp::lsp_types::Diagnostic> {
    let severity = match diag.severity {
        FwgslSeverity::Error => DiagnosticSeverity::ERROR,
        FwgslSeverity::Warning => DiagnosticSeverity::WARNING,
        FwgslSeverity::Info => DiagnosticSeverity::INFORMATION,
        FwgslSeverity::Hint => DiagnosticSeverity::HINT,
    };

    // Use the first label's span for the range, or fall back to start of file.
    let range = if let Some(label) = diag.labels.first() {
        span_to_range(source, label.span)
    } else {
        Range::new(Position::new(0, 0), Position::new(0, 0))
    };

    Some(tower_lsp::lsp_types::Diagnostic {
        range,
        severity: Some(severity),
        code: diag
            .code
            .as_ref()
            .map(|c| NumberOrString::String(c.clone())),
        source: Some("fwgsl".into()),
        message: diag.message.clone(),
        related_information: None,
        tags: None,
        code_description: None,
        data: None,
    })
}

// ============================================================================
// Position / offset utilities
// ============================================================================

/// Compute the byte offsets where each line starts.
pub fn compute_line_starts(source: &str) -> Vec<u32> {
    let mut starts = vec![0u32];
    for (i, b) in source.bytes().enumerate() {
        if b == b'\n' {
            starts.push((i + 1) as u32);
        }
    }
    starts
}

/// Convert a byte offset to (line, column), both 0-based.
pub fn offset_to_line_col(line_starts: &[u32], offset: u32) -> (u32, u32) {
    let line = match line_starts.binary_search(&offset) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };
    let col = offset - line_starts[line];
    (line as u32, col)
}

/// Convert an LSP Position (0-based line/character) to a byte offset.
pub fn position_to_offset(source: &str, pos: Position) -> Option<usize> {
    let line_starts = compute_line_starts(source);
    let line = pos.line as usize;
    if line >= line_starts.len() {
        return None;
    }
    let line_start = line_starts[line] as usize;
    let offset = line_start + pos.character as usize;
    if offset <= source.len() {
        Some(offset)
    } else {
        None
    }
}

/// Convert a fwgsl Span to an LSP Range.
pub fn span_to_range(source: &str, span: Span) -> Range {
    let line_starts = compute_line_starts(source);
    let (start_line, start_col) = offset_to_line_col(&line_starts, span.start);
    let (end_line, end_col) = offset_to_line_col(&line_starts, span.end);
    Range::new(
        Position::new(start_line, start_col),
        Position::new(end_line, end_col),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Completion tests ---------------------------------------------------

    #[test]
    fn test_keyword_completions_present() {
        let items = build_completions("", Position::new(0, 0));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();

        // All keywords should be present
        for kw in [
            "match", "let", "in", "if", "then", "else", "where", "data", "type", "class",
            "instance", "module", "import", "do", "forall", "case", "of", "infixl", "infixr",
            "infix", "deriving",
        ] {
            assert!(labels.contains(&kw), "Missing keyword completion: {}", kw);
        }
    }

    #[test]
    fn test_type_completions_present() {
        let items = build_completions("", Position::new(0, 0));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();

        for ty in ["I32", "U32", "F32", "Bool", "Vec", "Mat", "Array"] {
            assert!(labels.contains(&ty), "Missing type completion: {}", ty);
        }
    }

    #[test]
    fn test_builtin_function_completions_present() {
        let items = build_completions("", Position::new(0, 0));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();

        for f in [
            "map", "filter", "foldl", "foldr", "id", "const", "flip", "pure", "bind", "fmap",
        ] {
            assert!(
                labels.contains(&f),
                "Missing builtin function completion: {}",
                f
            );
        }
    }

    #[test]
    fn test_completions_include_document_identifiers() {
        let source = "add x y = x + y";
        let items = build_completions(source, Position::new(0, 0));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();

        assert!(
            labels.contains(&"add"),
            "Missing identifier 'add' from source"
        );
        assert!(labels.contains(&"x"), "Missing identifier 'x' from source");
        assert!(labels.contains(&"y"), "Missing identifier 'y' from source");
    }

    #[test]
    fn test_completions_include_upper_ident_from_document() {
        let source = "data Color = Red | Green | Blue";
        let items = build_completions(source, Position::new(0, 0));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();

        assert!(labels.contains(&"Color"), "Missing 'Color' from source");
        assert!(labels.contains(&"Red"), "Missing 'Red' from source");
        assert!(labels.contains(&"Green"), "Missing 'Green' from source");
        assert!(labels.contains(&"Blue"), "Missing 'Blue' from source");
    }

    #[test]
    fn test_completion_item_kinds() {
        let items = build_completions("", Position::new(0, 0));

        let let_item = items.iter().find(|i| i.label == "let").unwrap();
        assert_eq!(let_item.kind, Some(CompletionItemKind::KEYWORD));

        let i32_item = items.iter().find(|i| i.label == "I32").unwrap();
        assert_eq!(i32_item.kind, Some(CompletionItemKind::TYPE_PARAMETER));

        let map_item = items.iter().find(|i| i.label == "map").unwrap();
        assert_eq!(map_item.kind, Some(CompletionItemKind::FUNCTION));
    }

    #[test]
    fn test_completions_no_duplicates_for_keywords_in_source() {
        // When source contains keywords, they should not appear as
        // variable completions (the lexer emits them as keywords, not idents).
        let source = "let x = 1 in x";
        let items = build_completions(source, Position::new(0, 0));

        // "let" should appear exactly once (as keyword completion, not from source scan)
        let let_items: Vec<_> = items.iter().filter(|i| i.label == "let").collect();
        assert_eq!(let_items.len(), 1);
        assert_eq!(let_items[0].kind, Some(CompletionItemKind::KEYWORD));
    }

    // -- Semantic token classification tests --------------------------------

    #[test]
    fn test_classify_keywords() {
        let source = "let x = 1 in x";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let kw_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_KEYWORD)
            .collect();
        assert_eq!(kw_tokens.len(), 2, "Expected 2 keywords (let, in)");

        let let_text = kw_tokens[0].span.source_text(source);
        assert_eq!(let_text, "let");

        let in_text = kw_tokens[1].span.source_text(source);
        assert_eq!(in_text, "in");
    }

    #[test]
    fn test_classify_all_keywords() {
        let source = "module where import data type class instance let in case of match if then else do forall infixl infixr infix deriving";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let kw_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_KEYWORD)
            .collect();
        assert_eq!(kw_tokens.len(), 21, "Expected 21 keywords");
    }

    #[test]
    fn test_classify_numbers() {
        let source = "42 3.14";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let num_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_NUMBER)
            .collect();
        assert_eq!(num_tokens.len(), 2);
        assert_eq!(num_tokens[0].span.source_text(source), "42");
        assert_eq!(num_tokens[1].span.source_text(source), "3.14");
    }

    #[test]
    fn test_classify_strings() {
        let source = r#""hello" 'a'"#;
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let str_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_STRING)
            .collect();
        assert_eq!(str_tokens.len(), 2);
    }

    #[test]
    fn test_classify_comments() {
        let source = "-- a comment\nx";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let comment_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_COMMENT)
            .collect();
        assert_eq!(comment_tokens.len(), 1);
        assert_eq!(comment_tokens[0].span.source_text(source), "-- a comment");
    }

    #[test]
    fn test_classify_block_comment() {
        let source = "{- block -}";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let comment_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_COMMENT)
            .collect();
        assert_eq!(comment_tokens.len(), 1);
    }

    #[test]
    fn test_classify_upper_ident_as_type() {
        let source = "Maybe Int";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let type_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_TYPE)
            .collect();
        assert_eq!(type_tokens.len(), 2);
        assert_eq!(type_tokens[0].span.source_text(source), "Maybe");
        assert_eq!(type_tokens[1].span.source_text(source), "Int");
    }

    #[test]
    fn test_classify_ident_as_variable() {
        let source = "foo bar";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let var_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_VARIABLE)
            .collect();
        assert_eq!(var_tokens.len(), 2);
        assert_eq!(var_tokens[0].span.source_text(source), "foo");
        assert_eq!(var_tokens[1].span.source_text(source), "bar");
    }

    #[test]
    fn test_classify_operators() {
        let source = "1 + 2 * 3";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let op_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_OPERATOR)
            .collect();
        assert_eq!(op_tokens.len(), 2);
        assert_eq!(op_tokens[0].span.source_text(source), "+");
        assert_eq!(op_tokens[1].span.source_text(source), "*");
    }

    #[test]
    fn test_classify_decorator() {
        let source = "@vertex main";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let dec_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_DECORATOR)
            .collect();
        assert_eq!(
            dec_tokens.len(),
            2,
            "Expected @ and identifier as decorators"
        );
        assert_eq!(dec_tokens[0].span.source_text(source), "@");
        assert_eq!(dec_tokens[1].span.source_text(source), "vertex");
    }

    #[test]
    fn test_classify_decorator_with_upper_ident() {
        let source = "@Builtin func";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let dec_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_DECORATOR)
            .collect();
        assert_eq!(dec_tokens.len(), 2);
        assert_eq!(dec_tokens[0].span.source_text(source), "@");
        assert_eq!(dec_tokens[1].span.source_text(source), "Builtin");

        // "func" should be a variable, not a decorator
        let var_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_VARIABLE)
            .collect();
        assert_eq!(var_tokens.len(), 1);
        assert_eq!(var_tokens[0].span.source_text(source), "func");
    }

    #[test]
    fn test_classify_arrow_as_operator() {
        let source = "a -> b";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let op_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_OPERATOR)
            .collect();
        assert_eq!(op_tokens.len(), 1);
        assert_eq!(op_tokens[0].span.source_text(source), "->");
    }

    #[test]
    fn test_classify_fat_arrow_as_operator() {
        let source = "a => b";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let op_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_OPERATOR)
            .collect();
        assert_eq!(op_tokens.len(), 1);
        assert_eq!(op_tokens[0].span.source_text(source), "=>");
    }

    #[test]
    fn test_classify_pipe_as_operator() {
        let source = "| x -> y";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        let op_tokens: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_OPERATOR)
            .collect();
        // pipe and arrow
        assert_eq!(op_tokens.len(), 2);
    }

    #[test]
    fn test_classify_mixed_source() {
        let source = "let add :: I32 -> I32 -> I32\nadd x y = x + y";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);

        // Keywords: let
        let kws: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_KEYWORD)
            .collect();
        assert_eq!(kws.len(), 1);

        // Types: I32 (appears 3 times)
        let types: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_TYPE)
            .collect();
        assert_eq!(types.len(), 3);

        // Variables: add (2x), x (2x), y (2x) = 6
        let vars: Vec<_> = classified
            .iter()
            .filter(|c| c.token_type == TOK_VARIABLE)
            .collect();
        assert_eq!(vars.len(), 6);
    }

    #[test]
    fn test_classify_empty_source() {
        let source = "";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);
        assert!(classified.is_empty());
    }

    #[test]
    fn test_classify_only_whitespace() {
        let source = "   \n  \n";
        let tokens = lex(source);
        let classified = classify_tokens(&tokens, source);
        assert!(classified.is_empty());
    }

    // -- Semantic token encoding tests --------------------------------------

    #[test]
    fn test_semantic_tokens_single_line() {
        let source = "let x = 42";
        let tokens = build_semantic_tokens(source);
        // Should have: "let"(keyword), "x"(variable), "="(operator), "42"(number)
        assert_eq!(tokens.len(), 4);

        // First token: "let" at (0, 0)
        assert_eq!(tokens[0].delta_line, 0);
        assert_eq!(tokens[0].delta_start, 0);
        assert_eq!(tokens[0].length, 3);
        assert_eq!(tokens[0].token_type, TOK_KEYWORD);

        // Second token: "x" at (0, 4) => delta (0, 4)
        assert_eq!(tokens[1].delta_line, 0);
        assert_eq!(tokens[1].delta_start, 4);
        assert_eq!(tokens[1].token_type, TOK_VARIABLE);

        // Third token: "=" at (0, 6) => delta (0, 2)
        assert_eq!(tokens[2].delta_line, 0);
        assert_eq!(tokens[2].delta_start, 2);
        assert_eq!(tokens[2].token_type, TOK_OPERATOR);

        // Fourth token: "42" at (0, 8) => delta (0, 2)
        assert_eq!(tokens[3].delta_line, 0);
        assert_eq!(tokens[3].delta_start, 2);
        assert_eq!(tokens[3].token_type, TOK_NUMBER);
    }

    #[test]
    fn test_semantic_tokens_multi_line() {
        let source = "let x = 1\nlet y = 2";
        let tokens = build_semantic_tokens(source);
        // Line 1: let(kw) x(var) =(op) 1(num) => 4 tokens
        // Line 2: let(kw) y(var) =(op) 2(num) => 4 tokens
        assert_eq!(tokens.len(), 8);

        // First token of line 2: "let" at (1, 0), delta_line = 1
        assert_eq!(tokens[4].delta_line, 1);
        assert_eq!(tokens[4].delta_start, 0);
    }

    #[test]
    fn test_semantic_tokens_empty() {
        let tokens = build_semantic_tokens("");
        assert!(tokens.is_empty());
    }

    // -- Position/offset utility tests --------------------------------------

    #[test]
    fn test_compute_line_starts() {
        let source = "hello\nworld\nfoo";
        let starts = compute_line_starts(source);
        assert_eq!(starts, vec![0, 6, 12]);
    }

    #[test]
    fn test_compute_line_starts_empty() {
        let starts = compute_line_starts("");
        assert_eq!(starts, vec![0]);
    }

    #[test]
    fn test_compute_line_starts_single_line() {
        let starts = compute_line_starts("hello");
        assert_eq!(starts, vec![0]);
    }

    #[test]
    fn test_offset_to_line_col() {
        let starts = vec![0, 6, 12];
        assert_eq!(offset_to_line_col(&starts, 0), (0, 0));
        assert_eq!(offset_to_line_col(&starts, 3), (0, 3));
        assert_eq!(offset_to_line_col(&starts, 6), (1, 0));
        assert_eq!(offset_to_line_col(&starts, 8), (1, 2));
        assert_eq!(offset_to_line_col(&starts, 12), (2, 0));
        assert_eq!(offset_to_line_col(&starts, 14), (2, 2));
    }

    #[test]
    fn test_position_to_offset() {
        let source = "hello\nworld\nfoo";
        assert_eq!(position_to_offset(source, Position::new(0, 0)), Some(0));
        assert_eq!(position_to_offset(source, Position::new(0, 3)), Some(3));
        assert_eq!(position_to_offset(source, Position::new(1, 0)), Some(6));
        assert_eq!(position_to_offset(source, Position::new(1, 2)), Some(8));
        assert_eq!(position_to_offset(source, Position::new(2, 0)), Some(12));
    }

    #[test]
    fn test_position_to_offset_out_of_range() {
        let source = "hello\nworld";
        // Line beyond end
        assert_eq!(position_to_offset(source, Position::new(5, 0)), None);
    }

    #[test]
    fn test_span_to_range() {
        let source = "hello\nworld";
        let span = Span::new(6, 11);
        let range = span_to_range(source, span);
        assert_eq!(range.start.line, 1);
        assert_eq!(range.start.character, 0);
        assert_eq!(range.end.line, 1);
        assert_eq!(range.end.character, 5);
    }

    #[test]
    fn test_span_to_range_cross_line() {
        let source = "hello\nworld\nfoo";
        let span = Span::new(3, 14);
        let range = span_to_range(source, span);
        assert_eq!(range.start.line, 0);
        assert_eq!(range.start.character, 3);
        assert_eq!(range.end.line, 2);
        assert_eq!(range.end.character, 2);
    }

    // -- Hover tests --------------------------------------------------------

    #[test]
    fn test_hover_on_keyword() {
        let source = "let x = 42 in x";
        let hover = build_hover(source, Position::new(0, 0));
        assert!(hover.is_some());
        let hover = hover.unwrap();
        match hover.contents {
            HoverContents::Markup(m) => {
                assert!(m.value.contains("let"));
                assert!(m.value.contains("Introduces local bindings"));
            }
            _ => panic!("Expected Markup hover content"),
        }
    }

    #[test]
    fn test_hover_on_identifier_with_type() {
        let source = "add x y = x + y";
        // Hover over "add" at position (0, 0)
        let hover = build_hover(source, Position::new(0, 0));
        assert!(hover.is_some());
        let hover = hover.unwrap();
        match hover.contents {
            HoverContents::Markup(m) => {
                assert!(
                    m.value.contains("add"),
                    "Hover should contain the identifier name: {}",
                    m.value
                );
            }
            _ => panic!("Expected Markup hover content"),
        }
    }

    #[test]
    fn test_hover_on_whitespace_returns_none() {
        let source = "let  x = 42";
        // Position at column 4 which is whitespace between "let" and "x"
        let hover = build_hover(source, Position::new(0, 4));
        assert!(hover.is_none());
    }

    #[test]
    fn test_hover_on_number_returns_none() {
        let source = "let x = 42";
        // Position at column 8 which is the number 42
        let hover = build_hover(source, Position::new(0, 8));
        assert!(hover.is_none());
    }

    // -- Go-to-definition tests ---------------------------------------------

    #[test]
    fn test_goto_definition_finds_function() {
        let source = "add x y = x + y\nresult = add 1 2";
        let uri = Url::parse("file:///test.fwgsl").unwrap();
        // Position on "add" in "result = add 1 2" (line 1, col 9)
        let result = build_goto_definition(&uri, source, Position::new(1, 9));
        assert!(result.is_some());
        match result.unwrap() {
            GotoDefinitionResponse::Scalar(loc) => {
                assert_eq!(loc.range.start.line, 0);
                assert_eq!(loc.range.start.character, 0);
            }
            _ => panic!("Expected Scalar location"),
        }
    }

    #[test]
    fn test_goto_definition_on_non_identifier_returns_none() {
        let source = "let x = 42";
        let uri = Url::parse("file:///test.fwgsl").unwrap();
        // Position on "=" sign
        let result = build_goto_definition(&uri, source, Position::new(0, 6));
        assert!(result.is_none());
    }

    // -- Diagnostics conversion tests ---------------------------------------

    #[test]
    fn test_fwgsl_diag_to_lsp_error() {
        let diag = FwgslDiag::error("type mismatch")
            .with_label(fwgsl_diagnostics::Label::primary(Span::new(0, 5), "here"));
        let source = "hello world";
        let lsp = fwgsl_diag_to_lsp(&diag, source).unwrap();
        assert_eq!(lsp.severity, Some(DiagnosticSeverity::ERROR));
        assert_eq!(lsp.message, "type mismatch");
        assert_eq!(lsp.range.start.line, 0);
        assert_eq!(lsp.range.start.character, 0);
        assert_eq!(lsp.range.end.character, 5);
    }

    #[test]
    fn test_fwgsl_diag_to_lsp_warning() {
        let diag = FwgslDiag::warning("unused variable");
        let source = "hello";
        let lsp = fwgsl_diag_to_lsp(&diag, source).unwrap();
        assert_eq!(lsp.severity, Some(DiagnosticSeverity::WARNING));
    }

    #[test]
    fn test_fwgsl_diag_to_lsp_with_code() {
        let diag = FwgslDiag::error("oops").with_code("E001");
        let source = "";
        let lsp = fwgsl_diag_to_lsp(&diag, source).unwrap();
        assert_eq!(lsp.code, Some(NumberOrString::String("E001".into())));
        assert_eq!(lsp.source, Some("fwgsl".into()));
    }

    #[test]
    fn test_fwgsl_diag_to_lsp_no_labels() {
        let diag = FwgslDiag::error("generic error");
        let source = "some source";
        let lsp = fwgsl_diag_to_lsp(&diag, source).unwrap();
        // Falls back to (0,0)-(0,0) range
        assert_eq!(lsp.range.start.line, 0);
        assert_eq!(lsp.range.start.character, 0);
    }

    // -- Semantic tokens legend tests ---------------------------------------

    #[test]
    fn test_semantic_tokens_legend() {
        let legend = semantic_tokens_legend();
        assert_eq!(legend.token_types.len(), TOKEN_TYPES.len());
        assert!(legend.token_modifiers.is_empty());
        assert_eq!(legend.token_types[0], SemanticTokenType::KEYWORD);
        assert_eq!(legend.token_types[10], SemanticTokenType::DECORATOR);
    }
}
