//! Language Server Protocol implementation for shadml.
//!
//! Provides IDE features (diagnostics, completions, hover, semantic tokens,
//! go-to-definition) by integrating with the shadml compiler pipeline
//! (`shadml_parser`, `shadml_semantic`, `shadml_typechecker`).

use std::collections::HashSet;

use dashmap::DashMap;
use shadml_ide::{
    all_completion_specs, build_completions as ide_build_completions,
    build_goto_definition as ide_build_goto_definition, build_hover as ide_build_hover,
    build_references as ide_build_references, completion_item_from_spec, lookup_completion_spec,
    spec_matches_context, CompletionContext, CompletionSpec,
};
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use shadml_diagnostics::{Diagnostic as ShadmlDiag, Severity as ShadmlSeverity};
use shadml_parser::lexer::Token;
use shadml_parser::parser::Program;
use shadml_parser::{lex, Parser};
use shadml_semantic::SemanticAnalyzer;
use shadml_span::Span;
use shadml_syntax::SyntaxKind;
use shadml_typechecker::{InferEngine, Scheme};

/// Prepend prelude declarations to a parsed program.
fn with_prelude(program: &mut Program) {
    let prelude = shadml_parser::prelude_program();
    let mut combined = prelude.decls.clone();
    combined.append(&mut program.decls);
    program.decls = combined;
}

/// Check if the program has import declarations (needs multi-file resolution).
fn has_imports(program: &Program) -> bool {
    has_imports_in(&program.decls)
}

fn has_imports_in(decls: &[shadml_parser::parser::Decl]) -> bool {
    use shadml_parser::parser::Decl;
    decls.iter().any(|d| match d {
        Decl::ImportDecl { .. } => true,
        Decl::CfgDecl {
            then_decls,
            else_decls,
            ..
        } => has_imports_in(then_decls) || has_imports_in(else_decls),
        _ => false,
    })
}

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

/// The shadml language server backend.
pub struct ShadmlBackend {
    /// The LSP client used to send notifications (e.g. diagnostics).
    client: Client,
    /// In-memory document store: URI -> document text.
    documents: DashMap<Url, String>,
    /// Cached module files: root URI -> list of (file_path, source_text) for imported modules.
    module_files: DashMap<Url, Vec<(std::path::PathBuf, String)>>,
}

impl ShadmlBackend {
    /// Create a new backend with the given LSP client.
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
            module_files: DashMap::new(),
        }
    }

    // -- Diagnostics ---------------------------------------------------------

    /// Lex, parse, and semantically analyze the document, then publish
    /// diagnostics back to the client.
    async fn run_diagnostics(&self, uri: Url, text: &str) {
        let mut all_diagnostics: Vec<tower_lsp::lsp_types::Diagnostic> = Vec::new();

        // Phase 1: Parse
        let mut parser = Parser::new(text);
        let root_program = parser.parse_program();

        // Collect parser diagnostics
        for diag in parser.diagnostics().iter() {
            if let Some(lsp_diag) = shadml_diag_to_lsp(diag, text, Some(&uri)) {
                all_diagnostics.push(lsp_diag);
            }
        }

        // Phase 2: Module resolution (if the file has imports)
        let mut program = if has_imports(&root_program) {
            if let Some(file_path) = uri.to_file_path().ok() {
                let source_root = file_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .to_path_buf();
                let reader = shadml_parser::FsReader;
                match shadml_parser::resolve_modules(
                    &file_path,
                    root_program,
                    &[source_root],
                    &reader,
                ) {
                    Ok(graph) => {
                        // Cache imported module file paths and sources for
                        // cross-file goto-definition.
                        let mut imported = Vec::new();
                        for m in &graph.modules {
                            if m.path != file_path {
                                if let Ok(src) = std::fs::read_to_string(&m.path) {
                                    imported.push((m.path.clone(), src));
                                }
                            }
                        }
                        self.module_files.insert(uri.clone(), imported);

                        shadml_parser::merge_modules(&graph)
                    }
                    Err(errors) => {
                        for e in &errors {
                            all_diagnostics.push(tower_lsp::lsp_types::Diagnostic {
                                range: tower_lsp::lsp_types::Range::default(),
                                severity: Some(DiagnosticSeverity::ERROR),
                                message: e.to_string(),
                                ..Default::default()
                            });
                        }
                        // Fall back to the root program without resolved imports
                        parser = Parser::new(text);
                        parser.parse_program()
                    }
                }
            } else {
                root_program
            }
        } else {
            self.module_files.remove(&uri);
            root_program
        };

        // Prepend prelude
        with_prelude(&mut program);

        // Phase 3: Semantic analysis
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program);

        for diag in analyzer.diagnostics().iter() {
            if let Some(lsp_diag) = shadml_diag_to_lsp(diag, text, Some(&uri)) {
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
impl LanguageServer for ShadmlBackend {
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
                references_provider: Some(OneOf::Left(true)),
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
                document_formatting_provider: Some(OneOf::Left(true)),
                document_range_formatting_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "shadml-lsp".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "shadml language server initialized")
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

        let items = ide_build_completions(&text, pos);
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

        Ok(ide_build_hover(&text, pos))
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

        // Extract the identifier at cursor position (needed for cross-file lookup).
        let name = ident_at_position(&text, pos);

        // Try local goto-definition first.
        if let Some(result) = ide_build_goto_definition(uri, &text, pos) {
            // Verify the result actually points to a definition of the same name
            // in the current file. The IDE index can produce spurious results for
            // names imported from other modules.
            let is_valid = match &result {
                GotoDefinitionResponse::Scalar(loc) => name
                    .as_deref()
                    .map_or(true, |n| location_text_matches(&text, &loc.range, n)),
                GotoDefinitionResponse::Array(locs) => name.as_deref().map_or(true, |n| {
                    locs.iter()
                        .any(|loc| location_text_matches(&text, &loc.range, n))
                }),
                _ => true,
            };
            if is_valid {
                return Ok(Some(result));
            }
        }

        // If the cursor is on a record field (after `.`), search for field
        // definitions in data declarations — in the current file first, then
        // in imported modules.
        if let Some(ref name) = name {
            if is_field_access_at(&text, pos) {
                if let Some(range) = find_record_field_in_source(&text, name) {
                    return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                        uri: uri.clone(),
                        range,
                    })));
                }
                if let Some(imported) = self.module_files.get(uri) {
                    for (path, src) in imported.iter() {
                        if let Some(range) = find_record_field_in_source(src, name) {
                            let module_uri = match Url::from_file_path(path) {
                                Ok(u) => u,
                                Err(_) => continue,
                            };
                            return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                                uri: module_uri,
                                range,
                            })));
                        }
                    }
                }
            }
        }

        // If not found locally, try imported module files for top-level definitions.
        if let Some(ref name) = name {
            if let Some(imported) = self.module_files.get(uri) {
                for (path, src) in imported.iter() {
                    if let Some(range) = find_definition_in_source(src, name) {
                        let module_uri = match Url::from_file_path(path) {
                            Ok(u) => u,
                            Err(_) => continue,
                        };
                        return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                            uri: module_uri,
                            range,
                        })));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        Ok(ide_build_references(
            uri,
            &text,
            pos,
            params.context.include_declaration,
        ))
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

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = &params.text_document.uri;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        let formatted = shadml_formatter::format_default(&text);

        if formatted == text {
            return Ok(None);
        }

        let line_count = text.lines().count() as u32;
        let last_line_len = text.lines().last().map_or(0, |l| l.len()) as u32;

        Ok(Some(vec![TextEdit {
            range: Range::new(
                Position::new(0, 0),
                Position::new(line_count, last_line_len),
            ),
            new_text: formatted,
        }]))
    }

    async fn range_formatting(
        &self,
        params: DocumentRangeFormattingParams,
    ) -> Result<Option<Vec<TextEdit>>> {
        let uri = &params.text_document.uri;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        // Extract the selected range text
        let start_offset = position_to_offset(&text, params.range.start);
        let end_offset = position_to_offset(&text, params.range.end);

        let (Some(start), Some(end)) = (start_offset, end_offset) else {
            return Ok(None);
        };
        let end = end.min(text.len());

        // Expand to full lines to avoid partial-line formatting artefacts
        let line_start = text[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = text[end..].find('\n').map_or(text.len(), |i| end + i);

        let slice = &text[line_start..line_end];
        let formatted = shadml_formatter::format_default(slice);

        if formatted.trim_end() == slice.trim_end() {
            return Ok(None);
        }

        let start_pos = offset_to_position(&text, line_start);
        let end_pos = offset_to_position(&text, line_end);

        Ok(Some(vec![TextEdit {
            range: Range::new(start_pos, end_pos),
            new_text: formatted,
        }]))
    }

    // -- Document symbols ---------------------------------------------------

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;

        let text = match self.documents.get(uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        let symbols = build_document_symbols(&text);
        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }
}

// ============================================================================
// Completions
// ============================================================================

/// Build a list of completion items based on cursor context.
pub fn build_completions(source: &str, pos: Position) -> Vec<CompletionItem> {
    let prefix = completion_prefix(source, pos);
    let context = completion_context(source, pos, &prefix);
    let mut seen = HashSet::new();
    let mut items = Vec::new();

    for spec in all_completion_specs() {
        if !spec_matches_context(spec, context) {
            continue;
        }
        if !matches_prefix(spec.label, &prefix) {
            continue;
        }

        seen.insert(spec.label.to_owned());
        items.push(completion_item_from_spec(spec));
    }

    let mut parser = Parser::new(source);
    let mut program = parser.parse_program();
    with_prelude(&mut program);
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze(&program);
    let mut document_seen = HashSet::new();

    for tok in lex(source) {
        let label = tok.text(source);
        if !matches!(tok.kind, SyntaxKind::Ident | SyntaxKind::UpperIdent) {
            continue;
        }
        if !matches_prefix(label, &prefix) {
            continue;
        }
        if context == CompletionContext::Attribute {
            continue;
        }
        if context == CompletionContext::Type && tok.kind == SyntaxKind::Ident {
            continue;
        }
        if !document_seen.insert(label.to_owned()) {
            continue;
        }

        let kind = if analyzer.constructors.contains_key(label) {
            CompletionItemKind::CONSTRUCTOR
        } else if analyzer.data_types.contains_key(label) || tok.kind == SyntaxKind::UpperIdent {
            CompletionItemKind::TYPE_PARAMETER
        } else {
            CompletionItemKind::VARIABLE
        };
        let detail = document_symbol_detail(&analyzer, label, tok.kind);
        let documentation = document_symbol_documentation(&analyzer, label, tok.kind);

        items.retain(|item| item.label != label);
        items.push(CompletionItem {
            label: label.to_owned(),
            kind: Some(kind),
            detail: Some(detail),
            documentation: Some(Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: documentation,
            })),
            sort_text: Some(format!("05-{}", label)),
            filter_text: Some(label.to_owned()),
            ..Default::default()
        });
        seen.insert(label.to_owned());
    }

    items.sort_by(|left, right| {
        left.sort_text
            .as_deref()
            .cmp(&right.sort_text.as_deref())
            .then_with(|| left.label.cmp(&right.label))
    });
    items
}

fn completion_prefix(source: &str, pos: Position) -> String {
    let offset = position_to_offset(source, pos).unwrap_or(source.len());
    let mut start = offset;

    while start > 0 {
        let Some(ch) = source[..start].chars().next_back() else {
            break;
        };
        if !is_completion_word_char(ch) {
            break;
        }
        start -= ch.len_utf8();
    }

    source[start..offset].to_owned()
}

fn completion_context(source: &str, pos: Position, prefix: &str) -> CompletionContext {
    let offset = position_to_offset(source, pos).unwrap_or(source.len());
    let before_cursor = &source[..offset];
    let before_prefix = &before_cursor[..before_cursor.len().saturating_sub(prefix.len())];

    if before_prefix
        .chars()
        .rev()
        .find(|ch| !ch.is_whitespace())
        .is_some_and(|ch| ch == '@')
    {
        return CompletionContext::Attribute;
    }

    if prefix.chars().next().is_some_and(char::is_uppercase) {
        return CompletionContext::Type;
    }

    let line_start = before_cursor.rfind('\n').map_or(0, |index| index + 1);
    let line = &before_cursor[line_start..];
    let last_colon = line.rfind(':');
    let last_equals = line.rfind('=');

    if last_colon.is_some() && last_equals.is_none_or(|equals| last_colon > Some(equals)) {
        return CompletionContext::Type;
    }

    CompletionContext::Value
}

fn is_completion_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '\'' | '$')
}

fn matches_prefix(candidate: &str, prefix: &str) -> bool {
    prefix.is_empty() || candidate.starts_with(prefix)
}

fn document_symbol_detail(analyzer: &SemanticAnalyzer, label: &str, kind: SyntaxKind) -> String {
    if let Some(scheme) = analyzer.env.lookup(label) {
        return format!(
            "{} : {}",
            symbol_kind_label(analyzer, label, kind),
            format_scheme(&analyzer.engine, scheme)
        );
    }

    if analyzer.data_types.contains_key(label) {
        return "data type".to_owned();
    }

    "identifier".to_owned()
}

fn document_symbol_documentation(
    analyzer: &SemanticAnalyzer,
    label: &str,
    kind: SyntaxKind,
) -> String {
    if let Some(scheme) = analyzer.env.lookup(label) {
        return format!(
            "```shadml\n{} : {}\n```\n\n{}",
            label,
            format_scheme(&analyzer.engine, scheme),
            symbol_description(analyzer, label, kind),
        );
    }

    format!("**{}**", symbol_description(analyzer, label, kind))
}

fn symbol_kind_label(analyzer: &SemanticAnalyzer, label: &str, kind: SyntaxKind) -> &'static str {
    if analyzer.constructors.contains_key(label) {
        "constructor"
    } else if analyzer.data_types.contains_key(label) || kind == SyntaxKind::UpperIdent {
        "type"
    } else {
        "binding"
    }
}

fn symbol_description(analyzer: &SemanticAnalyzer, label: &str, kind: SyntaxKind) -> String {
    if let Some(constructor) = analyzer.constructors.get(label) {
        return format!("Constructor for `{}`.", constructor.type_name);
    }
    if analyzer.data_types.contains_key(label) || kind == SyntaxKind::UpperIdent {
        return "User-defined type or constructor.".to_owned();
    }
    "Identifier from the current document.".to_owned()
}

fn format_scheme(engine: &InferEngine, scheme: &Scheme) -> String {
    let ty = engine.finalize(&scheme.ty);
    if scheme.vars.is_empty() {
        format!("{}", ty)
    } else {
        let vars = scheme
            .vars
            .iter()
            .map(|var| format!("t{}", var))
            .collect::<Vec<_>>()
            .join(" ");
        format!("forall {}. {}", vars, ty)
    }
}

// ============================================================================
// Hover
// ============================================================================

/// Build hover information for the identifier at the given cursor position.
pub fn build_hover(source: &str, pos: Position) -> Option<Hover> {
    let offset = position_to_offset(source, pos)? as u32;
    let tokens = lex(source);
    let (tok_index, tok) = tokens
        .iter()
        .enumerate()
        .find(|(_, token)| token.span.start <= offset && offset < token.span.end)?;
    let range = span_to_range(source, tok.span);

    match tok.kind {
        SyntaxKind::Ident | SyntaxKind::UpperIdent => {
            let name = tok.text(source);

            if previous_non_trivia_token(&tokens, tok_index)
                .is_some_and(|prev| prev.kind == SyntaxKind::At)
            {
                if let Some(spec) = lookup_completion_spec(name, CompletionContext::Attribute) {
                    return Some(spec_hover(spec, range));
                }
            }

            if let Some(spec) = lookup_non_attribute_spec(name) {
                return Some(spec_hover(spec, range));
            }

            let mut parser = Parser::new(source);
            let mut program = parser.parse_program();
            with_prelude(&mut program);
            let mut analyzer = SemanticAnalyzer::new();
            analyzer.analyze(&program);

            if analyzer.env.lookup(name).is_some() || analyzer.data_types.contains_key(name) {
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: document_symbol_documentation(&analyzer, name, tok.kind),
                    }),
                    range: Some(range),
                });
            }

            None
        }
        kind if kind.is_keyword() => {
            let keyword = tok.text(source);
            lookup_non_attribute_spec(keyword).map(|spec| spec_hover(spec, range))
        }
        SyntaxKind::At => Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: "Use `@` to introduce a WGSL attribute such as `@compute`, `@vertex`, or `@workgroup_size(...)`.".to_owned(),
            }),
            range: Some(range),
        }),
        _ => None,
    }
}

fn previous_non_trivia_token(tokens: &[Token], index: usize) -> Option<&Token> {
    tokens[..index]
        .iter()
        .rev()
        .find(|token| !token.kind.is_trivia())
}

fn lookup_non_attribute_spec(label: &str) -> Option<&'static CompletionSpec> {
    lookup_completion_spec(label, CompletionContext::Value)
        .or_else(|| lookup_completion_spec(label, CompletionContext::Type))
}

fn spec_hover(spec: &CompletionSpec, range: Range) -> Hover {
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!(
                "**`{}`**\n\n_{}_\n\n{}",
                spec.label, spec.detail, spec.documentation
            ),
        }),
        range: Some(range),
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
            // Check if preceded by `data`, `alias`, `const`, or `extern` keyword
            if i > 0 {
                let prev = non_trivia[i - 1].kind;
                if prev == SyntaxKind::KwData
                    || prev == SyntaxKind::KwAlias
                    || prev == SyntaxKind::KwConst
                    || prev == SyntaxKind::KwExtern
                {
                    return Some(tok.span);
                }
            }
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

/// Check whether the text at the start of a Range in the source matches the expected name.
/// This validates that a goto-definition result actually points to a definition of the name,
/// not a spurious span from mismatched prelude symbols.
fn location_text_matches(source: &str, range: &Range, name: &str) -> bool {
    let Some(start) = shadml_ide::position_to_offset(source, range.start) else {
        return false;
    };
    let end = start + name.len();
    end <= source.len() && &source[start..end] == name
}

/// Extract the identifier text at a given cursor position.
fn ident_at_position(source: &str, pos: Position) -> Option<String> {
    let offset = shadml_ide::position_to_offset(source, pos)? as u32;
    let tokens = lex(source);
    let tok = tokens.iter().find(|t| {
        (t.kind == SyntaxKind::Ident || t.kind == SyntaxKind::UpperIdent)
            && t.span.start <= offset
            && offset < t.span.end
    })?;
    Some(tok.text(source).to_string())
}

/// Find the definition of a name in a source string, returning its Range.
fn find_definition_in_source(source: &str, name: &str) -> Option<Range> {
    let tokens = lex(source);
    let span = find_definition_span(&tokens, source, name)?;
    Some(shadml_ide::span_to_range(source, span))
}

/// Check whether the cursor is on a field name (preceded by `.` in the token stream).
fn is_field_access_at(source: &str, pos: Position) -> bool {
    let Some(offset) = shadml_ide::position_to_offset(source, pos) else {
        return false;
    };
    let offset = offset as u32;
    let tokens = lex(source);
    // Find the token at cursor
    let tok_idx = tokens.iter().position(|t| {
        (t.kind == SyntaxKind::Ident || t.kind == SyntaxKind::UpperIdent)
            && t.span.start <= offset
            && offset < t.span.end
    });
    let Some(idx) = tok_idx else { return false };
    // Check if the previous non-trivia token is `.`
    tokens[..idx]
        .iter()
        .rev()
        .find(|t| !t.kind.is_trivia())
        .is_some_and(|t| t.kind == SyntaxKind::Dot)
}

/// Find a record field definition in source. Scans for `Ident(name) Colon`
/// patterns inside `{ }` blocks that belong to data declarations, returning the
/// span of the field name token.
fn find_record_field_in_source(source: &str, field_name: &str) -> Option<Range> {
    let tokens = lex(source);
    let non_trivia: Vec<&Token> = tokens
        .iter()
        .filter(|t| {
            !t.kind.is_trivia()
                && t.kind != SyntaxKind::LayoutBraceOpen
                && t.kind != SyntaxKind::LayoutSemicolon
                && t.kind != SyntaxKind::LayoutBraceClose
        })
        .collect();

    // Track whether we're inside a data declaration's record braces.
    // Look for: `data Name = ConName {` ... `fieldName : Type` ... `}`
    let mut in_data_decl = false;
    let mut brace_depth: i32 = 0;
    let mut data_brace_depth: i32 = 0; // brace depth when we entered data decl braces

    for (i, tok) in non_trivia.iter().enumerate() {
        match tok.kind {
            SyntaxKind::KwData => {
                in_data_decl = true;
            }
            SyntaxKind::LBrace if in_data_decl => {
                brace_depth += 1;
                if data_brace_depth == 0 {
                    data_brace_depth = brace_depth;
                }
            }
            SyntaxKind::LBrace => {
                brace_depth += 1;
            }
            SyntaxKind::RBrace => {
                if brace_depth == data_brace_depth {
                    data_brace_depth = 0;
                    in_data_decl = false;
                }
                brace_depth -= 1;
            }
            SyntaxKind::Ident if data_brace_depth > 0 && brace_depth == data_brace_depth => {
                // Inside a data decl record block — check if this ident matches
                // and is followed by `:`
                if tok.text(source) == field_name {
                    if i + 1 < non_trivia.len() && non_trivia[i + 1].kind == SyntaxKind::Colon {
                        return Some(shadml_ide::span_to_range(source, tok.span));
                    }
                }
            }
            _ => {
                // If we see a top-level keyword that isn't part of data decl, reset
                if brace_depth == 0
                    && in_data_decl
                    && tok.kind.is_keyword()
                    && tok.kind != SyntaxKind::KwData
                {
                    in_data_decl = false;
                    data_brace_depth = 0;
                }
            }
        }
    }
    None
}

// ============================================================================
// Document symbols
// ============================================================================

/// Convert a byte offset to an LSP Position.
fn offset_to_position(source: &str, offset: usize) -> Position {
    let line_starts = shadml_ide::compute_line_starts(source);
    let (line, col) = shadml_ide::offset_to_line_col(&line_starts, offset as u32);
    Position::new(line, col)
}

/// Build a nested list of document symbols from the parsed AST.
#[allow(deprecated)] // DocumentSymbol.deprecated field
fn build_document_symbols(source: &str) -> Vec<DocumentSymbol> {
    let mut parser = Parser::new(source);
    let program = parser.parse_program();

    // Also run semantic analysis so we can show type info in details
    let mut full_program = program.clone();
    with_prelude(&mut full_program);
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze(&full_program);

    let mut symbols = Vec::new();

    for decl in &program.decls {
        match decl {
            shadml_parser::parser::Decl::TypeSig { name, span, .. } => {
                // Type signatures are paired with FunDecls — skip standalone ones
                // if the FunDecl follows. We'll emit them when we see the FunDecl.
                // But if there's no matching FunDecl, emit as a standalone symbol.
                let has_fundecl = program.decls.iter().any(|d| {
                    matches!(
                        d,
                        shadml_parser::parser::Decl::FunDecl { name: n, .. }
                        | shadml_parser::parser::Decl::EntryPoint { name: n, .. }
                        if n == name
                    )
                });
                if !has_fundecl {
                    let detail = analyzer
                        .env
                        .lookup(name)
                        .map(|s| format_scheme(&analyzer.engine, s));
                    symbols.push(DocumentSymbol {
                        name: name.clone(),
                        detail,
                        kind: SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        range: span_to_range(source, *span),
                        selection_range: first_name_range(source, name, *span),
                        children: None,
                    });
                }
            }
            shadml_parser::parser::Decl::FunDecl { name, span, .. } => {
                let detail = analyzer
                    .env
                    .lookup(name)
                    .map(|s| format_scheme(&analyzer.engine, s));
                // Merge span with preceding type signature if present
                let full_span = program
                    .decls
                    .iter()
                    .find_map(|d| {
                        if let shadml_parser::parser::Decl::TypeSig {
                            name: n, span: ts, ..
                        } = d
                        {
                            if n == name {
                                Some(Span::new(ts.start, span.end))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or(*span);
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail,
                    kind: SymbolKind::FUNCTION,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, full_span),
                    selection_range: first_name_range(source, name, *span),
                    children: None,
                });
            }
            shadml_parser::parser::Decl::EntryPoint {
                name,
                span,
                attributes,
                ..
            } => {
                let stage = attributes.iter().find_map(|a| {
                    if matches!(a.name.as_str(), "compute" | "vertex" | "fragment") {
                        Some(format!("@{}", a.name))
                    } else {
                        None
                    }
                });
                let detail = stage.or_else(|| {
                    analyzer
                        .env
                        .lookup(name)
                        .map(|s| format_scheme(&analyzer.engine, s))
                });
                let full_span = program
                    .decls
                    .iter()
                    .find_map(|d| {
                        if let shadml_parser::parser::Decl::TypeSig {
                            name: n, span: ts, ..
                        } = d
                        {
                            if n == name {
                                Some(Span::new(ts.start, span.end))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or(*span);
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail,
                    kind: SymbolKind::FUNCTION,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, full_span),
                    selection_range: first_name_range(source, name, *span),
                    children: None,
                });
            }
            shadml_parser::parser::Decl::DataDecl {
                name,
                constructors,
                span,
                ..
            } => {
                let children: Vec<DocumentSymbol> = constructors
                    .iter()
                    .map(|con| {
                        let mut con_children = Vec::new();
                        if let shadml_parser::parser::ConFields::Record(fields) = &con.fields {
                            for f in fields {
                                if let Some(field_range) =
                                    find_field_name_range(source, &f.name, con.span)
                                {
                                    con_children.push(DocumentSymbol {
                                        name: f.name.clone(),
                                        detail: Some(format_type(&f.ty)),
                                        kind: SymbolKind::FIELD,
                                        tags: None,
                                        deprecated: None,
                                        range: field_range,
                                        selection_range: field_range,
                                        children: None,
                                    });
                                }
                            }
                        }
                        DocumentSymbol {
                            name: con.name.clone(),
                            detail: None,
                            kind: SymbolKind::CONSTRUCTOR,
                            tags: None,
                            deprecated: None,
                            range: span_to_range(source, con.span),
                            selection_range: first_name_range(source, &con.name, con.span),
                            children: if con_children.is_empty() {
                                None
                            } else {
                                Some(con_children)
                            },
                        }
                    })
                    .collect();
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail: Some("data type".to_owned()),
                    kind: SymbolKind::STRUCT,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: first_name_range(source, name, *span),
                    children: if children.is_empty() {
                        None
                    } else {
                        Some(children)
                    },
                });
            }
            shadml_parser::parser::Decl::TypeAlias { name, span, .. } => {
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail: Some("type alias".to_owned()),
                    kind: SymbolKind::TYPE_PARAMETER,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: first_name_range(source, name, *span),
                    children: None,
                });
            }
            shadml_parser::parser::Decl::ConstDecl { name, span, .. } => {
                let detail = analyzer
                    .env
                    .lookup(name)
                    .map(|s| format_scheme(&analyzer.engine, s));
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail,
                    kind: SymbolKind::CONSTANT,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: first_name_range(source, name, *span),
                    children: None,
                });
            }
            shadml_parser::parser::Decl::BindingDecl { name, span, .. } => {
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail: Some("binding".to_owned()),
                    kind: SymbolKind::VARIABLE,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: first_name_range(source, name, *span),
                    children: None,
                });
            }
            shadml_parser::parser::Decl::BitfieldDecl {
                name, fields, span, ..
            } => {
                let children: Vec<DocumentSymbol> = fields
                    .iter()
                    .map(|f| DocumentSymbol {
                        name: f.name.clone(),
                        detail: Some(match &f.kind {
                            shadml_parser::parser::BitfieldFieldKind::Bare(w) => format!("{} bits", w),
                            shadml_parser::parser::BitfieldFieldKind::Typed { ty, width } => format!("{} : {}", ty, width),
                            shadml_parser::parser::BitfieldFieldKind::Bool => "Bool".to_owned(),
                            shadml_parser::parser::BitfieldFieldKind::EnumInferred(ty) => ty.clone(),
                        }),
                        kind: SymbolKind::FIELD,
                        tags: None,
                        deprecated: None,
                        range: span_to_range(source, f.span),
                        selection_range: first_name_range(source, &f.name, f.span),
                        children: None,
                    })
                    .collect();
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail: Some("bitfield".to_owned()),
                    kind: SymbolKind::STRUCT,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: first_name_range(source, name, *span),
                    children: if children.is_empty() {
                        None
                    } else {
                        Some(children)
                    },
                });
            }
            shadml_parser::parser::Decl::TraitDecl {
                name,
                methods,
                span,
                ..
            } => {
                let children: Vec<DocumentSymbol> = methods
                    .iter()
                    .map(|m| DocumentSymbol {
                        name: m.name.clone(),
                        detail: Some(format_type(&m.ty)),
                        kind: SymbolKind::METHOD,
                        tags: None,
                        deprecated: None,
                        range: span_to_range(source, m.span),
                        selection_range: first_name_range(source, &m.name, m.span),
                        children: None,
                    })
                    .collect();
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail: Some("trait".to_owned()),
                    kind: SymbolKind::INTERFACE,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: first_name_range(source, name, *span),
                    children: if children.is_empty() {
                        None
                    } else {
                        Some(children)
                    },
                });
            }
            shadml_parser::parser::Decl::ImplDecl {
                trait_name,
                ty,
                methods,
                span,
                ..
            } => {
                let impl_name = match trait_name {
                    Some(t) => format!("impl {} {}", t, format_type(ty)),
                    None => format!("impl {}", format_type(ty)),
                };
                let children: Vec<DocumentSymbol> = methods
                    .iter()
                    .map(|m| DocumentSymbol {
                        name: m.name.clone(),
                        detail: None,
                        kind: SymbolKind::METHOD,
                        tags: None,
                        deprecated: None,
                        range: span_to_range(source, m.span),
                        selection_range: first_name_range(source, &m.name, m.span),
                        children: None,
                    })
                    .collect();
                symbols.push(DocumentSymbol {
                    name: impl_name,
                    detail: None,
                    kind: SymbolKind::CLASS,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: span_to_range(source, *span),
                    children: if children.is_empty() {
                        None
                    } else {
                        Some(children)
                    },
                });
            }
            shadml_parser::parser::Decl::ExternDecl { name, span, .. } => {
                let detail = analyzer
                    .env
                    .lookup(name)
                    .map(|s| format_scheme(&analyzer.engine, s));
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail,
                    kind: SymbolKind::FUNCTION,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: first_name_range(source, name, *span),
                    children: None,
                });
            }
            shadml_parser::parser::Decl::ImportDecl {
                module_path, span, ..
            } => {
                symbols.push(DocumentSymbol {
                    name: format!("import {}", module_path),
                    detail: None,
                    kind: SymbolKind::MODULE,
                    tags: None,
                    deprecated: None,
                    range: span_to_range(source, *span),
                    selection_range: span_to_range(source, *span),
                    children: None,
                });
            }
            _ => {}
        }
    }

    symbols
}

/// Format a parser Type for display in document symbols.
fn format_type(ty: &shadml_parser::parser::Type) -> String {
    use shadml_parser::parser::Type;
    match ty {
        Type::Con(name, _) => name.clone(),
        Type::Var(name, _) => name.clone(),
        Type::Nat(n, _) => n.to_string(),
        Type::App(_, _, _) => {
            // Collect all type application arguments: F a b c → F<a, b, c>
            let mut args = Vec::new();
            let mut head = ty;
            while let Type::App(f, arg, _) = head {
                args.push(format_type(arg));
                head = f;
            }
            args.reverse();
            format!("{}<{}>", format_type(head), args.join(", "))
        }
        Type::Arrow(from, to, _) => format!("{} -> {}", format_type(from), format_type(to)),
        Type::Paren(inner, _) => format!("({})", format_type(inner)),
        Type::Tuple(items, _) => {
            let parts: Vec<_> = items.iter().map(format_type).collect();
            format!("({})", parts.join(", "))
        }
        Type::Unit(_) => "()".to_string(),
    }
}

/// Find the range of a specific name within a span, for use as `selection_range`.
fn first_name_range(source: &str, name: &str, within: Span) -> Range {
    let tokens = lex(source);
    for tok in &tokens {
        if tok.span.start >= within.start
            && tok.span.end <= within.end
            && (tok.kind == SyntaxKind::Ident || tok.kind == SyntaxKind::UpperIdent)
            && tok.text(source) == name
        {
            return span_to_range(source, tok.span);
        }
    }
    span_to_range(source, within)
}

/// Find the range of a record field name within a constructor span.
fn find_field_name_range(source: &str, field_name: &str, within: Span) -> Option<Range> {
    let tokens = lex(source);
    let non_trivia: Vec<&Token> = tokens
        .iter()
        .filter(|t| t.span.start >= within.start && t.span.end <= within.end && !t.kind.is_trivia())
        .collect();
    for (i, tok) in non_trivia.iter().enumerate() {
        if tok.kind == SyntaxKind::Ident
            && tok.text(source) == field_name
            && i + 1 < non_trivia.len()
            && non_trivia[i + 1].kind == SyntaxKind::Colon
        {
            return Some(span_to_range(source, tok.span));
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
            _ if tok.kind.is_keyword() => Some(TOK_KEYWORD),

            // Numbers
            SyntaxKind::IntLiteral | SyntaxKind::FloatLiteral => Some(TOK_NUMBER),

            // Strings
            SyntaxKind::StringLiteral | SyntaxKind::CharLiteral => Some(TOK_STRING),

            // Comments
            SyntaxKind::LineComment | SyntaxKind::BlockComment | SyntaxKind::DocComment => {
                Some(TOK_COMMENT)
            }

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

/// Convert a shadml diagnostic to an LSP diagnostic.
fn shadml_diag_to_lsp(
    diag: &ShadmlDiag,
    source: &str,
    uri: Option<&Url>,
) -> Option<tower_lsp::lsp_types::Diagnostic> {
    let severity = match diag.severity {
        ShadmlSeverity::Error => DiagnosticSeverity::ERROR,
        ShadmlSeverity::Warning => DiagnosticSeverity::WARNING,
        ShadmlSeverity::Info => DiagnosticSeverity::INFORMATION,
        ShadmlSeverity::Hint => DiagnosticSeverity::HINT,
    };

    // Use the first label's span for the range, or fall back to start of file.
    let range = if let Some(label) = diag.labels.first() {
        span_to_range(source, label.span)
    } else {
        Range::new(Position::new(0, 0), Position::new(0, 0))
    };
    let related_information = uri.and_then(|uri| {
        let related = diag
            .labels
            .iter()
            .skip(1)
            .map(|label| DiagnosticRelatedInformation {
                location: Location {
                    uri: uri.clone(),
                    range: span_to_range(source, label.span),
                },
                message: label.message.clone(),
            })
            .collect::<Vec<_>>();

        if related.is_empty() {
            None
        } else {
            Some(related)
        }
    });
    let mut message = diag.message.clone();
    if let Some(label) = diag.labels.first() {
        if !label.message.is_empty() {
            message.push_str("\n\n");
            message.push_str(&label.message);
        }
    }
    if let Some(help) = &diag.help {
        message.push_str("\n\nhelp: ");
        message.push_str(help);
    }

    Some(tower_lsp::lsp_types::Diagnostic {
        range,
        severity: Some(severity),
        code: diag
            .code
            .as_ref()
            .map(|c| NumberOrString::String(c.clone())),
        source: Some("shadml".into()),
        message,
        related_information,
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

/// Convert a shadml Span to an LSP Range.
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
            "match", "let", "in", "if", "then", "else", "where", "data", "alias", "class",
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

        for ty in [
            "I32", "U32", "F32", "Bool", "Scalar", "Sca", "Tensor", "Ten", "Vector", "Vec",
            "Matrix", "Mat", "Option", "Result", "Pair",
        ] {
            assert!(labels.contains(&ty), "Missing type completion: {}", ty);
        }
    }

    #[test]
    fn test_builtin_function_completions_present() {
        let items = build_completions("", Position::new(0, 0));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();

        for f in [
            "vec2",
            "vec3",
            "vec4",
            "sin",
            "cos",
            "mix",
            "dot",
            "normalize",
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

        let sin_item = items.iter().find(|i| i.label == "sin").unwrap();
        assert_eq!(sin_item.kind, Some(CompletionItemKind::FUNCTION));
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

    #[test]
    fn test_attribute_completions_after_at() {
        let items = build_completions("@", Position::new(0, 1));
        let labels: Vec<&str> = items.iter().map(|item| item.label.as_str()).collect();

        assert!(labels.contains(&"compute"));
        assert!(labels.contains(&"workgroup_size"));
        assert!(!labels.contains(&"let"));
    }

    #[test]
    fn test_keyword_completion_uses_snippet_and_docs() {
        let items = build_completions("", Position::new(0, 0));
        let match_item = items.iter().find(|item| item.label == "match").unwrap();

        assert_eq!(
            match_item.insert_text_format,
            Some(InsertTextFormat::SNIPPET)
        );
        match &match_item.documentation {
            Some(Documentation::MarkupContent(markup)) => {
                assert!(markup.value.contains("Pattern matching expression"));
            }
            other => panic!("Expected markdown documentation, got {other:?}"),
        }
    }

    #[test]
    fn test_document_completion_includes_type_detail() {
        let source = "id x = x\nmain = id";
        let items = build_completions(source, Position::new(1, 9));
        let item = items
            .iter()
            .find(|candidate| candidate.label == "id")
            .unwrap();

        let detail = item.detail.as_deref().unwrap_or_default();
        assert!(detail.contains("binding"));
        assert!(detail.contains("->"));
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
        let source = "module where import data alias trait impl let in case of match if then else do forall infixl infixr infix deriving";
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
    fn test_hover_on_attribute_identifier() {
        let source = "@compute\nmain x = x";
        let hover = build_hover(source, Position::new(0, 1));
        assert!(hover.is_some());

        match hover.unwrap().contents {
            HoverContents::Markup(markup) => {
                assert!(markup.value.contains("compute"));
                assert!(markup.value.contains("compute shader entry point"));
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
        let uri = Url::parse("file:///test.shadml").unwrap();
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
        let uri = Url::parse("file:///test.shadml").unwrap();
        // Position on "=" sign
        let result = build_goto_definition(&uri, source, Position::new(0, 6));
        assert!(result.is_none());
    }

    // -- Diagnostics conversion tests ---------------------------------------

    #[test]
    fn test_shadml_diag_to_lsp_error() {
        let diag = ShadmlDiag::error("type mismatch")
            .with_label(shadml_diagnostics::Label::primary(Span::new(0, 5), "here"));
        let source = "hello world";
        let lsp = shadml_diag_to_lsp(&diag, source, None).unwrap();
        assert_eq!(lsp.severity, Some(DiagnosticSeverity::ERROR));
        assert!(lsp.message.contains("type mismatch"));
        assert!(lsp.message.contains("here"));
        assert_eq!(lsp.range.start.line, 0);
        assert_eq!(lsp.range.start.character, 0);
        assert_eq!(lsp.range.end.character, 5);
    }

    #[test]
    fn test_shadml_diag_to_lsp_warning() {
        let diag = ShadmlDiag::warning("unused variable");
        let source = "hello";
        let lsp = shadml_diag_to_lsp(&diag, source, None).unwrap();
        assert_eq!(lsp.severity, Some(DiagnosticSeverity::WARNING));
    }

    #[test]
    fn test_shadml_diag_to_lsp_with_code() {
        let diag = ShadmlDiag::error("oops").with_code("E001");
        let source = "";
        let lsp = shadml_diag_to_lsp(&diag, source, None).unwrap();
        assert_eq!(lsp.code, Some(NumberOrString::String("E001".into())));
        assert_eq!(lsp.source, Some("shadml".into()));
    }

    #[test]
    fn test_shadml_diag_to_lsp_no_labels() {
        let diag = ShadmlDiag::error("generic error");
        let source = "some source";
        let lsp = shadml_diag_to_lsp(&diag, source, None).unwrap();
        // Falls back to (0,0)-(0,0) range
        assert_eq!(lsp.range.start.line, 0);
        assert_eq!(lsp.range.start.character, 0);
    }

    #[test]
    fn test_shadml_diag_to_lsp_includes_help_and_related_info() {
        let diag = ShadmlDiag::error("type mismatch")
            .with_label(shadml_diagnostics::Label::primary(
                Span::new(0, 3),
                "expected I32",
            ))
            .with_label(shadml_diagnostics::Label::new(Span::new(4, 7), "found Bool"))
            .with_help("add a conversion or change the annotation");
        let source = "foo bar";
        let uri = Url::parse("file:///test.shadml").unwrap();
        let lsp = shadml_diag_to_lsp(&diag, source, Some(&uri)).unwrap();

        assert!(lsp.message.contains("help: add a conversion"));
        assert_eq!(lsp.related_information.as_ref().map(Vec::len), Some(1));
        let related = &lsp.related_information.unwrap()[0];
        assert_eq!(related.location.uri, uri);
        assert_eq!(related.message, "found Bool");
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
