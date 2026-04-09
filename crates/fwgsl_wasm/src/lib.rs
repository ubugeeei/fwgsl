use lsp_types::{
    CompletionItem, CompletionItemKind, Hover, HoverContents, MarkupContent, Position, Range, Url,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct CompileResult {
    wgsl: String,
    diagnostics: Vec<DiagnosticOutput>,
}

#[derive(Serialize)]
struct DiagnosticOutput {
    severity: String,
    message: String,
    code: Option<String>,
    help: Option<String>,
    note: Option<String>,
    line: usize,
    col: usize,
    #[serde(rename = "endLine")]
    end_line: usize,
    #[serde(rename = "endCol")]
    end_col: usize,
}

#[derive(Serialize)]
struct EditorCompletionOutput {
    label: String,
    kind: Option<String>,
    detail: Option<String>,
    documentation: Option<String>,
    #[serde(rename = "insertText")]
    insert_text: Option<String>,
    #[serde(rename = "insertTextFormat")]
    insert_text_format: Option<u32>,
    #[serde(rename = "sortText")]
    sort_text: Option<String>,
    #[serde(rename = "filterText")]
    filter_text: Option<String>,
}

#[derive(Serialize)]
struct EditorHoverOutput {
    range: Option<EditorRangeOutput>,
    contents: String,
}

#[derive(Serialize)]
struct EditorLocationOutput {
    uri: String,
    range: EditorRangeOutput,
}

#[derive(Serialize)]
struct EditorRangeOutput {
    #[serde(rename = "startLineNumber")]
    start_line_number: u32,
    #[serde(rename = "startColumn")]
    start_column: u32,
    #[serde(rename = "endLineNumber")]
    end_line_number: u32,
    #[serde(rename = "endColumn")]
    end_column: u32,
}

fn with_prelude(program: &mut fwgsl_parser::parser::Program) {
    let prelude = fwgsl_parser::prelude_program();
    let mut combined = prelude.decls.clone();
    combined.append(&mut program.decls);
    program.decls = combined;
}

#[wasm_bindgen]
pub fn compile(source: &str) -> String {
    let mut parser = fwgsl_parser::parser::Parser::new(source);
    let mut program = parser.parse_program();
    with_prelude(&mut program);

    let mut diagnostics = Vec::new();

    // Collect parse diagnostics
    for diag in parser.diagnostics().iter() {
        diagnostics.push(convert_diagnostic(diag, source));
    }

    // Semantic analysis
    let mut analyzer = fwgsl_semantic::SemanticAnalyzer::new();
    analyzer.analyze(&program);

    for diag in analyzer.diagnostics().iter() {
        diagnostics.push(convert_diagnostic(diag, source));
    }

    let wgsl = if diagnostics.iter().any(|d| d.severity == "error") {
        "// Compilation failed. See diagnostics.".to_string()
    } else {
        // AST -> HIR lowering
        let mut lowering = fwgsl_ast_lowering::AstLowering::new(&analyzer);
        let hir = lowering.lower_program(&program);

        if lowering.has_errors() {
            for diag in lowering.diagnostics().iter() {
                diagnostics.push(convert_diagnostic(diag, source));
            }
            "// HIR lowering failed.".to_string()
        } else {
            match fwgsl_mir::lower::lower_hir_to_mir(&hir) {
                Ok(mir) => {
                    let mir = fwgsl_mir::reachability::eliminate_dead_code(&mir);
                    fwgsl_wgsl_codegen::emit_wgsl(&mir)
                }
                Err(errors) => {
                    format!("// MIR lowering failed: {}", errors.join(", "))
                }
            }
        }
    };

    let result = CompileResult { wgsl, diagnostics };
    serde_json::to_string(&result)
        .unwrap_or_else(|_| r#"{"wgsl":"// Internal error","diagnostics":[]}"#.to_string())
}

#[wasm_bindgen]
pub fn parse_ast(source: &str) -> String {
    let mut parser = fwgsl_parser::parser::Parser::new(source);
    let program = parser.parse_program();
    format!("{:#?}", program)
}

#[wasm_bindgen]
pub fn format(source: &str) -> String {
    fwgsl_formatter::format_default(source)
}

#[wasm_bindgen]
pub fn get_diagnostics(source: &str) -> String {
    let mut parser = fwgsl_parser::parser::Parser::new(source);
    let mut program = parser.parse_program();
    with_prelude(&mut program);

    let mut diagnostics = Vec::new();

    for diag in parser.diagnostics().iter() {
        diagnostics.push(convert_diagnostic(diag, source));
    }

    let mut analyzer = fwgsl_semantic::SemanticAnalyzer::new();
    analyzer.analyze(&program);

    for diag in analyzer.diagnostics().iter() {
        diagnostics.push(convert_diagnostic(diag, source));
    }

    serde_json::to_string(&diagnostics).unwrap_or_else(|_| "[]".to_string())
}

#[wasm_bindgen]
pub fn editor_completions(source: &str, line: u32, column: u32) -> String {
    let items = fwgsl_ide::build_completions(source, editor_position(line, column))
        .into_iter()
        .map(editor_completion_output)
        .collect::<Vec<_>>();
    serde_json::to_string(&items).unwrap_or_else(|_| "[]".to_string())
}

#[wasm_bindgen]
pub fn editor_hover(source: &str, line: u32, column: u32) -> String {
    let hover =
        fwgsl_ide::build_hover(source, editor_position(line, column)).map(editor_hover_output);
    serde_json::to_string(&hover).unwrap_or_else(|_| "null".to_string())
}

#[wasm_bindgen]
pub fn editor_definition(source: &str, line: u32, column: u32) -> String {
    let uri = editor_uri();
    let locations =
        match fwgsl_ide::build_goto_definition(&uri, source, editor_position(line, column)) {
            Some(lsp_types::GotoDefinitionResponse::Scalar(location)) => {
                vec![editor_location_output(location)]
            }
            Some(lsp_types::GotoDefinitionResponse::Array(locations)) => {
                locations.into_iter().map(editor_location_output).collect()
            }
            Some(lsp_types::GotoDefinitionResponse::Link(locations)) => locations
                .into_iter()
                .map(editor_location_like_output)
                .collect(),
            None => Vec::new(),
        };
    serde_json::to_string(&locations).unwrap_or_else(|_| "[]".to_string())
}

#[wasm_bindgen]
pub fn editor_references(
    source: &str,
    line: u32,
    column: u32,
    include_declaration: bool,
) -> String {
    let uri = editor_uri();
    let locations = fwgsl_ide::build_references(
        &uri,
        source,
        editor_position(line, column),
        include_declaration,
    )
    .unwrap_or_default()
    .into_iter()
    .map(editor_location_output)
    .collect::<Vec<_>>();
    serde_json::to_string(&locations).unwrap_or_else(|_| "[]".to_string())
}

fn convert_diagnostic(diag: &fwgsl_diagnostics::Diagnostic, source: &str) -> DiagnosticOutput {
    let severity = match diag.severity {
        fwgsl_diagnostics::Severity::Error => "error",
        fwgsl_diagnostics::Severity::Warning => "warning",
        fwgsl_diagnostics::Severity::Info => "info",
        fwgsl_diagnostics::Severity::Hint => "hint",
    };

    let (line, col) = if let Some(label) = diag.labels.first() {
        offset_to_line_col(source, label.span.start as usize)
    } else {
        (1, 1)
    };

    let (end_line, end_col) = if let Some(label) = diag.labels.first() {
        offset_to_line_col(source, label.span.end as usize)
    } else {
        (line, col + 1)
    };

    DiagnosticOutput {
        severity: severity.to_string(),
        message: diag.message.clone(),
        code: diag.code.clone(),
        help: diag.help.clone(),
        note: diag.labels.first().map(|label| label.message.clone()),
        line,
        col,
        end_line,
        end_col,
    }
}

fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn editor_position(line: u32, column: u32) -> Position {
    Position::new(line.saturating_sub(1), column.saturating_sub(1))
}

fn editor_uri() -> Url {
    Url::parse("inmemory://fwgsl/playground.fwgsl").expect("static playground URI")
}

fn editor_completion_output(item: CompletionItem) -> EditorCompletionOutput {
    EditorCompletionOutput {
        label: item.label,
        kind: item.kind.map(editor_completion_kind),
        detail: item.detail,
        documentation: item.documentation.map(editor_documentation),
        insert_text: item.insert_text,
        insert_text_format: item.insert_text_format.map(editor_insert_text_format),
        sort_text: item.sort_text,
        filter_text: item.filter_text,
    }
}

fn editor_completion_kind(kind: CompletionItemKind) -> String {
    match kind {
        CompletionItemKind::TEXT => "Text",
        CompletionItemKind::METHOD => "Method",
        CompletionItemKind::FUNCTION => "Function",
        CompletionItemKind::CONSTRUCTOR => "Constructor",
        CompletionItemKind::FIELD => "Field",
        CompletionItemKind::VARIABLE => "Variable",
        CompletionItemKind::CLASS => "Class",
        CompletionItemKind::INTERFACE => "Interface",
        CompletionItemKind::MODULE => "Module",
        CompletionItemKind::PROPERTY => "Property",
        CompletionItemKind::UNIT => "Unit",
        CompletionItemKind::VALUE => "Value",
        CompletionItemKind::ENUM => "Enum",
        CompletionItemKind::KEYWORD => "Keyword",
        CompletionItemKind::SNIPPET => "Snippet",
        CompletionItemKind::COLOR => "Color",
        CompletionItemKind::FILE => "File",
        CompletionItemKind::REFERENCE => "Reference",
        CompletionItemKind::FOLDER => "Folder",
        CompletionItemKind::ENUM_MEMBER => "EnumMember",
        CompletionItemKind::CONSTANT => "Constant",
        CompletionItemKind::STRUCT => "Struct",
        CompletionItemKind::EVENT => "Event",
        CompletionItemKind::OPERATOR => "Operator",
        CompletionItemKind::TYPE_PARAMETER => "TypeParameter",
        _ => "Text",
    }
    .to_owned()
}

fn editor_insert_text_format(format: lsp_types::InsertTextFormat) -> u32 {
    match format {
        lsp_types::InsertTextFormat::PLAIN_TEXT => 1,
        lsp_types::InsertTextFormat::SNIPPET => 2,
        _ => 1,
    }
}

fn editor_documentation(documentation: lsp_types::Documentation) -> String {
    match documentation {
        lsp_types::Documentation::String(value) => value,
        lsp_types::Documentation::MarkupContent(MarkupContent { value, .. }) => value,
    }
}

fn editor_hover_output(hover: Hover) -> EditorHoverOutput {
    EditorHoverOutput {
        range: hover.range.map(editor_range_output),
        contents: match hover.contents {
            HoverContents::Scalar(lsp_types::MarkedString::String(value)) => value,
            HoverContents::Scalar(lsp_types::MarkedString::LanguageString(language_string)) => {
                format!(
                    "```{}\n{}\n```",
                    language_string.language, language_string.value
                )
            }
            HoverContents::Array(values) => values
                .into_iter()
                .map(|value| match value {
                    lsp_types::MarkedString::String(text) => text,
                    lsp_types::MarkedString::LanguageString(language_string) => {
                        format!(
                            "```{}\n{}\n```",
                            language_string.language, language_string.value
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            HoverContents::Markup(MarkupContent { value, .. }) => value,
        },
    }
}

fn editor_location_output(location: lsp_types::Location) -> EditorLocationOutput {
    EditorLocationOutput {
        uri: location.uri.to_string(),
        range: editor_range_output(location.range),
    }
}

fn editor_location_like_output(link: lsp_types::LocationLink) -> EditorLocationOutput {
    EditorLocationOutput {
        uri: link.target_uri.to_string(),
        range: editor_range_output(link.target_selection_range),
    }
}

fn editor_range_output(range: Range) -> EditorRangeOutput {
    EditorRangeOutput {
        start_line_number: range.start.line + 1,
        start_column: range.start.character + 1,
        end_line_number: range.end.line + 1,
        end_column: range.end.character + 1,
    }
}
