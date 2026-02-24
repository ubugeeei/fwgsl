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
    line: usize,
    col: usize,
    #[serde(rename = "endLine")]
    end_line: usize,
    #[serde(rename = "endCol")]
    end_col: usize,
}

#[wasm_bindgen]
pub fn compile(source: &str) -> String {
    let mut parser = fwgsl_parser::parser::Parser::new(source);
    let program = parser.parse_program();

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
            "// HIR lowering failed.".to_string()
        } else {
            match fwgsl_mir::lower::lower_hir_to_mir(&hir) {
                Ok(mir) => fwgsl_wgsl_codegen::emit_wgsl(&mir),
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
pub fn format(source: &str) -> String {
    // TODO: implement formatter
    source.to_string()
}

#[wasm_bindgen]
pub fn get_diagnostics(source: &str) -> String {
    let mut parser = fwgsl_parser::parser::Parser::new(source);
    let program = parser.parse_program();

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
